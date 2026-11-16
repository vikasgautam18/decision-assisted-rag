from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from azure.identity.aio import AzureCliCredential

from app.provenance import validate_manifest
from app.resources import ExperimentResources
from clients.azure import AzureSession
from evaluation.comparison import PublicAnswer, capture_public_answer
from evaluation.dataset import load_cases
from evaluation.judge import GradingQuota, grade
from evaluation.paired_pilot import (
    baseline_snapshot,
    check_images,
    common_evidence,
    execute,
    pilot_report,
    public_checks,
)
from evaluation.run import ROOT, catalogue_at, check_server, fingerprint, persist
from ingestion.checkpoint import CheckpointQuotaExhausted
from pipelines.budget import ServiceError


def retry_needed(record: dict, grading: dict) -> tuple[bool, bool]:
    application = record["status"] != "completed"
    return application, application or grading.get("status") != "completed"


async def check_snapshot(manifest: dict) -> None:
    validate_manifest(ROOT)
    if (
        baseline_snapshot() != manifest["baseline"]
        or fingerprint(ROOT) != manifest["evaluation_files"]
        or hashlib.sha256((ROOT / "data/implementation-manifest.json").read_bytes()).hexdigest()
        != manifest["application_sha256"]
    ):
        raise ValueError("comparison snapshot changed; use a new run")
    async with httpx.AsyncClient(base_url=manifest["origins"]["F"], trust_env=False) as client:
        await check_server(client, manifest["application_sha256"])


def paired_results(report: dict) -> dict:
    by_id = {}
    for row in report["rows"]:
        by_id.setdefault(row["id"], {})[row["arm"]] = row
    counts = {"both_pass": 0, "A_only_pass": 0, "F_only_pass": 0, "neither_pass": 0}
    pairs = []
    for case_id, arms in by_id.items():
        a = arms["A"]["strict_provisional_pass"]
        f = arms["F"]["strict_provisional_pass"]
        category = (
            "both_pass"
            if a and f
            else "A_only_pass"
            if a
            else "F_only_pass"
            if f
            else "neither_pass"
        )
        counts[category] += 1
        pairs.append(
            {
                "id": case_id,
                "outcome": category,
                "grading_unknown_arms": [
                    arm
                    for arm, row in arms.items()
                    if row["status"] == "completed" and row["grading_status"] != "completed"
                ],
            }
        )
    return {
        "counts": counts,
        "pairs": pairs,
        "interpretation": "Descriptive paired development results, not a significance claim; "
        "unknown grading is not a confirmed semantic failure. Fact-family dependence "
        "is not modeled by these counts.",
    }


def save_review_queue(directory: Path, report: dict) -> None:
    pairs = paired_results(report)
    review_ids = {
        item["id"]
        for item in pairs["pairs"]
        if item["outcome"] != "both_pass"
        or int(hashlib.sha256(f"phase3-audit:{item['id']}".encode()).hexdigest()[:8], 16) % 10 == 0
    }
    questions = {case.id: case for case in load_cases()}
    queue_path = directory / "human-review-queue.json"
    existing = (
        {item["id"]: item for item in json.loads(queue_path.read_text())["items"]}
        if queue_path.exists()
        else {}
    )
    items = []
    for case_id in sorted(review_ids | existing.keys()):
        case = questions[case_id]
        arms = {}
        for arm in ("A", "F"):
            record = json.loads((directory / f"responses/{case_id}-{arm}.json").read_text())
            arms[arm] = {
                "answer": record.get("public_answer", {}).get("answer"),
                "application_status": record["status"],
                "scoring": next(
                    row for row in report["rows"] if row["id"] == case_id and row["arm"] == arm
                ),
            }
        old = existing.get(case_id, {})
        items.append(
            {
                "id": case_id,
                "question": case.question,
                "original_gold_answer": case.gold_answer,
                "gold_sources": [item.model_dump() for item in case.gold_sources],
                "arms": arms,
                "human_status": old.get("human_status", "pending"),
                "human_decision": old.get("human_decision"),
                "human_notes": old.get("human_notes"),
            }
        )
    persist(queue_path, {"status": "human_review_pending", "items": items})


async def retry_errors(source: Path) -> dict:
    manifest = json.loads((source / "manifest.json").read_text())
    await check_snapshot(manifest)
    target = source.with_name(source.name + "-retry1")
    target.mkdir(parents=True, exist_ok=True, mode=0o700)
    for folder in ("responses", "judgments"):
        (target / folder).mkdir(exist_ok=True, mode=0o700)
    policy = {
        "source_run": source.name,
        "policy": "One retry per application or grader operational failure on either arm; "
        "no semantic-failure resampling; all original attempts retained.",
        "maximum_extra_application_requests": len(manifest["schedule"]),
        "maximum_extra_grading_requests": len(manifest["schedule"]),
        "source_manifest_sha256": hashlib.sha256(
            (source / "manifest.json").read_bytes()
        ).hexdigest(),
    }
    policy_path = target / "retry-provenance.json"
    if policy_path.exists():
        if json.loads(policy_path.read_text()) != policy:
            raise ValueError("retry configuration changed")
    else:
        persist(policy_path, policy)
        persist(target / "manifest.json", manifest)
    cases = {case.id: case for case in load_cases()}
    catalogue, figures = catalogue_at(ROOT)
    for case_id, arm in manifest["schedule"]:
        name = f"{case_id}-{arm}.json"
        destination = target / "responses" / name
        if destination.exists():
            continue
        original = json.loads((source / "responses" / name).read_text())
        if not retry_needed(original, {})[0]:
            persist(destination, {**original, "reused_from": source.name})
            continue
        persist(destination, {"status": "error", "error_type": "interrupted_or_not_captured"})
        record = await capture_public_answer(
            arm,
            manifest["origins"][arm],
            cases[case_id],
            timeout_seconds=manifest["timeouts"][arm],
        )
        persist(destination, record)
        if record["status"] == "completed":
            record["images"] = await check_images(
                arm, manifest["origins"][arm], record, cases[case_id], figures
            )
        record["public_checks"] = public_checks(cases[case_id], record)
        record["retry_of"] = f"{source.name}/responses/{name}"
        persist(destination, record)
        print("retry", case_id, arm, record["status"], flush=True)
    resources = ExperimentResources.model_validate_json(
        (ROOT / "manifests/azure-resources.json").read_bytes()
    )
    async with (
        AzureCliCredential() as credential,
        httpx.AsyncClient(timeout=httpx.Timeout(150, connect=30), trust_env=False) as client,
    ):
        session = AzureSession(resources, credential, client)
        quota = GradingQuota(ROOT / "data/evaluation/grading-quota.sqlite3")
        for case_id, arm in manifest["schedule"]:
            name = f"{case_id}-{arm}.json"
            destination = target / "judgments" / name
            if destination.exists():
                continue
            record = json.loads((target / "responses" / name).read_text())
            if record["status"] != "completed":
                continue
            old_path = source / "judgments" / name
            original = json.loads(old_path.read_text()) if old_path.exists() else {}
            if record.get("reused_from") and not retry_needed(record, original)[1]:
                persist(destination, {**original, "reused_from": source.name})
                continue
            persist(destination, {"status": "error", "error_type": "interrupted_or_not_captured"})
            answer = PublicAnswer.model_validate_json(json.dumps(record["public_answer"]))
            try:
                result = await grade(
                    session,
                    quota,
                    cases[case_id],
                    {
                        "answer": answer.answer,
                        "citations": [cite.model_dump() for cite in answer.citations],
                    },
                    common_evidence(cases[case_id], answer, catalogue),
                )
            except (ServiceError, ValueError, CheckpointQuotaExhausted) as exc:
                result = {"status": "error", "error_type": type(exc).__name__}
            else:
                result["status"] = "completed"
            persist(destination, result)
            print("grade retry", case_id, arm, result["status"], flush=True)
    await check_snapshot(manifest)
    report = pilot_report(target, list(cases.values()))
    report.update(
        status="retry_adjusted_exploratory_comparison_human_review_pending",
        retry_policy=policy,
        first_attempt_by_arm=json.loads((source / "report.json").read_text())["by_arm"],
        paired=paired_results(report),
        latency_policy="Per-attempt latencies remain in raw records. Retry-selected latencies "
        "exclude the original failed attempt and must not be presented as total task latency.",
    )
    persist(target / "report.json", report)
    save_review_queue(target, report)
    return report


async def run(args) -> None:
    args.full_suite = True
    await execute(args)
    source = ROOT / "data/comparison" / args.run_id
    report = json.loads((source / "report.json").read_text())
    report["paired"] = paired_results(report)
    persist(source / "report.json", report)
    save_review_queue(source, report)
    final = await retry_errors(source)
    print("Final retry-adjusted counts:", json.dumps(final["by_arm"]), flush=True)
    print("Reports:", source, source.with_name(source.name + "-retry1"), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Approved 72-case paired comparison and one retry."
    )
    parser.add_argument("--execute-approved-comparison", required=True, action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8001")
    parser.add_argument("--app-url", default="http://127.0.0.1:8765")
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
