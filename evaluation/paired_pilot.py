from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import subprocess

import httpx
from azure.identity.aio import AzureCliCredential

from app.provenance import code_paths, validate_manifest
from app.resources import ExperimentResources
from clients.azure import AzureSession
from evaluation.baseline_copy import REFERENCE_ENV, reference_checkout
from evaluation.comparison import (
    PublicAnswer,
    capture_public_answer,
    paired_schedule,
    validate_origin,
)
from evaluation.dataset import audit, load_cases
from evaluation.judge import GradingQuota, grade
from evaluation.run import ROOT, catalogue_at, check_server, fingerprint, persist
from ingestion.checkpoint import CheckpointQuotaExhausted
from ingestion.corpus import DOSSIERS, REFERENCE_COMMIT
from pipelines.budget import ServiceError

PILOT_CASES = ("q01", "q28", "q39", "q58")


def baseline_snapshot() -> dict:
    reference = reference_checkout()
    if reference is None:
        raise ValueError(f"{REFERENCE_ENV} must point to a clean checkout at the pinned revision")

    def git(*args):
        return subprocess.check_output(["git", "-C", str(reference), *args])

    commit = git("rev-parse", "HEAD").decode().strip()
    if commit != REFERENCE_COMMIT or git("status", "--porcelain").strip():
        raise ValueError("reference must be clean at the pinned revision")
    paths = git("ls-files", "-z").decode().split("\0")
    return {
        "commit": commit,
        "tracked_files": {
            name: hashlib.sha256((reference / name).read_bytes()).hexdigest()
            for name in paths
            if name
        },
        "deployment_revision_proof": "local process launched from pinned clean checkout",
        "internal_calls_usage_and_cost": "unknown; native retries unchanged",
    }


async def check_images(arm, origin, record, case, figure_pages):
    results = []
    payload = json.loads(record["raw_body"])
    gold = {(item.doc, item.page) for item in case.gold_sources}
    async with httpx.AsyncClient(
        base_url=origin, timeout=30, trust_env=False, follow_redirects=False
    ) as client:
        for item in payload["figures"]:
            key = item["key"]
            location = (item.get("doc"), item.get("page")) if arm == "A" else figure_pages.get(key)
            result = {"key": key, "location": location, "matches_gold_page": location in gold}
            valid_location = (
                location is not None
                and location[0] in DOSSIERS
                and type(location[1]) is int
                and 1 <= location[1] <= DOSSIERS[location[0]][1]
            )
            try:
                image = await client.get(item["url"])
            except httpx.RequestError as exc:
                result.update(delivered_png=False, error_type=type(exc).__name__)
            else:
                result.update(
                    delivered_png=image.status_code == 200
                    and image.content.startswith(b"\x89PNG\r\n\x1a\n"),
                    sha256=hashlib.sha256(image.content).hexdigest(),
                    byte_count=len(image.content),
                    http_status=image.status_code,
                )
            result["valid_location"] = valid_location
            results.append(result)
    return results


def public_checks(case, record):
    if record["status"] != "completed":
        return {"completed": False}
    answer = PublicAnswer.model_validate_json(json.dumps(record["public_answer"]))
    pages = {
        (cite.document, page)
        for cite in answer.citations
        for page in range(cite.page_start, cite.page_end + 1)
    }
    gold = {(item.doc, item.page) for item in case.gold_sources}
    factual = case.expected_behavior in ("answer", "flag_conflict")
    images = record.get("images", [])
    return {
        "completed": True,
        "citation_present": not factual or bool(answer.citations),
        "gold_page_match": not factual or bool(pages & gold),
        "png_delivery": all(item["delivered_png"] for item in images),
        "figure_metadata": all(item["valid_location"] for item in images),
        "required_figure": not case.requires_figure
        or any(item["matches_gold_page"] and item["delivered_png"] for item in images),
        "no_wrong_figure": all(item["matches_gold_page"] for item in images) if gold else True,
    }


def common_evidence(case, answer, catalogue):
    return [
        item
        for item in catalogue.values()
        if any(
            item["document"] == cite.document
            and item["page_start"] <= cite.page_end
            and item["page_end"] >= cite.page_start
            for cite in answer.citations
        )
        or any(
            item["document"] == source.doc and item["page_start"] <= source.page <= item["page_end"]
            for source in case.gold_sources
        )
    ]


def pilot_report(directory, cases):
    rows = []
    for case in cases:
        for arm in ("A", "F"):
            record = json.loads((directory / f"responses/{case.id}-{arm}.json").read_text())
            grading_path = directory / f"judgments/{case.id}-{arm}.json"
            grading = json.loads(grading_path.read_text()) if grading_path.exists() else {}
            judgment = grading.get("judgment")
            expected = (
                "conflict" if case.expected_behavior == "flag_conflict" else case.expected_behavior
            )
            passed = (
                all(record.get("public_checks", {"completed": False}).values())
                and judgment is not None
                and judgment["behavior"] == expected
                and all(
                    judgment[key]
                    for key in ("correct", "complete", "claims_supported", "citations_supported")
                )
                and not judgment["gold_issue"]
                and not judgment["uncertain"]
            )
            rows.append(
                {
                    "id": case.id,
                    "arm": arm,
                    "status": record["status"],
                    "public_checks": record.get("public_checks"),
                    "application_seconds": record.get("application_seconds"),
                    "grading_status": grading.get("status", "not_run"),
                    "judgment": judgment,
                    "strict_provisional_pass": bool(passed),
                }
            )
    report = {
        "status": "exploratory_paired_comparison_human_review_pending",
        "cases_per_arm": len(cases),
        "rows": rows,
        "by_arm": {
            arm: {
                "application_completed": sum(
                    row["status"] == "completed" for row in rows if row["arm"] == arm
                ),
                "strict_provisional_passes": sum(
                    row["strict_provisional_pass"] for row in rows if row["arm"] == arm
                ),
            }
            for arm in ("A", "F")
        },
        "limitations": [
            "Exposed development cases; no confirmatory superiority or stability claim.",
            "Native retries and deadlines differ; no harness retries.",
            "PNG delivery and metadata are checked, not original-image semantic identity.",
            "Common documentary evidence comes from the new ingestion, "
            "not baseline private context.",
            "Private runtime verdicts are excluded from common scoring for both arms.",
            "Runtime usage/cost unknown for public responses; not zero.",
            "No phase-2 human adjudication is silently applied to either arm.",
        ],
    }
    persist(directory / "report.json", report)
    return report


async def execute(args):
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("invalid pilot run ID")
    origins = {"A": validate_origin(args.baseline_url), "F": validate_origin(args.app_url)}
    if origins["A"] == origins["F"]:
        raise ValueError("comparison arms must have distinct origins")
    validate_manifest(ROOT)
    full_suite = getattr(args, "full_suite", False)
    cases = [case for case in load_cases() if full_suite or case.id in PILOT_CASES]
    schedule = paired_schedule(cases, seed=20261103)
    frozen = {
        "baseline": baseline_snapshot(),
        "origins": origins,
        "application_sha256": hashlib.sha256(
            (ROOT / "data/implementation-manifest.json").read_bytes()
        ).hexdigest(),
        "evaluation_files": fingerprint(ROOT),
        "audit": audit(),
        "schedule": schedule,
        "budgets": {
            "application_requests": 2 * len(cases),
            "grading_requests": 2 * len(cases),
        },
        "timeouts": {"A": 420, "F": 195},
        "F_internal_deadline_seconds": 180,
    }
    directory = ROOT / "data/comparison" / args.run_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        if json.loads(manifest_path.read_text()) != json.loads(json.dumps(frozen)):
            raise ValueError("pilot configuration changed; use a new run ID")
    else:
        persist(manifest_path, frozen)
        for name in ("responses", "judgments"):
            (directory / name).mkdir(mode=0o700)
        paths = code_paths(ROOT) + list((ROOT / "evaluation").rglob("*.py"))
        paths += [ROOT / "data/implementation-manifest.json"]
        for path in paths:
            target = directory / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    catalogue, figure_pages = catalogue_at(ROOT)
    async with httpx.AsyncClient(base_url=origins["F"], trust_env=False) as client:
        await check_server(client, frozen["application_sha256"])
    async with httpx.AsyncClient(base_url=origins["A"], trust_env=False) as client:
        config = await client.get("/api/config")
        config.raise_for_status()
        if config.json().get("kb") != "dossier-kb":
            raise ValueError("unexpected baseline knowledge base")
    by_id = {case.id: case for case in cases}
    for case_id, arm in schedule:
        target = directory / f"responses/{case_id}-{arm}.json"
        if target.exists():
            continue
        persist(target, {"status": "error", "error_type": "interrupted_or_not_captured"})
        record = await capture_public_answer(
            arm, origins[arm], by_id[case_id], timeout_seconds=frozen["timeouts"][arm]
        )
        persist(target, record)
        if record["status"] == "completed":
            record["images"] = await check_images(
                arm, origins[arm], record, by_id[case_id], figure_pages
            )
        record["public_checks"] = public_checks(by_id[case_id], record)
        persist(target, record)
        print(case_id, arm, record["status"], flush=True)
    resources = ExperimentResources.model_validate_json(
        (ROOT / "manifests/azure-resources.json").read_bytes()
    )
    async with (
        AzureCliCredential() as credential,
        httpx.AsyncClient(timeout=httpx.Timeout(150, connect=30), trust_env=False) as client,
    ):
        session = AzureSession(resources, credential, client)
        quota = GradingQuota(ROOT / "data/evaluation/grading-quota.sqlite3")
        for case_id, arm in schedule:
            target = directory / f"judgments/{case_id}-{arm}.json"
            record = json.loads((directory / f"responses/{case_id}-{arm}.json").read_text())
            if target.exists() or record["status"] != "completed":
                continue
            persist(target, {"status": "error", "error_type": "interrupted_or_not_captured"})
            answer = PublicAnswer.model_validate_json(json.dumps(record["public_answer"]))
            try:
                result = await grade(
                    session,
                    quota,
                    by_id[case_id],
                    {
                        "answer": answer.answer,
                        "citations": [cite.model_dump() for cite in answer.citations],
                    },
                    common_evidence(by_id[case_id], answer, catalogue),
                )
            except (ServiceError, ValueError, CheckpointQuotaExhausted) as exc:
                result = {"status": "error", "error_type": type(exc).__name__}
            else:
                result["status"] = "completed"
            persist(target, result)
            print("grade", case_id, arm, result["status"], flush=True)
    if baseline_snapshot() != frozen["baseline"] or fingerprint(ROOT) != frozen["evaluation_files"]:
        raise ValueError("comparison code changed during pilot")
    validate_manifest(ROOT)
    async with httpx.AsyncClient(base_url=origins["F"], trust_env=False) as client:
        await check_server(client, frozen["application_sha256"])
    report = pilot_report(directory, cases)
    print(json.dumps(report["by_arm"], indent=2), flush=True)
    print("Report:", directory / "report.json", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Authorized four-case paired A/F pilot only.")
    parser.add_argument("--execute-approved-pilot", required=True, action="store_true")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--baseline-url", default="http://127.0.0.1:8001")
    parser.add_argument("--app-url", default="http://127.0.0.1:8765")
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
