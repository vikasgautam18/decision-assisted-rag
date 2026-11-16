from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path

import httpx
from azure.identity.aio import AzureCliCredential

from app.resources import ExperimentResources
from clients.azure import AzureSession
from clients.rate import ModelAdmission
from evaluation.comparison import PublicAnswer
from evaluation.dataset import load_cases
from evaluation.experiment import (
    Approval,
    Evaluator,
    ExperimentLedger,
    Protocol,
    identity,
    schedule,
)
from evaluation.judge import GradingQuota, grade, grading_request
from evaluation.observed_run import frozen_inputs, materialize_cases
from evaluation.paired_pilot import common_evidence
from evaluation.run import ROOT, catalogue_at, persist
from pipelines.budget import ServiceError


def review_case(case, overlay: dict):
    guidance = overlay.get("cases", {}).get(case.id)
    if guidance is None:
        return case
    if not isinstance(guidance, dict) or not all(
        isinstance(guidance.get(key), str) and guidance[key].strip()
        for key in ("guidance", "review_reference")
    ):
        raise ValueError("review overlay needs explicit guidance and source review reference")
    return case.model_copy(
        update={
            "notes": case.notes
            + "\nSeparate pre-frozen reviewed guidance (not a gold rewrite): "
            + guidance["guidance"]
            + "\nReview reference: "
            + guidance["review_reference"],
        }
    )


def selected_records(protocol: Protocol, ledger: ExperimentLedger) -> list[tuple[int, dict]]:
    entries = ledger.records()
    initial = {row.observation_id: row for row in schedule(protocol)}
    captured = {entry["id"] for entry in entries if entry["kind"] == "application"}
    if not initial.keys() <= captured:
        raise ValueError("complete serving collection before starting paid grading")
    by_logical = {}
    for entry in entries:
        if entry["kind"] != "application":
            continue
        if entry["state"] != "recorded":
            raise ValueError("reconcile interrupted application attempts before grading")
        record = entry["record"]
        if record["status"] != "completed":
            continue
        logical = record["logical_request_id"]
        if logical in by_logical:
            raise ValueError("multiple completed responses for one logical observation")
        by_logical[logical] = record
    return [
        (row.ordinal, by_logical[row.logical_request_id])
        for row in initial.values()
        if row.logical_request_id in by_logical
    ]


def grading_inputs(record: dict, cases: dict, catalogue: dict, overlay: dict) -> tuple:
    case = review_case(cases[record["case_id"]], overlay)
    public = PublicAnswer.model_validate_json(json.dumps(record["public_answer"]))
    response = {
        "answer": public.answer,
        "citations": [item.model_dump() for item in public.citations],
        "figures": [{"key": key} for key in public.figure_keys],
    }
    return case, response, common_evidence(case, public, catalogue)


def evaluator_assignments(
    pool: tuple[Evaluator, ...], requests: list[tuple[int, dict, int]]
) -> list[list[dict]]:
    assigned: list[list[dict]] = [[] for _ in pool]
    for ordinal, record, estimate in requests:
        eligible = [
            index for index, evaluator in enumerate(pool)
            if estimate <= int(evaluator.tokens_per_minute * 0.8)
        ]
        if not eligible:
            raise ValueError(
                f"no approved evaluator fits {estimate} estimated tokens for "
                f"observation {record['observation_id']}; no grading dispatched"
            )
        assigned[eligible[ordinal % len(eligible)]].append(record)
    return assigned


async def grade_worker(
    evaluator: Evaluator,
    selected: list[dict],
    protocol: Protocol,
    ledger: ExperimentLedger,
    approval: Approval,
    directory: Path,
    cases: dict,
    catalogue: dict,
    resources: ExperimentResources,
    overlay: dict,
    *,
    retry_errors: bool,
) -> None:
    from clients.telemetry import observation, span

    quota = GradingQuota(
        ROOT / "data/evaluation/grading-quota.sqlite3",
        limits={"grading": approval.runtime_cumulative_limits["grading"]},
    )
    async with (
        AzureCliCredential() as credential,
        httpx.AsyncClient(timeout=httpx.Timeout(90, connect=30), trust_env=False) as client,
    ):
        session = AzureSession(
            resources,
            credential,
            client,
            deployment_capacities={
                evaluator.deployment: (
                    evaluator.tokens_per_minute,
                    evaluator.requests_per_minute,
                ),
            },
        )
        for record in selected:
            key = record["observation_id"]
            original = directory / "judgments" / f"{key}.json"
            previous = json.loads(original.read_text()) if original.exists() else None
            if retry_errors:
                if previous is None or previous.get("status") != "error":
                    continue
                attempt = 2
                target = directory / "judgments" / f"{key}-retry.json"
            else:
                attempt = 1
                target = original
            call_id = identity(protocol.experiment_id, "grading", key, attempt)
            if not ledger.reserve(call_id, "grading"):
                continue
            persist(
                target,
                {
                    "status": "reserved_not_captured",
                    "observation_id": key,
                    "grading_attempt": attempt,
                    "grading_call_id": call_id,
                    "evaluator_deployment": evaluator.deployment,
                },
            )
            case, response, evidence = grading_inputs(record, cases, catalogue, overlay)
            metadata = {
                "experiment_id": protocol.experiment_id,
                "observation_id": key,
                "phase": "grading",
                "grading_attempt": attempt,
                "grading_call_id": call_id,
            }
            with observation(directory / "runner-spans", metadata) as recorder:
                with span("evaluation.grade", deployment=evaluator.deployment) as attributes:
                    try:
                        result = await grade(
                            session,
                            quota,
                            case,
                            response,
                            evidence,
                            deployment=evaluator.deployment,
                        )
                    except (ServiceError, ValueError) as exc:
                        result = {
                            "status": "error",
                            "error_type": type(exc).__name__,
                            "usage": None,
                        }
                    else:
                        result["status"] = "completed"
                    attributes.update(usage=result.get("usage"), model=result.get("model"))
                    result.update(
                        observation_id=key,
                        grading_attempt=attempt,
                        grading_call_id=call_id,
                        evaluator_deployment=evaluator.deployment,
                        expected_behavior=(
                            "conflict"
                            if case.expected_behavior == "flag_conflict"
                            else case.expected_behavior
                        ),
                        grading_trace_id=recorder.trace_id,
                    )
                    persist(target, result)
                    ledger.complete(call_id, result)


async def execute_grading(
    protocol: Protocol,
    directory: Path,
    approval: Approval,
    ledger: ExperimentLedger,
    resources: ExperimentResources,
    overlay: dict,
    *,
    retry_errors: bool,
) -> None:
    if protocol.phase == "load":
        raise ValueError("load sweep has no paid grading")
    if protocol.phase == "pilot":
        # Synthetic instrumentation-only cases have no paid grading by protocol.
        original_ids = {case.id for case in load_cases()}
        selected_case_ids = tuple(key for key in protocol.case_ids if key in original_ids)
        cases = materialize_cases(
            protocol.model_copy(
                update={
                    "case_ids": selected_case_ids,
                    "extra_cases_sha256": None,
                }
            )
        )
    else:
        cases = materialize_cases(protocol)
    catalogue, _ = catalogue_at(ROOT)
    pool = protocol.evaluator_pool
    selected = [
        (ordinal, record)
        for ordinal, record in selected_records(protocol, ledger)
        if record["case_id"] in cases
    ]
    admission = ModelAdmission()
    requests = []
    for ordinal, record in selected:
        request = grading_request(*grading_inputs(record, cases, catalogue, overlay))
        estimate = admission.estimate(
            request, output_tokens=request["max_completion_tokens"]
        )
        requests.append((ordinal, record, estimate))
    assignments = evaluator_assignments(pool, requests)
    async with asyncio.TaskGroup() as group:
        for index, evaluator in enumerate(pool):
            group.create_task(
                grade_worker(
                    evaluator,
                    assignments[index],
                    protocol,
                    ledger,
                    approval,
                    directory,
                    cases,
                    catalogue,
                    resources,
                    overlay,
                    retry_errors=retry_errors,
                )
            )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explicit approved grading on frozen evaluator pool."
    )
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--rate-card", type=Path, required=True)
    parser.add_argument("--review-overlay", type=Path, required=True)
    parser.add_argument("--resources", type=Path, default=ROOT / "manifests/azure-resources.json")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if not args.execute_approved:
        parser.error("--execute-approved required; no paid calls made")
    protocol = frozen_inputs(
        args.directory, ROOT, args.rate_card, args.review_overlay, args.preflight
    )
    approval = Approval.model_validate_json(args.approval.read_bytes())
    ledger = ExperimentLedger(args.directory / "accounting.sqlite3", protocol, approval)
    resources = ExperimentResources.model_validate_json(args.resources.read_bytes())
    overlay = json.loads(args.review_overlay.read_text())
    asyncio.run(
        execute_grading(
            protocol,
            args.directory,
            approval,
            ledger,
            resources,
            overlay,
            retry_errors=args.retry_errors,
        )
    )


if __name__ == "__main__":
    main()
