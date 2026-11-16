from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import time
from datetime import UTC, datetime
from pathlib import Path

import httpx

from evaluation.comparison import capture_public_answer
from evaluation.dataset import Case, load_cases
from evaluation.experiment import (
    Approval,
    ExperimentLedger,
    Observation,
    Protocol,
    digest,
    identity,
    schedule,
    schedule_audit,
)
from evaluation.paired_pilot import check_images, public_checks
from evaluation.run import ROOT, catalogue_at, persist


def source_files(root: Path) -> dict[str, Path]:
    files = {
        str(path.relative_to(root)): path
        for folder in ("app", "clients", "pipelines", "ingestion", "evaluation")
        for path in sorted((root / folder).rglob("*"))
        if path.is_file()
        and path.suffix in (".py", ".html", ".jsonl")
        and "__pycache__" not in path.parts
    }
    files["pyproject.toml"] = root / "pyproject.toml"
    return files


def code_hash(root: Path) -> str:
    return digest(
        {
            name: hashlib.sha256(path.read_bytes()).hexdigest()
            for name, path in source_files(root).items()
        }
    )


def dataset_hash() -> str:
    return digest([case.model_dump(mode="json") for case in load_cases()])


def prepare(directory: Path, protocol: Protocol, root: Path = ROOT) -> None:
    if directory.exists():
        raise ValueError("experiment directory exists; frozen runs are not overwritten")
    if code_hash(root) != protocol.code_sha256:
        raise ValueError("source changed before preparation; freeze a matching protocol")
    directory.mkdir(parents=True, mode=0o700)
    for folder in ("responses", "judgments", "private", "runner-spans", "reviews"):
        (directory / folder).mkdir(mode=0o700)
    persist(directory / "protocol.json", protocol.model_dump(mode="json"))
    files = source_files(root)
    files.update(
        {str(path.relative_to(root)): path for path in (root / "manifests").glob("*.json")}
    )
    if any(
        path.is_symlink() or not path.resolve().is_relative_to(root.resolve())
        for path in files.values()
    ):
        raise ValueError("source snapshot must not follow symlinks or leave the project")
    for name, path in files.items():
        target = directory / "source" / name
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        target.write_bytes(path.read_bytes())
        target.chmod(0o600)
    snapshot = directory / "source"
    snapshot.chmod(0o700)
    for folder in snapshot.rglob("*"):
        if folder.is_dir():
            folder.chmod(0o700)
    if code_hash(directory / "source") != protocol.code_sha256:
        raise ValueError("source changed while archiving; preparation is not authorized")
    rows = schedule(protocol)
    persist(
        directory / "schedule.json",
        {
            "protocol_sha256": protocol.sha256,
            "rows": [row.model_dump(mode="json") for row in rows],
            "audit": schedule_audit(rows),
        },
    )
    persist(
        directory / "status.json",
        {
            "status": "prepared_not_authorized",
            "initial_application_requests": len(rows),
            "paid_calls_made": 0,
            "source_snapshot_complete": True,
            "cost_scope": protocol.cost_scope.report(),
            "approval_required": True,
            "budget_note": "No new model or infrastructure quotas authorized by preparation.",
        },
    )


def frozen_inputs(
    directory: Path, root: Path, rate_card: Path, overlay: Path, gate: Path
) -> Protocol:
    protocol = Protocol.model_validate_json((directory / "protocol.json").read_bytes())
    if (
        code_hash(root) != protocol.code_sha256
        or code_hash(directory / "source") != protocol.code_sha256
        or dataset_hash() != protocol.dataset_sha256
        or hashlib.sha256(rate_card.read_bytes()).hexdigest() != protocol.rate_card_sha256
        or hashlib.sha256(overlay.read_bytes()).hexdigest() != protocol.review_overlay_sha256
    ):
        raise ValueError("code, dataset, prices or review overlay drifted after freeze")
    evidence = json.loads(gate.read_text())
    checks = (
        "offline_tests_passed",
        "baseline_logging_parity_passed",
        "baseline_visibility_accepted",
        "rates_and_meter_semantics_verified",
        "model_versions_and_quotas_verified",
        "runtime_cumulative_caps_explicitly_approved",
        "review_overlay_frozen",
    )
    if evidence.get("protocol_sha256") != protocol.sha256 or any(
        evidence.get(key) is not True for key in checks
    ):
        raise ValueError("preflight evidence incomplete or belongs to a different protocol")
    if protocol.phase != "pilot" and evidence.get("instrumentation_pilot_accepted") is not True:
        raise ValueError("main/load execution requires accepted instrumentation pilot")
    stored = json.loads((directory / "schedule.json").read_text())
    expected = [row.model_dump(mode="json") for row in schedule(protocol)]
    if stored.get("protocol_sha256") != protocol.sha256 or stored.get("rows") != expected:
        raise ValueError("frozen schedule changed")
    return protocol


async def check_target(protocol: Protocol, arm: str, transport=None) -> None:
    target = protocol.targets[arm]
    async with httpx.AsyncClient(
        base_url=target.origin, timeout=30, trust_env=False, transport=transport
    ) as client:
        response = await client.get("/health")
        response.raise_for_status()
        state = response.json()
    if (
        state.get("implementation_sha256") != target.application_sha256
        or state.get("arm") != arm
        or state.get("experiment_id") != protocol.experiment_id
    ):
        raise ValueError("server snapshot, immutable arm or experiment does not match protocol")


def materialize_cases(protocol: Protocol, extras: list[Case] | None = None) -> dict[str, Case]:
    originals = {case.id: case for case in load_cases()}
    if protocol.phase == "main" and set(protocol.case_ids) != set(originals):
        raise ValueError("main experiment must use exactly the unchanged 72 case IDs")
    cases = dict(originals)
    if extras:
        if digest([case.model_dump(mode="json") for case in extras]) != (
            protocol.extra_cases_sha256
        ):
            raise ValueError("synthetic question text is not part of the frozen protocol")
    elif protocol.extra_cases_sha256 is not None:
        raise ValueError("frozen synthetic cases must be supplied")
    for case in extras or []:
        if case.id in cases or protocol.phase != "pilot":
            raise ValueError("synthetic cases must be new IDs in the pilot only")
        cases[case.id] = case
    if not set(protocol.case_ids) <= cases.keys():
        raise ValueError("schedule contains an unknown question")
    return {key: cases[key] for key in protocol.case_ids}


async def collect_one(
    row: Observation,
    case: Case,
    protocol: Protocol,
    directory: Path,
    ledger: ExperimentLedger,
    figure_pages: dict,
    root: Path = ROOT,
    transport=None,
) -> None:
    from clients.telemetry import observation, span

    await check_target(protocol, row.arm, transport)
    # Even completed or interrupted existing intents cannot be submitted again.
    # A deliberate external retry gets a new attempt identity.
    if not ledger.reserve(row.observation_id, "application"):
        return
    metadata = row.model_dump(mode="json")
    persist(
        directory / "responses" / f"{row.observation_id}.json",
        {
            **metadata,
            "status": "reserved_not_captured",
            "dispatched": False,
        },
    )
    with observation(directory / "runner-spans", metadata) as recorder:
        with span("runner.application", target_concurrency=row.concurrency):
            started = datetime.now(UTC).isoformat()
            clock = time.monotonic()
            record = await capture_public_answer(
                row.arm,
                protocol.targets[row.arm].origin,
                case,
                timeout_seconds=protocol.targets[row.arm].timeout_seconds,
                correlation_headers=row.headers,
                transport=transport,
            )
            record.update(
                **metadata,
                dispatched=True,
                started_utc=started,
                completed_utc=datetime.now(UTC).isoformat(),
                runner_attempt_seconds=time.monotonic() - clock,
                runner_trace_id=recorder.trace_id,
                placement=protocol.targets[row.arm].placement,
            )
            correlation = record.get("correlation", {})
            if correlation.get("x-observation-id") != row.observation_id or not re.fullmatch(
                r"[a-f0-9]{32}", correlation.get("x-experiment-trace-id") or ""
            ):
                # Preserve the actual result and paid attempt before stopping.
                record["telemetry_error"] = "missing_or_wrong_exact_server_correlation"
            if record["status"] == "completed":
                record["images"] = (
                    await check_images(
                        row.arm, protocol.targets[row.arm].origin, record, case, figure_pages
                    )
                    if json.loads(record["raw_body"]).get("figures")
                    else []
                )
                record["public_checks"] = public_checks(case, record)
                if row.arm != "A":
                    payload = json.loads(record["raw_body"])
                    trace_id = payload.get("trace_id", "")
                    if not re.fullmatch(r"[a-f0-9]{32}", trace_id):
                        record["telemetry_error"] = "missing_private_pipeline_trace_id"
                    else:
                        path = (
                            root
                            / "data/review/experiments"
                            / protocol.experiment_id
                            / row.arm
                            / "traces"
                            / f"{trace_id}.json"
                        )
                        if path.exists():
                            private = json.loads(path.read_text())
                            if private.get("trace_id") != trace_id:
                                raise ValueError("private trace identity mismatch")
                            persist(directory / "private" / f"{row.observation_id}.json", private)
                        else:
                            record["telemetry_error"] = "private_pipeline_trace_missing"
            persist(directory / "responses" / f"{row.observation_id}.json", record)
            ledger.complete(row.observation_id, record)
            if "telemetry_error" in record:
                raise ValueError(record["telemetry_error"])
            await check_target(protocol, row.arm, transport)


def retry_rows(rows: list[Observation], ledger: ExperimentLedger) -> list[Observation]:
    records = {record["id"]: record for record in ledger.records()}
    result = []
    for row in rows:
        prior = records.get(row.observation_id)
        if not prior:
            raise ValueError("initial collection is incomplete; no retry phase yet")
        if prior["state"] != "recorded":
            raise ValueError("uncertain dispatch requires reconciliation, not automatic retry")
        record = prior["record"]
        if "telemetry_error" in record:
            raise ValueError("telemetry defect requires a new experiment, not resampling")
        if record["status"] != "completed":
            result.append(
                row.model_copy(
                    update={
                        "attempt": 2,
                        "observation_id": identity(row.logical_request_id, 2),
                    }
                )
            )
    return result


async def execute(
    protocol: Protocol,
    directory: Path,
    ledger: ExperimentLedger,
    cases: dict[str, Case],
    *,
    retries: bool = False,
) -> None:
    _, figures = catalogue_at(ROOT)
    rows = schedule(protocol)
    if retries:
        if protocol.phase == "load":
            raise ValueError("load sweep has no application retries")
        rows = retry_rows(rows, ledger)
    if protocol.phase != "load":
        for row in rows:
            await collect_one(row, cases[row.case_id], protocol, directory, ledger, figures)
    else:
        # A load cell contains one arm at one concurrency, never competing arms.
        for level in protocol.concurrency_levels:
            arms = list(dict.fromkeys(row.arm for row in rows if row.concurrency == level))
            for arm in arms:
                cell = [row for row in rows if row.arm == arm and row.concurrency == level]
                queue = asyncio.Queue()
                for row in cell:
                    queue.put_nowait(row)

                async def worker(pending):
                    while not pending.empty():
                        row = pending.get_nowait()
                        await collect_one(
                            row, cases[row.case_id], protocol, directory, ledger, figures
                        )

                async with asyncio.TaskGroup() as group:
                    for _ in range(level):
                        group.create_task(worker(queue))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare without spending; execute only approved runs."
    )
    parser.add_argument("operation", choices=("prepare", "collect", "retry"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument("--rate-card", type=Path)
    parser.add_argument("--review-overlay", type=Path)
    parser.add_argument("--extra-cases", type=Path)
    parser.add_argument("--execute-approved", action="store_true")
    args = parser.parse_args()
    if args.operation == "prepare":
        if args.protocol is None:
            parser.error("prepare requires --protocol")
        prepare(args.directory, Protocol.model_validate_json(args.protocol.read_bytes()))
        return
    if not args.execute_approved or any(
        value is None
        for value in (args.approval, args.preflight, args.rate_card, args.review_overlay)
    ):
        parser.error("paid execution requires explicit switch, approval and all frozen gate inputs")
    protocol = frozen_inputs(
        args.directory, ROOT, args.rate_card, args.review_overlay, args.preflight
    )
    approval = Approval.model_validate_json(args.approval.read_bytes())
    ledger = ExperimentLedger(args.directory / "accounting.sqlite3", protocol, approval)
    extras = (
        [
            Case.model_validate_json(json.dumps(item))
            for item in json.loads(args.extra_cases.read_text())
        ]
        if args.extra_cases
        else None
    )
    cases = materialize_cases(protocol, extras)
    asyncio.run(execute(protocol, args.directory, ledger, cases, retries=args.operation == "retry"))


if __name__ == "__main__":
    main()
