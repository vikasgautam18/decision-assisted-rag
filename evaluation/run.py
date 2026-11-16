from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import re
import time
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from azure.identity.aio import AzureCliCredential

from app.provenance import code_paths, dossier_directory, validate_manifest
from app.resources import ExperimentResources
from clients.azure import AzureSession
from evaluation.dataset import DATA, Case, audit, load_cases
from evaluation.judge import DEPLOYMENT, MODEL, SYSTEM, VERSION, GradingQuota, grade
from evaluation.scoring import deterministic, page_evidence, trace_checks
from ingestion.corpus import DOSSIERS
from ingestion.layout import LayoutArtifact
from pipelines.budget import ServiceError

ROOT = Path(__file__).resolve().parents[1]


def persist(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with os.fdopen(os.open(temporary, os.O_CREAT | os.O_TRUNC | os.O_WRONLY, 0o600), "w") as out:
        json.dump(payload, out, indent=2)
        out.write("\n")
        out.flush()
        os.fsync(out.fileno())
    temporary.replace(path)


def fingerprint(root: Path) -> dict:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted((root / "evaluation").rglob("*"))
        if path.suffix in (".py", ".jsonl") and "__pycache__" not in path.parts
    }


def catalogue_at(root: Path) -> tuple[dict, dict]:
    catalogue, figures = {}, {}
    for document in DOSSIERS:
        records = json.loads((dossier_directory(root, document) / "chunks.json").read_text())
        catalogue.update(
            {
                item["id"]: {key: value for key, value in item.items() if key != "vector"}
                for item in records["chunks"]
            }
        )
        layout = LayoutArtifact.model_validate_json(
            (root / f"data/live-checkpoint/{document}.layout.json").read_bytes()
        )
        figures.update(
            {item.figure_id: (document, item.regions[0].page_number) for item in layout.figures}
        )
    return catalogue, figures


def stable_manifest(root: Path, url: str, workers: int, selected: list[str] | None = None) -> dict:
    validate_manifest(root)
    snapshot = (root / "data/implementation-manifest.json").read_bytes()
    return {
        "application_sha256": hashlib.sha256(snapshot).hexdigest(),
        "evaluation_files": fingerprint(root),
        "audit": audit(),
        "api_url": url,
        "protocol": {
            "kind": "exposed-development-regression-not-confirmatory",
            "selected_case_ids": selected or [case.id for case in load_cases()],
            "application_retries": 0,
            "grader_retries": 0,
            "workers": workers,
            "admission": "single-process per-deployment token/request windows, 80% quota, "
            "65-second window, o200k/character estimate + output reservation; "
            "429 cooldown without automatic inference retries",
            "cookie_isolation": "fresh client per question; no reused conversations",
            "application_seconds": "HTTP chat only; excludes grading and artifact retrieval",
            "gold_page_policy": "any original gold page plus independently judged claim support; "
            "all-page recall diagnostic due mixed alternative/dependency lists",
            "judging": "anonymous answer, no runtime identity, stage judgments or probabilities",
            "human_review": "all failures, uncertainty, gold issues, grader/deterministic "
            "disagreement plus deterministic 10% sample of agreements",
            "grader": {
                "deployment": DEPLOYMENT,
                "model": MODEL,
                "version": VERSION,
                "prompt_sha256": hashlib.sha256(SYSTEM.encode()).hexdigest(),
                "cap": GradingQuota.limits["grading"],
                "automatic_retries": False,
            },
            "cost": "usage captured where available; missing is unknown, prices not frozen",
        },
    }


async def check_server(client: httpx.AsyncClient, expected: str) -> None:
    response = await client.get("/health")
    response.raise_for_status()
    if response.json().get("implementation_sha256") != expected:
        raise ValueError("running application snapshot differs from evaluation manifest")


async def capture(
    case: Case,
    directory: Path,
    url: str,
    expected: str,
    catalogue: dict,
    figures: dict,
    semaphore: asyncio.Semaphore,
) -> None:
    target = directory / "responses" / f"{case.id}.json"
    if target.exists():
        return
    async with (
        semaphore,
        httpx.AsyncClient(
            base_url=url, headers={"Origin": url}, timeout=195, trust_env=False
        ) as client,
    ):
        await check_server(client, expected)
        interrupted = {"status": "error", "error_code": "interrupted_or_not_captured"}
        record = {
            "id": case.id,
            "category": case.category,
            "split": "historical-heldout" if int(case.id[1:]) >= 58 else "benchmark",
            "http_status": None,
            "application_seconds": None,
            "response": interrupted,
            "deterministic": deterministic(case, interrupted, catalogue, figures),
            "trace_checks": trace_checks(interrupted, None),
            "images": [],
            "attempt_started_utc": datetime.now(UTC).isoformat(),
        }
        # A durable intent precedes spending; interrupted attempts are never retried.
        persist(target, record)
        started = time.monotonic()
        status = None
        try:
            response = await client.post("/api/chat", json={"question": case.question})
            seconds = time.monotonic() - started
            status = response.status_code
            persist(
                directory / "raw" / f"{case.id}.json",
                {
                    "http_status": status,
                    "application_seconds": seconds,
                    "body": response.text,
                },
            )
            result = response.json()
            if (
                not isinstance(result, dict)
                or not isinstance(result.get("answer", ""), (str, type(None)))
                or any(
                    not isinstance(result.get(key, []), list)
                    or any(not isinstance(item, dict) for item in result.get(key, []))
                    for key in ("citations", "figures")
                )
            ):
                raise ValueError("API returned a malformed response")
            if status != 200 and result.get("status") != "error":
                result = {"status": "error", "error_code": f"http_{status}"}
        except (httpx.RequestError, ValueError) as exc:
            seconds = time.monotonic() - started
            result = {"status": "error", "error_code": type(exc).__name__}
        record.update(
            {
                "http_status": status,
                "application_seconds": seconds,
                "response": result,
                "deterministic": deterministic(case, result, catalogue, figures),
            }
        )
        persist(target, record)
        trace = None
        trace_id = result.get("trace_id", "")
        if isinstance(trace_id, str) and re.fullmatch(r"[a-f0-9]{32}", trace_id):
            trace_path = ROOT / f"data/review/traces/{trace_id}.json"
            if trace_path.exists():
                trace = json.loads(trace_path.read_text())
                persist(directory / "traces" / f"{case.id}.json", trace)
        deterministic_result = deterministic(case, result, catalogue, figures)
        image_checks = []
        for item in result.get("figures", []):
            key = item.get("key")
            if key not in figures or item.get("url") != f"/api/figures/{key}":
                image_checks.append({"key": key, "valid": False})
                continue
            document = figures[key][0]
            try:
                image = await client.get(f"/api/figures/{key}")
                original = dossier_directory(ROOT, document) / f"images/{key}.png"
                valid = (
                    image.status_code == 200
                    and (
                        hashlib.sha256(image.content).digest()
                        == hashlib.sha256(original.read_bytes()).digest()
                    )
                    and image.content.startswith(b"\x89PNG\r\n\x1a\n")
                )
            except httpx.RequestError:
                valid = False
            image_checks.append({"key": key, "valid": valid})
        deterministic_result["checks"]["image_delivery"] = all(
            item["valid"] for item in image_checks
        )
        deterministic_result["pass"] = all(deterministic_result["checks"].values())
        record.update(
            {
                "deterministic": deterministic_result,
                "trace_checks": trace_checks(result, trace),
                "images": image_checks,
            }
        )
        persist(target, record)
        await check_server(client, expected)
        print(
            case.id,
            status,
            result.get("behavior", result.get("error_code")),
            f"{seconds:.1f}s",
            flush=True,
        )


async def execute(args) -> None:
    parsed = urlsplit(args.url)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in ("127.0.0.1", "localhost")
        or (parsed.username or parsed.password or parsed.path not in ("", "/") or parsed.query)
    ):
        raise ValueError("evaluation API must be a loopback HTTP origin")
    if not re.fullmatch(r"[a-zA-Z0-9_-]+", args.run_id):
        raise ValueError("invalid run ID")
    directory = ROOT / "data/evaluation" / args.run_id
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    for name in ("responses", "judgments", "traces", "raw"):
        (directory / name).mkdir(exist_ok=True, mode=0o700)
    all_cases = load_cases()
    if args.cases and (
        len(set(args.cases)) != len(args.cases)
        or not set(args.cases) <= {case.id for case in all_cases}
    ):
        raise ValueError("case selectors must be unique original case IDs")
    cases = [case for case in all_cases if not args.cases or case.id in args.cases]
    selected = [case.id for case in cases]
    frozen = stable_manifest(ROOT, args.url, args.workers, selected)
    manifest_path = directory / "manifest.json"
    if manifest_path.exists():
        existing = json.loads(manifest_path.read_text())
        if existing["frozen"] != frozen:
            raise ValueError("run configuration changed; use a new run ID, never mix snapshots")
    else:
        persist(manifest_path, {"created_utc": datetime.now(UTC).isoformat(), "frozen": frozen})
        persist(
            directory / "application-snapshot.json",
            json.loads((ROOT / "data/implementation-manifest.json").read_text()),
        )
        for name in ("benchmark.jsonl", "heldout.jsonl"):
            (directory / name).write_bytes((DATA / name).read_bytes())
        sources = code_paths(ROOT) + [ROOT / "pyproject.toml"]
        sources += list((ROOT / "manifests").glob("*.json"))
        sources += [
            path
            for path in (ROOT / "evaluation").rglob("*")
            if path.is_file() and "__pycache__" not in path.parts
        ]
        for path in sources:
            target = directory / "source" / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(path.read_bytes())
    catalogue, figures = catalogue_at(ROOT)
    if not args.grade_only:
        semaphore = asyncio.Semaphore(args.workers)
        await asyncio.gather(
            *[
                capture(
                    case,
                    directory,
                    args.url,
                    frozen["application_sha256"],
                    catalogue,
                    figures,
                    semaphore,
                )
                for case in cases
            ]
        )
    if args.grade:
        resources = ExperimentResources.model_validate_json(
            (ROOT / "manifests/azure-resources.json").read_bytes()
        )
        quota = GradingQuota(ROOT / "data/evaluation/grading-quota.sqlite3")
        async with (
            AzureCliCredential() as credential,
            httpx.AsyncClient(timeout=httpx.Timeout(150, connect=15), trust_env=False) as client,
        ):
            session = AzureSession(resources, credential, client)
            for case in cases:
                target = directory / "judgments" / f"{case.id}.json"
                captured = directory / "responses" / f"{case.id}.json"
                if target.exists() or not captured.exists():
                    continue
                result = json.loads(captured.read_text())["response"]
                if result.get("status") != "completed":
                    persist(target, {"status": "not_run_application_error"})
                    continue
                started = time.monotonic()
                persist(
                    target,
                    {
                        "status": "error",
                        "error_type": "interrupted_or_not_captured",
                        "attempt_started_utc": datetime.now(UTC).isoformat(),
                    },
                )
                try:
                    judgment = await grade(
                        session, quota, case, result, page_evidence(case, result, catalogue)
                    )
                    payload = {"status": "completed", **judgment}
                except (ServiceError, ValueError) as exc:
                    payload = {"status": "error", "error_type": type(exc).__name__}
                    if isinstance(exc, ServiceError):
                        payload.update(http_status=exc.status_code, retry_after=exc.retry_after)
                payload["grading_seconds"] = time.monotonic() - started
                persist(target, payload)
                print("grade", case.id, payload["status"], flush=True)
    if stable_manifest(ROOT, args.url, args.workers, selected) != frozen:
        raise ValueError("configuration changed during run; result is not snapshot-consistent")
    print("Captured run:", directory, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pinned 72-case standalone evaluation.")
    parser.add_argument("--execute-approved-evaluation", required=True, action="store_true")
    parser.add_argument("--url", default="http://127.0.0.1:8765")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1)
    parser.add_argument("--grade", action="store_true")
    parser.add_argument("--grade-only", action="store_true")
    parser.add_argument(
        "--cases", nargs="+", help="Original IDs for a separate targeted development run"
    )
    args = parser.parse_args()
    if args.grade_only and not args.grade:
        parser.error("--grade-only requires --grade")
    asyncio.run(execute(args))


if __name__ == "__main__":
    main()
