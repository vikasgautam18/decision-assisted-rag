from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import sqlite3
from pathlib import Path

import httpx
from azure.identity.aio import AzureCliCredential

from app.resources import ExperimentResources
from clients.azure import AzureSession
from clients.decision import connectivity_check
from ingestion.analyze import analyze_pdf
from ingestion.normalize import normalize_layout

logger = logging.getLogger(__name__)


class CheckpointQuotaExhausted(RuntimeError):
    pass


class CheckpointQuota:
    limits = {"di_analysis": 6, "decision": 4800, "model": 1200}

    def __init__(self, path: Path, *, limits: dict[str, int] | None = None) -> None:
        if limits is not None:
            if set(limits) != set(type(self).limits) or any(
                type(value) is not int or value < 0 for value in limits.values()
            ):
                raise ValueError("approved quota limits must match the ledger categories")
            self.limits = dict(limits)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with sqlite3.connect(path) as connection:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS calls "
                "(id INTEGER PRIMARY KEY, category TEXT NOT NULL, state TEXT NOT NULL, "
                "summary TEXT)"
            )

    def reserve(self, category: str) -> int:
        if category not in self.limits:
            raise ValueError("unknown checkpoint quota category")
        with sqlite3.connect(self.path, isolation_level=None) as connection:
            connection.execute("BEGIN IMMEDIATE")
            count = connection.execute(
                "SELECT COUNT(*) FROM calls WHERE category=?", (category,)
            ).fetchone()[0]
            if count >= self.limits[category]:
                raise CheckpointQuotaExhausted(
                    f"approved checkpoint {category} budget is exhausted"
                )
            cursor = connection.execute(
                "INSERT INTO calls(category,state) VALUES (?, 'reserved')", (category,)
            )
            call_id = cursor.lastrowid
            connection.execute("COMMIT")
        if call_id is None:
            raise RuntimeError("could not reserve checkpoint call")
        return call_id

    def complete(self, call_id: int, summary: dict) -> None:
        with sqlite3.connect(self.path) as connection:
            connection.execute(
                "UPDATE calls SET state='completed', summary=? WHERE id=? AND state='reserved'",
                (json.dumps(summary), call_id),
            )

    def snapshot(self) -> list[dict]:
        with sqlite3.connect(self.path) as connection:
            return [
                {"id": row[0], "category": row[1], "state": row[2]}
                for row in connection.execute("SELECT id,category,state FROM calls ORDER BY id")
            ]


async def execute(root: Path, resources_path: Path) -> None:
    resources = ExperimentResources.model_validate_json(
        await asyncio.to_thread(resources_path.read_bytes)
    )
    directory = root / "data" / "live-checkpoint"
    await asyncio.to_thread(directory.mkdir, parents=True, exist_ok=True)
    quota = await asyncio.to_thread(CheckpointQuota, directory / "quota.sqlite3")
    manifest = json.loads(await asyncio.to_thread((root / "data/corpus/manifest.json").read_text))
    pdf_relative = "docs/NOVA-3_Technical_Dossier.pdf"
    record = next(item for item in manifest["artifacts"] if item["path"] == pdf_relative)
    pdf = await asyncio.to_thread((root / "data/corpus" / pdf_relative).read_bytes)
    pdf_hash = hashlib.sha256(pdf).hexdigest()
    if pdf_hash != record["sha256"]:
        raise ValueError("local PDF no longer matches pinned corpus inventory")
    async with (
        AzureCliCredential() as credential,
        httpx.AsyncClient(timeout=90, trust_env=False) as client,
    ):
        session = AzureSession(resources, credential, client)
        analysis_path = directory / "NOVA-3.analysis.json"
        if not await asyncio.to_thread(analysis_path.exists):
            call_id = await asyncio.to_thread(quota.reserve, "di_analysis")
            logger.info("Starting approved DI analysis: NOVA-3, pages 1-15")
            raw = await analyze_pdf(session, pdf, pages=15)
            await asyncio.to_thread(write_exclusive, analysis_path, raw)
            await asyncio.to_thread(
                quota.complete, call_id, {"pages_requested": 15, "artifact": analysis_path.name}
            )
        artifact = normalize_layout(
            await asyncio.to_thread(analysis_path.read_bytes),
            document="NOVA-3",
            revision="D",
            pdf_sha256=pdf_hash,
        )
        await asyncio.to_thread(
            (directory / "NOVA-3.layout.json").write_text, artifact.model_dump_json(indent=2) + "\n"
        )
        decision_path = directory / "decision-connectivity.json"
        if not await asyncio.to_thread(decision_path.exists):
            call_id = await asyncio.to_thread(quota.reserve, "decision")
            logger.info("Starting the single approved Decision-1 connectivity request")
            result = await connectivity_check(session)
            await asyncio.to_thread(
                write_exclusive, decision_path, (json.dumps(result, indent=2) + "\n").encode()
            )
            await asyncio.to_thread(quota.complete, call_id, result)
    summary = {
        "document": "NOVA-3",
        "revision": "D",
        "pages": len(artifact.pages),
        "tables": len(artifact.tables),
        "figures": len(artifact.figures),
        "review_issues": artifact.issues,
        "quota": await asyncio.to_thread(quota.snapshot),
        "phase": 1,
        "phase_1_complete": False,
        "phase_3_authorized": False,
    }
    await asyncio.to_thread(
        (directory / "summary.json").write_text, json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


def write_exclusive(path: Path, data: bytes) -> None:
    with path.open("xb") as output:
        output.write(data)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="One explicitly approved, billable live checkpoint."
    )
    parser.add_argument("--execute-approved-checkpoint", action="store_true", required=True)
    args = parser.parse_args()
    if args.execute_approved_checkpoint:
        logging.basicConfig(level=logging.INFO)
        root = Path(__file__).resolve().parents[1]
        asyncio.run(execute(root, root / "manifests/azure-resources.json"))


if __name__ == "__main__":
    main()
