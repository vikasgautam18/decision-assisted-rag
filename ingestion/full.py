from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
from pathlib import Path

import httpx
from azure.identity.aio import AzureCliCredential

from app.resources import ExperimentResources
from clients.azure import AzureSession
from ingestion.analyze import analyze_pdf
from ingestion.checkpoint import CheckpointQuota, write_exclusive
from ingestion.corpus import DOSSIERS
from ingestion.normalize import normalize_layout
from ingestion.pilot import prepare


async def analyze(root: Path) -> None:
    resources = ExperimentResources.model_validate_json(
        (root / "manifests/azure-resources.json").read_bytes()
    )
    inventory = json.loads((root / "data/corpus/manifest.json").read_text())
    hashes = {item["path"]: item["sha256"] for item in inventory["artifacts"]}
    directory = root / "data/live-checkpoint"
    quota = CheckpointQuota(directory / "quota.sqlite3")
    async with (
        AzureCliCredential() as credential,
        httpx.AsyncClient(timeout=httpx.Timeout(90, connect=15), trust_env=False) as client,
    ):
        session = AzureSession(resources, credential, client)
        for document, (revision, pages) in DOSSIERS.items():
            relative = f"docs/{document}_Technical_Dossier.pdf"
            pdf = await asyncio.to_thread((root / "data/corpus" / relative).read_bytes)
            digest = hashlib.sha256(pdf).hexdigest()
            if digest != hashes[relative]:
                raise ValueError(f"{document} does not match pinned corpus")
            raw_path = directory / f"{document}.analysis.json"
            if not raw_path.exists():
                call = quota.reserve("di_analysis")
                print(f"Analyzing {document}, {pages} pages", flush=True)
                raw = await analyze_pdf(session, pdf, pages=pages)
                await asyncio.to_thread(write_exclusive, raw_path, raw)
                quota.complete(call, {"document": document, "pages_requested": pages})
            artifact = normalize_layout(
                raw_path.read_bytes(),
                document=document,
                revision=revision,
                pdf_sha256=digest,
            )
            (directory / f"{document}.layout.json").write_text(
                artifact.model_dump_json(indent=2) + "\n"
            )
            print(
                f"{document}: {len(artifact.pages)} pages, {len(artifact.tables)} tables, "
                f"{len(artifact.figures)} figures",
                flush=True,
            )


async def execute(root: Path, analyze_only: bool) -> None:
    await analyze(root)
    if not analyze_only:
        for document in DOSSIERS:
            await prepare(root, document)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare the approved six-dossier corpus.")
    parser.add_argument("--execute-approved-checkpoint", action="store_true", required=True)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()
    asyncio.run(execute(Path(__file__).resolve().parents[1], args.analyze_only))


if __name__ == "__main__":
    main()
