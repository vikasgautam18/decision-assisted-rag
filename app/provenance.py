from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path

from pydantic import Field

from ingestion.corpus import DOSSIERS, REFERENCE_COMMIT, REFERENCE_REPOSITORY
from pipelines.contracts import Artifact


class FileRecord(Artifact):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ImplementationManifest(Artifact):
    status: str
    created_utc: str
    reference_repository: str
    reference_commit: str
    documents: tuple[str, ...]
    index_name: str
    artifacts: tuple[FileRecord, ...]
    deployments: dict[str, str]
    policies: dict[str, str]
    usage_and_cost: str
    packages: dict[str, str]


def code_paths(root: Path) -> list[Path]:
    return sorted(
        p
        for folder in ("app", "clients", "ingestion", "pipelines")
        for p in (root / folder).rglob("*")
        if p.suffix in (".py", ".html") and "__pycache__" not in p.parts
    )


def dossier_directory(root: Path, document: str) -> Path:
    if document not in DOSSIERS:
        raise ValueError("unknown corpus document")
    return root / ("data/pilot" if document == "NOVA-3" else f"data/dossiers/{document}")


def write_manifest(root: Path) -> Path:
    paths = code_paths(root) + [
        root / "manifests/azure-resources.json",
        root / "data/corpus/manifest.json",
        root / "pyproject.toml",
    ]
    for document in DOSSIERS:
        directory = dossier_directory(root, document)
        paths.extend(
            [
                root / f"data/corpus/docs/{document}_Technical_Dossier.pdf",
                root / f"data/live-checkpoint/{document}.analysis.json",
                root / f"data/live-checkpoint/{document}.layout.json",
                root / f"manifests/{document}-context.json",
                directory / "chunks.json",
            ]
        )
        paths.extend(sorted(directory.glob("*.description.json")))
        paths.extend(sorted((directory / "images").glob("*.png")))
    manifest = ImplementationManifest(
        status="implementation_snapshot_not_frozen_evaluation",
        created_utc=datetime.now(UTC).isoformat(),
        reference_repository=REFERENCE_REPOSITORY,
        reference_commit=REFERENCE_COMMIT,
        documents=tuple(DOSSIERS),
        index_name=json.loads((root / "manifests/azure-resources.json").read_text())["index_name"],
        artifacts=tuple(
            FileRecord(
                path=str(p.relative_to(root)), sha256=hashlib.sha256(p.read_bytes()).hexdigest()
            )
            for p in paths
        ),
        deployments={
            "writer": "gpt-5.6-sol;2026-07-09;westus3;GlobalStandard",
            "embedding": "text-embedding-3-large;1;3072;westus3;Standard",
            "decision": "decision1-deployment;Microsoft-Decision-1;1;westus3;GlobalStandard",
            "layout": "prebuilt-layout;2024-11-30;westus2",
            "search_api": "2025-09-01;westus2",
        },
        policies={
            "deadline": "180 seconds shared; no inference retries",
            "limits": "20 semantic decisions (ordinary-model backend in G) and "
            "5 aggregate embedding/writer calls per request; 1 expansion; 3 repairs",
            "approved_cumulative_caps": (
                "DI6; Decision-1 4800; aggregate embedding/vision/writer 1200; "
                "independent grader 600"
            ),
            "admission": "per-deployment 65s rolling window; 80% observed token quota; "
            "max(o200k tokens, characters/3) + output reservation +256; "
            "429 cooldown respects Retry-After, no automatic inference retries; "
            "single-process only; not a shared-account throughput guarantee",
            "verification": "supported probability >=0.8 if supplied; provisional, uncalibrated",
            "visual_evidence": "versioned cached descriptions; original crops delivered",
            "calculations": "named operations; exact quote or original-line-address provenance; "
            "no coordinate metadata operands; transitive typed intermediate references; "
            "explicit f_to_c only; half_even default, explicit half_up allowed",
            "retention": "conversations 30min; default private traces 7 days; "
            "explicit experiment traces retained for review",
        },
        usage_and_cost="Provider usage when reported; failed reservations retained; unknown "
        "usage is not zero. Experiment model/API estimates require a frozen exact-meter rate "
        "card and explicit cost scope; Search/Defender exclusions are not zero charges. "
        "This implementation snapshot alone does not authorize or finalize evaluation.",
        packages={
            name: version(name)
            for name in (
                "azure-identity",
                "fastapi",
                "httpx",
                "pydantic",
                "uvicorn",
                "tiktoken",
            )
        },
    )
    target = root / "data/implementation-manifest.json"
    target.write_text(manifest.model_dump_json(indent=2) + "\n")
    return target


def validate_manifest(root: Path) -> ImplementationManifest:
    manifest = ImplementationManifest.model_validate_json(
        (root / "data/implementation-manifest.json").read_bytes()
    )
    if set(manifest.documents) != set(DOSSIERS):
        raise ValueError("manifest must contain the complete configured corpus")
    listed = set()
    for record in manifest.artifacts:
        path = (root / record.path).resolve()
        if not path.is_relative_to(root.resolve()) or record.path in listed:
            raise ValueError("invalid or duplicate manifest artifact path")
        if hashlib.sha256(path.read_bytes()).hexdigest() != record.sha256:
            raise ValueError(f"manifest artifact changed: {record.path}")
        listed.add(record.path)
    required_code = {str(p.relative_to(root)) for p in code_paths(root)}
    if not required_code <= listed:
        raise ValueError("manifest does not cover the current implementation")
    return manifest


if __name__ == "__main__":
    print(write_manifest(Path(__file__).resolve().parents[1]))
