import json

import pytest

from app.provenance import dossier_directory, validate_manifest, write_manifest
from ingestion.corpus import DOSSIERS


def prepared_tree(root):
    resources = root / "manifests/azure-resources.json"
    resources.parent.mkdir()
    resources.write_text(json.dumps({"index_name": "decision-rag-chunks-v1"}))
    (root / "pyproject.toml").write_text("# fixture\n")
    for document in DOSSIERS:
        for relative in (
            f"data/corpus/docs/{document}_Technical_Dossier.pdf",
            f"data/live-checkpoint/{document}.analysis.json",
            f"data/live-checkpoint/{document}.layout.json",
            f"manifests/{document}-context.json",
        ):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture, never live evidence")
        directory = dossier_directory(root, document)
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "chunks.json").write_text("{}")
    (root / "data/corpus/manifest.json").write_text("{}")
    return write_manifest(root)


def test_snapshot_is_implementation_provenance_not_frozen_evaluation(tmp_path):
    prepared_tree(tmp_path)
    manifest = validate_manifest(tmp_path)
    assert set(manifest.documents) == set(DOSSIERS)
    assert manifest.status == "implementation_snapshot_not_frozen_evaluation"
    assert "unknown" in manifest.usage_and_cost


def test_modified_artifact_prevents_startup(tmp_path):
    prepared_tree(tmp_path)
    (tmp_path / "manifests/NOVA-3-context.json").write_text("changed")
    with pytest.raises(ValueError, match="artifact changed"):
        validate_manifest(tmp_path)


def test_unrecorded_code_prevents_startup(tmp_path):
    prepared_tree(tmp_path)
    (tmp_path / "app").mkdir()
    (tmp_path / "app/new.py").write_text("unexpected = True\n")
    with pytest.raises(ValueError, match="current implementation"):
        validate_manifest(tmp_path)


def test_manifest_cannot_read_outside_root(tmp_path):
    target = prepared_tree(tmp_path)
    payload = json.loads(target.read_text())
    payload["artifacts"][0]["path"] = "../outside"
    target.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="path"):
        validate_manifest(tmp_path)
