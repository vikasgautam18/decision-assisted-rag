"""The published Config A source must stay exactly what the benchmark ran."""

import json
import shutil
from pathlib import Path

import pytest

from evaluation.baseline_copy import SOURCE_SHA
from evaluation.baseline_serve import validate_copy
from ingestion.corpus import REFERENCE_COMMIT

ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "baseline/config-a"
MANIFEST_NAME = "BASELINE_COPY_MANIFEST.json"
PROTOCOL = ROOT / "docs/benchmark-results/earlier-runs/protocol.json"


@pytest.fixture(scope="module")
def manifest() -> dict:
    return json.loads((BASELINE / MANIFEST_NAME).read_text())


def file_differences(directory: Path, manifest: dict) -> set[str]:
    present = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file() and path.name != MANIFEST_NAME
    }
    return present ^ {item["path"] for item in manifest["files"]}


@pytest.fixture
def copy(tmp_path) -> Path:
    destination = tmp_path / "config-a"
    shutil.copytree(BASELINE, destination)
    return destination


def test_published_copy_matches_its_manifest_and_pinned_hooks():
    assert validate_copy(BASELINE)["source_sha"] == SOURCE_SHA


def test_published_copy_holds_exactly_the_listed_files(manifest):
    assert file_differences(BASELINE, manifest) == set()


def test_published_copy_uses_the_corpus_reference_commit(manifest):
    assert manifest["source_sha"] == REFERENCE_COMMIT


def test_published_protocol_records_the_same_config_a_snapshot(manifest):
    protocol = json.loads(PROTOCOL.read_text())
    assert protocol["targets"]["A"]["application_sha256"] == manifest["implementation_sha256"]


def test_published_copy_carries_its_mit_license():
    assert (BASELINE / "LICENSE").read_text().startswith("MIT License")


def test_changed_source_is_rejected(copy):
    target = copy / "backend/app.py"
    target.write_bytes(target.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="source changed"):
        validate_copy(copy)


def test_extra_and_missing_files_are_reported(copy):
    manifest = json.loads((copy / MANIFEST_NAME).read_text())
    (copy / "backend/extra.py").write_text("")
    (copy / "backend/audit.py").unlink()
    assert file_differences(copy, manifest) == {"backend/extra.py", "backend/audit.py"}
