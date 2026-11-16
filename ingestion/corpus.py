from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

REFERENCE_REPOSITORY = "vikasgautam18/semantic_rag"
REFERENCE_COMMIT = "3c7f847137db5d0b2d4680e527c39fa2173ccb10"
PDF_NAMES = (
    "ATLAS-6_Technical_Dossier.pdf",
    "NOVA-3_Technical_Dossier.pdf",
    "TERRA-9_Technical_Dossier.pdf",
    "VEGA-7_Technical_Dossier.pdf",
    "VOLTA-4_Technical_Dossier.pdf",
    "ZENITH-11_Technical_Dossier.pdf",
)
DOSSIERS = {
    "ATLAS-6": ("F", 15),
    "NOVA-3": ("D", 15),
    "TERRA-9": ("E", 15),
    "VEGA-7": ("C", 16),
    "VOLTA-4": ("C", 15),
    "ZENITH-11": ("B", 15),
}


class CorpusError(RuntimeError):
    pass


def _github_json(path: str) -> dict:
    response = subprocess.run(
        ["gh", "api", path],
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return json.loads(response.stdout)


def fetch_pinned_corpus(destination: Path) -> Path:
    """Fetch only the six fixed PDF paths and license, verifying Git object identities."""
    manifest_path = destination / "manifest.json"
    if destination.exists() and any(destination.iterdir()):
        raise CorpusError(
            "destination must be empty; existing corpus files will not be overwritten"
        )
    tree = _github_json(f"repos/{REFERENCE_REPOSITORY}/git/trees/{REFERENCE_COMMIT}?recursive=1")
    if tree.get("truncated"):
        raise CorpusError("reference tree response is truncated")
    entries = {entry["path"]: entry for entry in tree["tree"] if entry["type"] == "blob"}
    expected = (*[f"docs/{name}" for name in PDF_NAMES], "LICENSE")
    if any(path not in entries for path in expected):
        raise CorpusError("pinned repository is missing an expected PDF or license")

    downloads = []
    for path in expected:
        entry = entries[path]
        blob = _github_json(f"repos/{REFERENCE_REPOSITORY}/git/blobs/{entry['sha']}")
        if blob.get("encoding") != "base64":
            raise CorpusError(f"unsupported GitHub blob encoding for {path}")
        data = base64.b64decode("".join(blob["content"].split()), validate=True)
        git_sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        if git_sha != entry["sha"] or len(data) != entry["size"]:
            raise CorpusError(f"Git object identity mismatch for {path}")
        if path != "LICENSE" and not data.startswith(b"%PDF-"):
            raise CorpusError(f"expected PDF bytes for {path}")
        downloads.append(
            (
                path,
                data,
                {"path": path, "git_blob_sha": git_sha, "sha256": hashlib.sha256(data).hexdigest()},
            )
        )

    destination.mkdir(parents=True, exist_ok=True)
    for path, data, _ in downloads:
        target = destination / path
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as output:
            output.write(data)
    manifest = {
        "schema_version": "1",
        "status": "source_inventory_only",
        "repository": REFERENCE_REPOSITORY,
        "commit": REFERENCE_COMMIT,
        "artifacts": [record for _, _, record in downloads],
        "analysis_performed": False,
        "index_created": False,
    }
    with manifest_path.open("x") as output:
        json.dump(manifest, output, indent=2)
        output.write("\n")
    return manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and verify the fixed reference PDF corpus.")
    parser.add_argument("--destination", type=Path, default=Path("data/corpus"))
    args = parser.parse_args()
    print(fetch_pinned_corpus(args.destination))


if __name__ == "__main__":
    main()
