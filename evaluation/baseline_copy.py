"""Prepare an audited runtime-only copy; never read the reference working tree."""

from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
from pathlib import Path, PurePosixPath

SOURCE_SHA = "3c7f847137db5d0b2d4680e527c39fa2173ccb10"
REFERENCE_ENV = "SEMANTIC_RAG_REFERENCE"


def reference_checkout() -> Path | None:
    """Local checkout of the original (private) project, or None when it is not configured."""
    value = os.environ.get(REFERENCE_ENV)
    return Path(value) if value else None


def _git(source: Path, *args: str) -> bytes:
    return subprocess.check_output(["git", "-C", str(source), *args])


def _runtime_file(path: PurePosixPath) -> bool:
    if any(
        part.startswith(".")
        or part.lower() in {"secrets", "venv", "env", "__pycache__"}
        or "secret" in part.lower()
        or part.lower().endswith((".pem", ".key"))
        for part in path.parts
    ):
        return False
    if len(path.parts) == 1:
        return path.name in {"LICENSE", "COPYING", "NOTICE", "requirements.txt", "README.md"}
    return (path.parts[0] in {"backend", "common"} and path.suffix == ".py") or (
        path.parts[0] == "web"
        and path.suffix.lower() in {".html", ".js", ".css", ".svg", ".png", ".ico"}
    )


def prepare_copy(
    source: Path,
    destination: Path,
    *,
    source_sha: str = SOURCE_SHA,
) -> dict:
    """Create a NEW explicit destination from pinned git blobs, without configuration.

    No checkout, archive extraction, symlink following, resource copying or source
    edits occur. ``source_sha`` is overridable for offline fixture repositories.
    """
    source = Path(source).resolve(strict=True)
    source = Path(_git(source, "rev-parse", "--show-toplevel").decode().strip()).resolve(
        strict=True,
    )
    destination = Path(destination).absolute()
    if not re.fullmatch(r"[a-f0-9]{40}", source_sha):
        raise ValueError("source_sha must be an exact 40-character commit")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(destination)
    parent = destination.parent.resolve(strict=True)
    destination = parent / destination.name
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError("destination must be separate from the reference checkout")
    resolved = _git(source, "rev-parse", f"{source_sha}^{{commit}}").decode().strip()
    if resolved != source_sha:
        raise ValueError("source commit does not match pin")
    tree = _git(source, "ls-tree", "-r", "-z", source_sha)
    files, excluded, blobs = [], [], []
    for entry in tree.split(b"\0"):
        if not entry:
            continue
        metadata, raw_path = entry.split(b"\t", 1)
        mode, kind, oid = metadata.decode().split()
        name = raw_path.decode("utf-8")
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("unsafe git tree path")
        if not _runtime_file(path) or kind != "blob" or mode not in {"100644", "100755"}:
            excluded.append({"path": name, "reason": "not regular allowlisted runtime source"})
            continue
        data = _git(source, "cat-file", "blob", oid)
        blobs.append((name, mode, data))
        files.append(
            {
                "path": name,
                "git_blob": oid,
                "git_mode": mode,
                "sha256": hashlib.sha256(data).hexdigest(),
            }
        )
    if not any(item["path"] == "LICENSE" for item in files):
        raise ValueError("a tracked LICENSE is required")
    manifest = {
        "schema": "baseline-copy-v1",
        "source_sha": source_sha,
        "files": files,
        "excluded": excluded,
        "source_patches": [],
        "instrumentation": {
            "kind": "external opt-in hooks; no copied-source edits",
            "module": "evaluation.baseline_observability",
            "sha256": hashlib.sha256(
                Path(__file__).with_name("baseline_observability.py").read_bytes()
            ).hexdigest(),
        },
        "configuration": "not copied; approved environment and deployment preflight required",
    }
    manifest["implementation_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode(),
    ).hexdigest()
    destination.mkdir(mode=0o700)
    for name, mode, data in blobs:
        target = destination / name
        missing = []
        ancestor = target.parent
        while ancestor != destination:
            missing.append(ancestor)
            ancestor = ancestor.parent
        for directory in reversed(missing):
            directory.mkdir(mode=0o700, exist_ok=True)
        with target.open("xb") as stream:
            target.chmod(0o700 if mode == "100755" else 0o600)
            stream.write(data)
    target = destination / "BASELINE_COPY_MANIFEST.json"
    with target.open("x") as stream:
        target.chmod(0o600)
        json.dump(manifest, stream, indent=2)
        stream.write("\n")
    return manifest
