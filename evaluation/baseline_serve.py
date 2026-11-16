from __future__ import annotations

import hashlib
import importlib
import json
import os
import sys
from pathlib import Path

from evaluation.baseline_copy import SOURCE_SHA
from evaluation.experiment import Approval, Protocol, digest


def validate_copy(directory: Path) -> dict:
    directory = directory.resolve(strict=True)
    if (directory / ".git").exists() or (directory / ".env").exists():
        raise ValueError("isolated baseline must not contain git metadata or an environment file")
    manifest = json.loads((directory / "BASELINE_COPY_MANIFEST.json").read_text())
    snapshot = manifest["implementation_sha256"]
    unsigned = {key: value for key, value in manifest.items() if key != "implementation_sha256"}
    if manifest["source_sha"] != SOURCE_SHA or digest(unsigned) != snapshot:
        raise ValueError("baseline source pin or manifest hash mismatch")
    for item in manifest["files"]:
        target = directory / item["path"]
        if (
            target.is_symlink()
            or not target.resolve(strict=True).is_relative_to(directory)
            or hashlib.sha256(target.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise ValueError("isolated baseline source changed")
    hooks = Path(__file__).with_name("baseline_observability.py").read_bytes()
    if hashlib.sha256(hooks).hexdigest() != manifest["instrumentation"]["sha256"]:
        raise ValueError("baseline hook implementation changed after copy preparation")
    return manifest


def create_baseline_app():
    """Run only as a fresh, explicitly configured isolated uvicorn factory."""
    directory = Path(os.environ["DECISION_RAG_BASELINE_COPY"])
    manifest = validate_copy(directory)
    protocol = Protocol.model_validate_json(Path(os.environ["DECISION_RAG_PROTOCOL"]).read_bytes())
    approval = Approval.model_validate_json(Path(os.environ["DECISION_RAG_APPROVAL"]).read_bytes())
    if approval.protocol_sha256 != protocol.sha256:
        raise ValueError("baseline approval differs from the frozen protocol")
    if protocol.targets["A"].application_sha256 != manifest["implementation_sha256"]:
        raise ValueError("baseline copy differs from the frozen arm A snapshot")
    if os.environ.get("AGENT_BACKEND", "foundry") != "foundry":
        raise ValueError("baseline Foundry backend must not be changed")
    required = (
        "SUBSCRIPTION_ID",
        "SUFFIX",
        "SEARCH_ENDPOINT",
        "AI_ENDPOINT",
        "STORAGE_ACCOUNT",
        "PROJECT_ENDPOINT",
        "KB_NAME",
        "CHAT_DEPLOYMENT",
    )
    if any(not os.environ.get(key) for key in required):
        raise ValueError("approved explicit baseline environment is incomplete")
    if os.environ.get("ENV_FILE"):
        raise ValueError("supply approved settings explicitly; do not load a reference .env")
    if any(name in sys.modules for name in ("backend", "common")):
        raise ValueError("baseline factory needs a fresh process with no other checkout imports")
    sys.path.insert(0, str(directory.resolve()))
    ask = importlib.import_module("backend.agent.ask")
    baseline = importlib.import_module("backend.app").app
    from azure.ai.projects import AIProjectClient

    from evaluation.baseline_observability import (
        BaselineObservationMiddleware,
        instrument_foundry,
    )

    root = Path(__file__).resolve().parents[1]
    wrapped = BaselineObservationMiddleware(
        baseline,
        root / "data/experiments" / protocol.experiment_id / "server-spans",
        metadata={
            "arm": "A",
            "experiment_id": protocol.experiment_id,
            "phase": protocol.phase,
            "implementation_sha256": manifest["implementation_sha256"],
        },
    )

    async def application(scope, receive, send):
        if scope["type"] == "lifespan":
            with instrument_foundry(ask, AIProjectClient):
                await wrapped(scope, receive, send)
        else:
            await wrapped(scope, receive, send)

    return application
