from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from azure.identity.aio import AzureCliCredential
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, JSONResponse
from pydantic import Field

from app.boundary import install_api_boundary
from app.main import LookupRequest
from app.provenance import dossier_directory, validate_manifest
from app.resources import ExperimentResources
from app.sessions import ConversationBusy, ConversationNotFound, SessionStore
from clients.azure import AzureSession
from clients.models import Models
from clients.search import Search
from ingestion.checkpoint import CheckpointQuota
from ingestion.context import SourceContext
from ingestion.corpus import DOSSIERS
from ingestion.layout import LayoutArtifact
from pipelines.arms import policy_for
from pipelines.evidence import section_candidates
from pipelines.live import LivePipeline

ROOT = Path(__file__).resolve().parents[1]
KB = "decision-rag-dossiers-v1"
COOKIE = "decision_rag_session"


class ChatRequest(LookupRequest):
    kb: str | None = None
    conversation_id: str | None = Field(default=None, pattern=r"^[a-f0-9]{32}$")


def create_live_app(
    root: Path = ROOT,
    *,
    runner=None,
    arm: str = "F",
    pilot_profile: str = "baseline",
    experiment_id: str | None = None,
    telemetry_directory: Path | None = None,
    resource_path: Path | None = None,
    approved_limits: dict[str, int] | None = None,
) -> FastAPI:
    policy = policy_for(arm)
    if pilot_profile not in ("baseline", "candidate"):
        raise ValueError("Unknown Arm F pilot profile")
    if pilot_profile == "candidate" and arm != "F":
        raise ValueError("Pilot profile is defined only for Arm F")
    if (experiment_id is None) != (telemetry_directory is None):
        raise ValueError("experiment ID and telemetry directory must be configured together")
    if experiment_id is not None and not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", experiment_id):
        raise ValueError("invalid configured experiment ID")

    @asynccontextmanager
    async def lifespan(application: FastAPI):
        review = (
            root / "data/review/experiments" / experiment_id / policy.arm
            if experiment_id is not None
            else root / "data/review"
        )
        store = await asyncio.to_thread(SessionStore, review / "conversations.sqlite3")
        application.state.store = store
        if runner is not None:
            application.state.runner = runner
            application.state.figures = {}
            yield
            return
        manifest = await asyncio.to_thread(validate_manifest, root)
        application.state.snapshot_sha256 = hashlib.sha256(
            await asyncio.to_thread((root / "data/implementation-manifest.json").read_bytes)
        ).hexdigest()
        resources = ExperimentResources.model_validate_json(
            await asyncio.to_thread(
                (resource_path or root / "manifests/azure-resources.json").read_bytes
            )
        )
        if manifest.index_name != resources.index_name:
            raise RuntimeError("manifest index does not match configured experiment")
        application.state.figures = {}
        catalogue = []
        sections = {}
        for document, (revision, _) in DOSSIERS.items():
            layout = LayoutArtifact.model_validate_json(
                await asyncio.to_thread(
                    (root / f"data/live-checkpoint/{document}.layout.json").read_bytes
                )
            )
            if layout.document != document or layout.revision != revision:
                raise RuntimeError("layout document identity does not match configured corpus")
            reviewed = SourceContext.model_validate_json(
                await asyncio.to_thread((root / f"manifests/{document}-context.json").read_bytes)
            )
            reviewed.validate_layout(layout)
            sections[document] = reviewed.sections
            directory = dossier_directory(root, document)
            for figure in layout.figures:
                application.state.figures[figure.figure_id] = (
                    directory / f"images/{figure.figure_id}.png"
                )
            cached = json.loads(await asyncio.to_thread((directory / "chunks.json").read_text))
            for chunk in cached["chunks"]:
                if "vector" not in chunk or len(chunk["vector"]) != resources.embedding_dimensions:
                    raise RuntimeError("corpus embedding preparation is incomplete")
                catalogue.append({key: value for key, value in chunk.items() if key != "vector"})
        for document, mapping in sections.items():
            if any(
                not section_candidates(catalogue, document, section, True, sections)
                for section in mapping
            ):
                raise RuntimeError("reviewed section has no prepared documentary evidence")
        quota = await asyncio.to_thread(
            CheckpointQuota, root / "data/live-checkpoint/quota.sqlite3", limits=approved_limits
        )
        application.state.quota = quota
        async with (
            AzureCliCredential() as credential,
            httpx.AsyncClient(timeout=httpx.Timeout(90, connect=30), trust_env=False) as client,
        ):
            session = AzureSession(resources, credential, client)
            application.state.runner = LivePipeline(
                Models(session, quota),
                Search(session),
                review / "traces",
                catalogue,
                sections,
                arm=policy.arm,
                pilot_profile=pilot_profile,
            )
            yield

    application = FastAPI(title="Decision-assisted RAG: six technical dossiers", lifespan=lifespan)
    install_api_boundary(application)

    if experiment_id is not None:
        from app.experiment_boundary import ExperimentTelemetryMiddleware

        application.add_middleware(
            ExperimentTelemetryMiddleware,
            experiment_id=experiment_id,
            arm=policy.arm,
            directory=telemetry_directory,
        )

    @application.get("/health")
    async def health():
        return {
            "status": "ok",
            "execution_mode": "live" if runner is None else "test",
            "scope": list(DOSSIERS),
            "corpus_ready": runner is None,
            "phase_3_authorized": False,
            "implementation_sha256": getattr(application.state, "snapshot_sha256", None),
            "arm": policy.arm,
            "pilot_profile": pilot_profile,
            "experiment_id": experiment_id,
        }

    @application.get("/api/status")
    async def status():
        if runner is not None:
            return {"scope": list(DOSSIERS), "budget": {}}
        calls = await asyncio.to_thread(application.state.quota.snapshot)
        used = {
            key: sum(item["category"] == key for item in calls)
            for key in application.state.quota.limits
        }
        return {
            "scope": list(DOSSIERS),
            "budget": {
                key: {"used": used[key], "limit": limit, "remaining": max(0, limit - used[key])}
                for key, limit in application.state.quota.limits.items()
            },
        }

    def session(request: Request, response: Response) -> str:
        sid, cookie = application.state.store.session(request.cookies.get(COOKIE, ""))
        response.set_cookie(
            COOKIE,
            cookie,
            max_age=1800,
            httponly=True,
            samesite="strict",
            secure=request.url.scheme == "https",
            path="/",
        )
        return sid

    @application.post("/api/chat")
    async def chat(body: ChatRequest, request: Request, response: Response):
        if body.kb not in (None, KB):
            raise HTTPException(403, "Only the configured dossier knowledge base is available.")
        sid = session(request, response)
        store = application.state.store
        try:
            cid, history = await asyncio.to_thread(store.begin, sid, body.conversation_id)
        except ConversationNotFound as exc:
            raise HTTPException(404, "Conversation not found.") from exc
        except ConversationBusy as exc:
            raise HTTPException(409, "Conversation is busy or capacity was reached.") from exc
        result = None
        try:
            result = await application.state.runner.run(
                body.question,
                [message["content"] for message in history if message["role"] == "user"],
            )
        finally:
            await asyncio.shield(asyncio.to_thread(store.finish, sid, cid, body.question, result))
        result = {**result, "conversation_id": cid, "kb": KB}
        if result["status"] == "error":
            error_response = JSONResponse(status_code=503, content=result)
            error_response.raw_headers.extend(
                (key, value) for key, value in response.raw_headers if key == b"set-cookie"
            )
            return error_response
        return result

    @application.get("/api/conversations/{cid}")
    async def conversation(cid: str, request: Request, response: Response):
        sid = session(request, response)
        try:
            messages = await asyncio.to_thread(application.state.store.history, sid, cid)
        except ConversationNotFound as exc:
            raise HTTPException(404, "Conversation not found.") from exc
        return {"conversation_id": cid, "kb": KB, "messages": messages}

    @application.get("/api/figures/{key}")
    async def figure(key: str):
        path = application.state.figures.get(key)
        if path is None:
            raise HTTPException(404, "Figure not found.")
        return FileResponse(path, media_type="image/png")

    @application.get("/api/documents/{key}")
    async def document(key: str):
        if key not in DOSSIERS:
            raise HTTPException(404, "Document not found.")
        return FileResponse(
            root / f"data/corpus/docs/{key}_Technical_Dossier.pdf", media_type="application/pdf"
        )

    @application.get("/")
    async def index():
        return FileResponse(ROOT / "app/static/index.html")

    return application


def create_experiment_app() -> FastAPI:
    """Explicit uvicorn --factory entry point; never changes the default app."""
    experiment_id = os.environ["DECISION_RAG_EXPERIMENT"]
    resources = os.environ.get("DECISION_RAG_RESOURCES")
    from evaluation.experiment import Approval, Protocol

    protocol = Protocol.model_validate_json(Path(os.environ["DECISION_RAG_PROTOCOL"]).read_bytes())
    approval = Approval.model_validate_json(Path(os.environ["DECISION_RAG_APPROVAL"]).read_bytes())
    if protocol.experiment_id != experiment_id or approval.protocol_sha256 != protocol.sha256:
        raise ValueError("server experiment and paid approval do not match frozen protocol")
    return create_live_app(
        arm=os.environ["DECISION_RAG_ARM"],
        pilot_profile=os.environ.get("RAG_F_PILOT_PROFILE", "baseline"),
        experiment_id=experiment_id,
        telemetry_directory=ROOT / "data/experiments" / experiment_id / "server-spans",
        resource_path=Path(resources) if resources else None,
        approved_limits={
            key: value
            for key, value in approval.runtime_cumulative_limits.items()
            if key != "grading"
        },
    )


app = create_live_app()
