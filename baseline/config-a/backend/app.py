"""Anonymous sandbox API with browser sessions and a private-storage figure proxy.

Run: uvicorn backend.app:app --host 127.0.0.1 --port 8000
"""
import hashlib
import hmac
import re
import secrets
import threading
import time
from pathlib import Path
import uuid

import requests
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from backend import audit
from backend import settings as S
from common import config as C

WEB = Path(__file__).resolve().parent.parent / "web"
FIG_KEY = re.compile(r"^[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-fig\d{2}$")
SESSION_COOKIE = "sandbox_session"


class ChatRequest(BaseModel):
    kb: str | None = None
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


class ExcelAsk(BaseModel):
    question: str = Field(min_length=1, max_length=2000)


def create_app(agent_fn=None, figure_fn=None, excel_mod=None):
    app = FastAPI(title="Dossier assistant")
    session_key = secrets.token_bytes(32)
    excel_sessions = {}  # session id -> session from backend.excel_agent.prepare
    excel_locks = {}  # one upload/question at a time per session
    excel_guard = threading.Lock()
    conversations = {}  # id -> {"owner": session id, "kb": str, "messages": [...]}

    @app.middleware("http")
    async def api_boundary(request, call_next):
        if not request.url.path.startswith("/api/"):
            return await call_next(request)
        origin = request.headers.get("origin")
        if ((origin is not None and origin != str(request.base_url).rstrip("/"))
                or request.headers.get("sec-fetch-site") == "cross-site"):
            audit.record(event="request_rejected", status=403)
            return Response("cross-origin API requests are not allowed", status_code=403)
        try:
            response = await call_next(request)
        except Exception as exc:
            audit.record(event="request_failed", path=getattr(request.scope.get("route"), "path", "/api/"),
                         error=type(exc).__name__)
            raise
        if response.status_code >= 400:
            audit.record(event="request_failed", path=getattr(request.scope.get("route"), "path", "/api/"),
                         status=response.status_code)
        response.headers.setdefault("Cache-Control", "no-store")
        return response

    def agent(question, history, kb):
        if agent_fn:
            return agent_fn(question, history, kb)
        from backend.agent.ask import run
        return run(question, history=history, kb=kb)

    def session(request: Request, response: Response):
        cookie = request.cookies.get(SESSION_COOKIE, "")
        sid, _, signature = cookie.partition(".")
        expected = hmac.new(session_key, sid.encode(), hashlib.sha256).hexdigest()
        if (not re.fullmatch(r"[a-f0-9]{64}\.[a-f0-9]{64}", cookie)
                or not hmac.compare_digest(signature, expected)):
            sid = secrets.token_hex(32)
            signature = hmac.new(session_key, sid.encode(), hashlib.sha256).hexdigest()
            # A process-local signing key avoids a registry and invalidates cookies on restart.
            response.set_cookie(SESSION_COOKIE, f"{sid}.{signature}", httponly=True,
                                samesite="strict", secure=request.url.scheme == "https", path="/")
        return sid

    @app.post("/api/chat")
    def chat(req: ChatRequest, sid=Depends(session)):
        kb = C.KB_NAME if req.kb is None else req.kb
        if kb != C.KB_NAME:
            raise HTTPException(403, "only the configured demo knowledge base is available")
        cid = req.conversation_id
        if cid:
            conv = conversations.get(cid)
            if not conv or conv["owner"] != sid:
                raise HTTPException(404, "conversation not found")
            if conv["kb"] != kb:
                raise HTTPException(400, "conversation belongs to a different knowledge base")
        else:
            cid = uuid.uuid4().hex
            conv = conversations[cid] = {"owner": sid, "kb": kb, "messages": []}
        history = conv["messages"][-2 * S.MAX_HISTORY_TURNS:]
        res = agent(req.question, history, kb)
        conv["messages"] += [{"role": "user", "content": req.question},
                             {"role": "assistant", "content": res["answer"]}]
        audit.record(event="chat", kb=kb, figures=[f["key"] for f in res["figures"]],
                     kb_calls=res.get("kb_calls"), seconds=res.get("seconds"), usage=res.get("tokens"), error=res["answer"].startswith("ERROR"))
        figures = [{**f, "url": f"/api/figures/{f['key']}"} for f in res["figures"]]
        return {"conversation_id": cid, "kb": kb, "answer": res["answer"], "citations": res["citations"],
                "figures": figures}

    @app.get("/api/conversations/{cid}")
    def get_conversation(cid: str, sid=Depends(session)):
        conv = conversations.get(cid)
        if not conv or conv["owner"] != sid:
            raise HTTPException(404, "conversation not found")
        return {"conversation_id": cid, "kb": conv["kb"], "messages": conv["messages"]}

    @app.get("/api/figures/{key}")
    def figure(key: str):
        if not FIG_KEY.match(key):
            raise HTTPException(400, "bad figure key")
        data = (figure_fn or fetch_figure)(key)
        if data is None:
            raise HTTPException(404, "figure not found")
        audit.record(event="figure", key=key)
        return Response(content=data, media_type="image/png", headers={"Cache-Control": "private, max-age=300"})

    def xl():
        if excel_mod:
            return excel_mod
        from backend import excel_agent
        return excel_agent

    def xl_lock(sid):
        with excel_guard:
            return excel_locks.setdefault(sid, threading.Lock())

    def xl_sweep():
        for sid, s in list(excel_sessions.items()):
            if time.time() - s["created"] <= S.EXCEL_TTL_SECONDS:
                continue
            lock = xl_lock(sid)
            if lock.acquire(blocking=False):
                try:
                    if excel_sessions.get(sid) is s:
                        excel_sessions.pop(sid)
                        xl().delete(s)
                finally:
                    lock.release()

    @app.post("/api/excel/upload")
    async def excel_upload(request: Request, sid=Depends(session)):
        if int(request.headers.get("content-length") or 0) > S.EXCEL_MAX_BYTES:
            raise HTTPException(413, "file too large (25 MB limit)")
        filename = re.sub(r"[^\w.\- ]", "_", request.headers.get("x-filename", "workbook.xlsx"))[:100]
        data = await request.body()
        if len(data) > S.EXCEL_MAX_BYTES:
            raise HTTPException(413, "file too large (25 MB limit)")
        lock = xl_lock(sid)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "another request for this session is still running")
        try:
            await run_in_threadpool(xl_sweep)
            try:
                sess = await run_in_threadpool(xl().prepare, data, filename)
            except ValueError as e:
                raise HTTPException(400, str(e))
            old = excel_sessions.pop(sid, None)
            if old:
                await run_in_threadpool(xl().delete, old)
            sess["questions"] = 0
            excel_sessions[sid] = sess
        finally:
            lock.release()
        audit.record(event="excel_upload", size=len(data), recalculated=sess["recalculated"])
        return {"filename": filename, "recalculated": sess["recalculated"]}

    @app.post("/api/excel/ask")
    async def excel_ask(req: ExcelAsk, sid=Depends(session)):
        lock = xl_lock(sid)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "another request for this session is still running")
        try:
            sess = excel_sessions.get(sid)
            if not sess:
                raise HTTPException(409, "upload a workbook first")
            if sess["questions"] >= S.EXCEL_MAX_QUESTIONS:
                raise HTTPException(429, "question limit reached for this workbook; upload it again")
            sess["questions"] += 1
            res = await run_in_threadpool(xl().ask, sess, req.question)
        finally:
            lock.release()
        audit.record(event="excel_ask",
                     seconds=res.get("seconds"), usage=res.get("tokens"), error=res["answer"].startswith("ERROR"))
        return {"answer": res["answer"], "code": res["code"], "seconds": res["seconds"]}

    @app.delete("/api/excel")
    async def excel_clear(sid=Depends(session)):
        lock = xl_lock(sid)
        if not lock.acquire(blocking=False):
            raise HTTPException(429, "another request for this session is still running")
        try:
            sess = excel_sessions.pop(sid, None)
            if sess:
                await run_in_threadpool(xl().delete, sess)
        finally:
            lock.release()
        audit.record(event="excel_clear")
        return {"ok": True}

    @app.get("/api/config")
    def public_config(sid=Depends(session)):
        return {"kb": C.KB_NAME}

    @app.get("/")
    def index():
        return FileResponse(WEB / "index.html")

    app.mount("/static", StaticFiles(directory=WEB), name="static")
    return app


def fetch_figure(key):
    """Figure image bytes via the backend identity; the path comes from the index, never from the caller."""
    from common.answer import chunk_meta
    m = chunk_meta(key)
    if m.get("kind") != "figure" or not m.get("image_path"):
        return None
    r = requests.get(f"https://{C.STORAGE_ACCOUNT}.blob.core.windows.net/{C.FIGURES_CONTAINER}/{m['image_path']}",
                     headers={"Authorization": f"Bearer {C.get_token('https://storage.azure.com')}",
                              "x-ms-version": "2023-11-03"}, timeout=30)
    return r.content if r.ok else None


app = create_app()
