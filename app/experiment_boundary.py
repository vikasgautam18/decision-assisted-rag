from __future__ import annotations

import re
import time
from pathlib import Path

from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

IDENTITY_HEADERS = {
    "experiment_id": "x-experiment-id",
    "observation_id": "x-observation-id",
    "logical_request_id": "x-logical-request-id",
    "arm": "x-experiment-arm",
    "pass_number": "x-experiment-pass",
    "case_id": "x-experiment-case",
    "attempt": "x-experiment-attempt",
    "phase": "x-experiment-phase",
}


class ExperimentTelemetryMiddleware:
    def __init__(self, app: ASGIApp, *, experiment_id: str, arm: str, directory: Path) -> None:
        self.app, self.experiment_id, self.arm, self.directory = (
            app,
            experiment_id,
            arm,
            directory,
        )

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http" or scope["path"] != "/api/chat":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        values = {key: headers.get(header, "") for key, header in IDENTITY_HEADERS.items()}
        valid = (
            values["experiment_id"] == self.experiment_id
            and values["arm"] == self.arm
            and re.fullmatch(r"[a-f0-9]{32}", values["observation_id"])
            and re.fullmatch(r"[a-f0-9]{32}", values["logical_request_id"])
            and re.fullmatch(r"q\d{2,}", values["case_id"])
            and values["pass_number"] in ("1", "2", "3", "4", "5")
            and values["attempt"] in ("1", "2")
            and values["phase"] in ("pilot", "main", "load")
        )
        if not valid:
            await JSONResponse(status_code=400, content={"error": "invalid_experiment_identity"})(
                scope, receive, send
            )
            return
        from clients.telemetry import observation, span

        with observation(self.directory, values) as recorder:
            with span("api.chat", arm=self.arm) as attributes:
                started = time.monotonic()

                async def observed_send(message: Message) -> None:
                    if message["type"] == "http.response.start":
                        attributes["http_status"] = message["status"]
                        attributes["response_headers_seconds"] = time.monotonic() - started
                        outgoing = MutableHeaders(scope=message)
                        outgoing["X-Observation-ID"] = values["observation_id"]
                        outgoing["X-Experiment-Trace-ID"] = recorder.trace_id
                    await send(message)
                    if message["type"] == "http.response.body" and not message.get("more_body"):
                        attributes["response_body_sent_seconds"] = time.monotonic() - started

                await self.app(scope, receive, observed_send)
