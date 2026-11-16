"""Opt-in logging-only Foundry hooks for an isolated, pinned baseline process.

No Azure configuration changes, backend selection, retry or request mutations.
Never serialize requests: MCP authorization may be inside the request body.
"""

from __future__ import annotations

import contextvars
import functools
import hashlib
import importlib
import json
import os
import re
import sys
import uuid
from contextlib import contextmanager
from pathlib import Path

_invocation = contextvars.ContextVar("baseline_invocation", default=None)
_IDENTIFIER = re.compile(r"[A-Za-z0-9_.:-]{1,160}\Z")
CORRELATION_HEADERS = {
    "x-experiment-id": "experiment_id",
    "x-experiment-phase": "phase",
    "x-experiment-arm": "arm",
    "x-experiment-pass": "pass",
    "x-experiment-case": "case_id",
    "x-logical-request-id": "logical_request_id",
    "x-experiment-attempt": "attempt_id",
    "x-observation-id": "observation_id",
}
PROVIDER_HEADERS = ("x-request-id", "x-ms-request-id", "apim-request-id", "request-id")
VISIBILITY = {
    "api_request_and_runner_join": "measured: exact validated headers, trace and observation IDs",
    "native_attempts": "measured: one SDK create span per original native-loop iteration",
    "sdk_http_attempts": "measured: underlying synchronous HTTPX send, including SDK retries",
    "sdk_backoff": "unavailable: bounded by SDK create duration, not a fabricated sleep span",
    "native_backoff": "measured: module-local proxy around the original time.sleep callable",
    "provider_ids_and_usage": "service-reported: only fields exposed by SDK/result/HTTP headers",
    "output": "service-reported: message text and allowlisted item metadata; no MCP payloads",
    "hosted_model_tool_kb_spans": "unavailable without separately joinable provider exports",
    "cache_reasoning_audio_image_usage": "service-reported if present; missing is null, never zero",
    "failed_attempt_usage": "unavailable unless provider exposes usage on a successful SDK result",
    "dns_tls_pool_timing": "unavailable: HTTPX send boundary is not connection-level timing",
    "preflight": "unavailable offline: deployed agent/resources need approved preflight",
}


def _telemetry(telemetry):
    return telemetry if telemetry is not None else importlib.import_module("clients.telemetry")


def _identifier(value):
    return value if isinstance(value, str) and _IDENTIFIER.fullmatch(value) else None


def _mapping(value):
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    return value if isinstance(value, dict) else {}


def _usage(value):
    """Retain all numeric usage categories, including future nested counters.

    Provider free text is not usage and is deliberately excluded. No missing
    category is synthesized as zero.
    """
    result = {}
    for key, item in _mapping(value).items():
        if not isinstance(key, str) or not re.fullmatch(r"[a-zA-Z0-9_]{1,100}", key):
            continue
        if item is None or isinstance(item, (bool, int, float)):
            result[key] = item
        elif isinstance(item, dict):
            result[key] = _usage(item)
    return result


def safe_result(response) -> dict:
    """Project SDK response fields, without dumping tool arguments/results or auth."""
    from clients.telemetry import normalized_usage

    data = {}
    for key in ("id", "model", "status", "service_tier", "_request_id"):
        value = _identifier(getattr(response, key, None))
        if value is not None:
            data[key] = value
    data["provider_ids"] = {
        target: value
        for source, target in (
            ("id", "response_id"),
            ("_request_id", "request_id"),
            ("run_id", "run_id"),
            ("activity_id", "activity_id"),
        )
        if (value := _identifier(getattr(response, source, None))) is not None
    }
    data["usage"] = _usage(getattr(response, "usage", None)) or None
    usage = data["usage"] or {}
    data["normalized_usage"] = {
        **normalized_usage(usage),
        "total_tokens": usage.get("total_tokens"),
    }
    output = []
    for item in getattr(response, "output", None) or []:
        item = _mapping(item)
        safe = {
            key: value
            for key in (
                "id",
                "type",
                "status",
                "role",
                "name",
                "call_id",
                "server_label",
                "response_id",
                "run_id",
                "activity_id",
                "finish_reason",
            )
            if (value := _identifier(item.get(key))) is not None
        }
        if item.get("type") == "message":
            safe["content"] = [
                {"type": "output_text", "text": part["text"]}
                for part in item.get("content", [])
                if isinstance(part, dict)
                and part.get("type") == "output_text"
                and isinstance(part.get("text"), str)
            ]
        output.append(safe)
    data["output"] = output
    return data


def _record_error(state, exc):
    state["error_type"] = type(exc).__name__
    status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        state["http_status"] = status
    request_id = _identifier(getattr(exc, "request_id", None))
    if request_id:
        state["provider_request_id"] = request_id


class _TelemetryFailure(BaseException):
    """Cross SDK's broad Exception retry handler without triggering a paid retry."""

    def __init__(self, cause):
        self.cause = cause


@contextmanager
def _span(telemetry, name, **attributes):
    try:
        manager = telemetry.span(name, **attributes)
        state = manager.__enter__()
    except Exception as exc:
        raise _TelemetryFailure(exc) from exc
    try:
        yield state
    except BaseException as body_error:
        try:
            suppressed = manager.__exit__(*sys.exc_info())
        except Exception as exc:
            if exc is body_error:
                raise
            raise _TelemetryFailure(exc) from exc
        if not suppressed:
            raise
    else:
        try:
            manager.__exit__(None, None, None)
        except Exception as exc:
            raise _TelemetryFailure(exc) from exc


def _instrument_client(client, telemetry):
    """Hook the existing client instance; do not construct/configure a replacement."""
    create = client.responses.create
    transport = client._client
    send = transport.send

    @functools.wraps(send)
    def observed_send(request, *args, **kwargs):
        context = _invocation.get()
        if context is None:
            return send(request, *args, **kwargs)
        context["sdk_attempt"] += 1
        # HTTPX has already encoded the body. Never inspect JSON or log headers
        # other than the fixed provider response ID allowlist.
        try:
            body = request.content
        except Exception:
            body = None  # Do not consume a stream to obtain a hash.
        attributes = {
            "call_id": context["call_id"],
            "native_attempt": context["native_attempt"],
            "sdk_attempt": context["sdk_attempt"],
            "retry_origin": "sdk",
            "method": request.method,
            "transport_call_id": uuid.uuid4().hex,
            "endpoint_host": request.url.host,
        }
        if isinstance(body, bytes):
            attributes.update(
                request_sha256=hashlib.sha256(body).hexdigest(),
                request_bytes=len(body),
            )
        with _span(telemetry, "baseline.sdk_http_attempt", **attributes) as state:
            try:
                response = send(request, *args, **kwargs)
                state["http_status"] = response.status_code
                state["transport_status"] = "ok" if response.status_code < 400 else "http_error"
                state["provider_ids"] = {
                    key: value
                    for key in PROVIDER_HEADERS
                    if (value := _identifier(response.headers.get(key))) is not None
                }
                return response
            except BaseException as exc:
                _record_error(state, exc)
                raise

    @functools.wraps(create)
    def observed_create(*args, **kwargs):
        context = _invocation.get()
        if context is None:
            return create(*args, **kwargs)
        context["native_attempt"] += 1
        context["sdk_attempt"] = 0
        context["call_id"] = uuid.uuid4().hex
        with _span(
            telemetry,
            "baseline.sdk_create",
            call_id=context["call_id"],
            retry_origin="native",
            native_attempt=context["native_attempt"],
            backend="foundry",
        ) as state:
            try:
                state["generation_settings"] = {
                    key: value
                    for key in ("max_output_tokens", "temperature", "top_p", "timeout")
                    if isinstance(value := kwargs.get(key), (int, float))
                }
                response = create(*args, **kwargs)
                state.update(safe_result(response))
                state["sdk_http_attempts"] = context["sdk_attempt"]
                return response
            except BaseException as exc:
                _record_error(state, exc)
                raise

    client.responses.create = observed_create
    try:
        transport.send = observed_send
    except BaseException:
        client.responses.create = create
        raise

    def restore():
        client.responses.create = create
        transport.send = send

    return restore


class _TimeProxy:
    def __init__(self, original, telemetry):
        self._original = original
        self._telemetry = telemetry

    def __getattr__(self, name):
        return getattr(self._original, name)

    def sleep(self, *args, **kwargs):
        context = _invocation.get()
        if context is None:
            return self._original.sleep(*args, **kwargs)
        attributes = {
            "call_id": context["call_id"],
            "retry_origin": "native",
            "retry_index": context["native_attempt"],
        }
        seconds = args[0] if args else kwargs.get("secs")
        if isinstance(seconds, (int, float)):
            attributes["wait_seconds"] = seconds
        with _span(self._telemetry, "baseline.native_backoff", **attributes) as state:
            try:
                return self._original.sleep(*args, **kwargs)
            except BaseException as exc:
                _record_error(state, exc)
                raise


@contextmanager
def instrument_foundry(ask_module, project_client_class, *, telemetry=None):
    """Process-local launcher hook; install once before serving, never on a live server.

    ``ask_module`` is the COPY's backend.agent.ask and ``project_client_class`` is
    azure.ai.projects.AIProjectClient. A configured non-Foundry backend is rejected,
    not silently switched. Restore both methods on exit. The shared telemetry
    exporter owns private append-only files and propagates write failures.
    """
    if os.getenv("AGENT_BACKEND", "foundry") != "foundry":
        raise ValueError("baseline A requires the unchanged foundry backend")
    telemetry = _telemetry(telemetry)
    original_run = ask_module.run_foundry
    original_get = project_client_class.get_openai_client
    original_time = ask_module.time

    @functools.wraps(original_get)
    def get_client(self, *args, **kwargs):
        client = original_get(self, *args, **kwargs)
        context = _invocation.get()
        if context is not None and id(client) not in context["seen_clients"]:
            context["restorations"].append(_instrument_client(client, telemetry))
            context["seen_clients"].add(id(client))
        return client

    @functools.wraps(original_run)
    def run(*args, **kwargs):
        context = {
            "native_attempt": 0,
            "sdk_attempt": 0,
            "call_id": None,
            "restorations": [],
            "seen_clients": set(),
        }
        token = _invocation.set(context)
        try:
            with _span(telemetry, "baseline.agent", backend="foundry") as state:
                try:
                    result = original_run(*args, **kwargs)
                    state["native_attempts"] = context["native_attempt"]
                    state["baseline_error_response"] = result.get("answer", "").startswith("ERROR")
                    return result
                except BaseException as exc:
                    _record_error(state, exc)
                    raise
        except _TelemetryFailure as exc:
            raise exc.cause from None
        finally:
            try:
                for restore in reversed(context["restorations"]):
                    restore()
            finally:
                _invocation.reset(token)

    ask_module.run_foundry = run
    ask_module.time = _TimeProxy(original_time, telemetry)
    project_client_class.get_openai_client = get_client
    try:
        yield
    finally:
        ask_module.run_foundry = original_run
        ask_module.time = original_time
        project_client_class.get_openai_client = original_get


@contextmanager
def create_baseline_launcher(
    app,
    ask_module,
    project_client_class,
    directory: Path,
    *,
    metadata: dict,
    telemetry=None,
    header_map=None,
):
    """Yield the wrapped ASGI app while COPY-only hooks remain installed.

    This factory neither starts a service nor probes Azure. Keep its context open
    until the isolated server and all active requests have stopped.
    """
    middleware = BaselineObservationMiddleware(
        app,
        directory,
        metadata=metadata,
        telemetry=telemetry,
        header_map=header_map,
    )
    with instrument_foundry(ask_module, project_client_class, telemetry=telemetry):
        yield middleware


class BaselineObservationMiddleware:
    """Transparent ASGI boundary using server-chosen storage, never request paths.

    Only /api/chat with a complete runner identity is observed. Other requests
    (including existing browser clients) pass through unchanged. Header names
    can be explicitly mapped by the coordinator; no request body is read.
    """

    def __init__(
        self,
        app,
        directory: Path,
        *,
        metadata: dict,
        telemetry=None,
        header_map=None,
    ):
        self.app = app
        self.directory = Path(directory)
        self.metadata = dict(metadata)
        if self.metadata.get("arm") != "A":
            raise ValueError("baseline server metadata must select arm A")
        if not _identifier(self.metadata.get("experiment_id")):
            raise ValueError("baseline server requires a valid experiment_id")
        if not re.fullmatch(r"[a-f0-9]{64}", self.metadata.get("implementation_sha256", "")):
            raise ValueError("baseline server requires the copy manifest implementation_sha256")
        self.telemetry = _telemetry(telemetry)
        self.header_map = dict(CORRELATION_HEADERS if header_map is None else header_map)
        if set(self.header_map.values()) != set(CORRELATION_HEADERS.values()):
            raise ValueError("header map must supply exactly the baseline identity fields")

    async def __call__(self, scope, receive, send):
        if (
            scope["type"] == "http"
            and scope.get("path") == "/health"
            and scope.get("method") == "GET"
        ):
            body = json.dumps(
                {
                    key: self.metadata[key]
                    for key in ("arm", "experiment_id", "implementation_sha256")
                }
            ).encode()
            await send(
                {
                    "type": "http.response.start",
                    "status": 200,
                    "headers": [(b"content-type", b"application/json")],
                }
            )
            await send({"type": "http.response.body", "body": body})
            return
        if scope["type"] != "http" or scope.get("path") != "/api/chat":
            return await self.app(scope, receive, send)
        headers = {}
        duplicates = False
        for name, value in scope.get("headers", []):
            name = name.decode("latin-1").lower()
            if name in self.header_map:
                duplicates |= name in headers
                headers[name] = value.decode("latin-1")
        if not headers:
            return await self.app(scope, receive, send)
        if duplicates or set(headers) != set(self.header_map):
            raise ValueError("incomplete or duplicate experiment correlation headers")
        identity = {field: _identifier(headers[name]) for name, field in self.header_map.items()}
        if None in identity.values() or identity["arm"] != "A":
            raise ValueError("invalid baseline runner identity")
        if (
            not re.fullmatch(r"[a-f0-9]{32}", identity["observation_id"])
            or not re.fullmatch(r"[a-f0-9]{32}", identity["logical_request_id"])
            or identity["pass"] not in {"1", "2", "3", "4", "5"}
            or not re.fullmatch(r"q\d+", identity["case_id"])
            or identity["attempt_id"] not in {"1", "2"}
            or identity["phase"] not in {"pilot", "main", "load"}
        ):
            raise ValueError("invalid baseline runner identity fields")
        for key in ("experiment_id", "phase", "arm"):
            if key in self.metadata and identity[key] != self.metadata[key]:
                raise ValueError(f"runner {key} does not match server configuration")
        with self.telemetry.observation(
            self.directory,
            {**self.metadata, **identity},
        ) as recorder:
            with self.telemetry.span(
                "baseline.api_request",
                observation_id=identity["observation_id"],
                method=scope.get("method"),
                trace_id=recorder.trace_id,
            ) as state:

                async def observed_send(message):
                    if message["type"] == "http.response.start":
                        state["http_status"] = message["status"]
                        headers = [
                            (key, value)
                            for key, value in message.get("headers", [])
                            if key.lower()
                            not in {
                                b"x-observation-id",
                                b"x-experiment-trace-id",
                            }
                        ]
                        message = {
                            **message,
                            "headers": [
                                *headers,
                                (b"x-observation-id", identity["observation_id"].encode()),
                                (b"x-experiment-trace-id", recorder.trace_id.encode()),
                            ],
                        }
                    await send(message)

                await self.app(scope, receive, observed_send)
