from __future__ import annotations

import asyncio
import json
import math
import time
import uuid
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

from clients.azure import AzureSession
from clients.decision import ChoiceResult, parse_choices
from clients.telemetry import normalized_usage, payload_hash, response_ids, span
from ingestion.checkpoint import CheckpointQuota, CheckpointQuotaExhausted
from pipelines.budget import ServiceError

MAX_REPAIRS = 3
MAX_MODEL_CALLS = MAX_REPAIRS + 2


@dataclass
class RequestCalls:
    counts: dict[str, int] = field(default_factory=dict)
    records: list[dict] = field(default_factory=list)


CURRENT_CALLS: ContextVar[RequestCalls | None] = ContextVar("request_calls", default=None)


@contextmanager
def request_calls():
    calls = RequestCalls()
    token = CURRENT_CALLS.set(calls)
    try:
        yield calls
    finally:
        CURRENT_CALLS.reset(token)


class Models:
    def __init__(self, session: AzureSession, quota: CheckpointQuota) -> None:
        self.session = session
        self.quota = quota

    async def _invoke(
        self, category: str, path: str, body: dict, *, role: str | None = None
    ) -> dict:
        role = role or ("decision" if category == "decision" else "writer")
        with span(
            "model.attempt",
            role=role,
            service="ai",
            provider="azure",
            category=category,
            resource_id=getattr(getattr(self.session, "resources", None), "ai_endpoint", None),
            call_id=uuid.uuid4().hex,
            payload_hash=payload_hash(body),
            request_bytes=len(json.dumps(body).encode()),
            deployment=body.get("model")
            or (path.split("/")[3] if path.startswith("/openai/deployments/") else None),
            requested_output_limit=body.get("max_completion_tokens"),
            prompt_hash=payload_hash(body.get("messages", body.get("state"))),
            schema_hash=payload_hash(body.get("response_format", body.get("questions"))),
            generation_settings={
                key: body[key]
                for key in ("max_completion_tokens", "response_format")
                if key in body
            },
            sdk_retries=0,
            api_version="2024-10-21" if category == "model" else None,
            usage=None,
            usage_status="unknown",
            **normalized_usage(None),
        ) as attributes:
            return await self._invoke_attempt(category, path, body, role, attributes)

    async def _invoke_attempt(
        self, category: str, path: str, body: dict, role: str, attributes: dict
    ) -> dict:
        calls = CURRENT_CALLS.get()
        budget = "decision" if role in ("decision", "semantic_decision") else category
        record = {
            **attributes,
            "category": category,
            "path": path,
            "role": role,
            "service": "ai",
            "state": "reserved",
        }
        started = time.monotonic()
        call_id = None
        try:
            with span("model.reservation", role=role, category=category) as reservation:
                if calls is not None:
                    count = calls.counts.get(budget, 0)
                    if count >= {"decision": 20, "model": MAX_MODEL_CALLS}[budget]:
                        raise CheckpointQuotaExhausted("per-request call budget exhausted")
                    calls.counts[budget] = count + 1
                    calls.records.append(record)
                task = asyncio.create_task(asyncio.to_thread(self.quota.reserve, category))
                try:
                    call_id = await asyncio.shield(task)
                except asyncio.CancelledError:
                    # Reservation may already have committed in the worker thread.
                    call_id = await task
                    raise
                reservation["ledger_id"] = call_id
                record.update(ledger_id=call_id, state="reserved")
                attributes["ledger_id"] = call_id
            response = await self.session.request(
                "ai",
                "POST",
                path,
                params={"api-version": "2024-10-21"} if category == "model" else None,
                json=body,
            )
            record.update(
                http_status=response.status_code,
                request_ids=response_ids(response.headers),
                request_id=response.headers.get("x-ms-request-id"),
                response_bytes=len(response.content),
            )
            with span("model.parse", role=role):
                try:
                    result = response.json()
                except ValueError:
                    raise ServiceError("model returned invalid JSON") from None
                if not isinstance(result, dict):
                    raise ServiceError("model response must be an object")
            choices = result.get("choices")
            record.update(
                model=result.get("model"),
                usage=result.get("usage"),
                usage_status="reported" if result.get("usage") is not None else "unknown",
                response_id=result.get("id"),
                finish_reason=(
                    choices[0].get("finish_reason")
                    if isinstance(choices, list) and choices and isinstance(choices[0], dict)
                    else None
                ),
                **normalized_usage(result.get("usage")),
                state="completed",
            )
            return result
        except BaseException as exc:
            terminal_state = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
            record.update(
                state=terminal_state
                if ("trace_id" in attributes or isinstance(exc, ServiceError))
                else "reserved",
                terminal_state=terminal_state,
                error_type=type(exc).__name__,
            )
            if isinstance(exc, ServiceError):
                record.update(
                    http_status=exc.status_code,
                    retry_after=exc.retry_after,
                    request_ids=getattr(exc, "request_ids", {}),
                    request_id=getattr(exc, "request_ids", {}).get("x-ms-request-id"),
                )
            raise
        finally:
            record["elapsed_seconds"] = time.monotonic() - started
            attributes.update(record)
            if call_id is not None and (record["state"] == "completed" or "trace_id" in attributes):
                record["ledger_id"] = call_id
                attributes["ledger_id"] = call_id
                completion = asyncio.create_task(
                    asyncio.to_thread(self.quota.complete, call_id, record.copy())
                )
                try:
                    await asyncio.shield(completion)
                except asyncio.CancelledError:
                    await completion
                    raise

    async def embed(self, texts: list[str]) -> list[list[float]]:
        with span("model.embedding", role="embedding"):
            return await self._embed(texts)

    async def _embed(self, texts: list[str]) -> list[list[float]]:
        if not texts or len(texts) > 16 or sum(map(len, texts)) > 100_000:
            raise ValueError("embedding batch exceeds pilot limits")
        result = await self._invoke(
            "model",
            f"/openai/deployments/{self.session.resources.embedding_deployment}/embeddings",
            {"input": texts, "dimensions": self.session.resources.embedding_dimensions},
            role="embedding",
        )
        with span("model.embedding.parse", role="embedding"):
            return self._embedding_vectors(result, len(texts))

    def _embedding_vectors(self, result: dict, batch_size: int) -> list[list[float]]:
        data = result.get("data")
        if not isinstance(data, list) or len(data) != batch_size:
            raise ServiceError("embedding response has an invalid batch length")
        if any(
            not isinstance(item, dict)
            or type(item.get("index")) is not int
            or not isinstance(item.get("embedding"), list)
            for item in data
        ):
            raise ServiceError("embedding response contains invalid records")
        ordered = sorted(data, key=lambda item: item["index"])
        if [item["index"] for item in ordered] != list(range(batch_size)):
            raise ServiceError("embedding indexes do not match request")
        vectors = [item["embedding"] for item in ordered]
        if any(
            len(vector) != self.session.resources.embedding_dimensions
            or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
            for vector in vectors
        ):
            raise ServiceError("embedding dimension mismatch")
        return vectors

    async def complete(self, system: str, content: str | list, *, structured: bool = False) -> str:
        with span("model.writer", role="writer"):
            return await self._complete(system, content, structured=structured)

    async def semantic_complete(self, system: str, content: str) -> str:
        with span("model.semantic_decision", role="semantic_decision"):
            return await self._complete(system, content, role="semantic_decision")

    async def _complete(
        self, system: str, content: str | list, *, structured: bool = False, role: str = "writer"
    ) -> str:
        if len(json.dumps(content)) > 2_000_000:
            raise ValueError("model input exceeds pilot limits")
        body = {
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": content},
            ],
            "max_completion_tokens": 6000,
        }
        if structured:
            body["response_format"] = {"type": "json_object"}
        result = await self._invoke(
            "model",
            f"/openai/deployments/{self.session.resources.writer_deployment}/chat/completions",
            body,
            role=role,
        )
        with span("model.writer.parse", role=role):
            return self._completion_text(result)

    @staticmethod
    def _completion_text(result: dict) -> str:
        choices = result.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            raise ServiceError("writer returned an invalid choice list")
        choice = choices[0]
        if choice.get("finish_reason") != "stop":
            raise ServiceError("writer did not finish its answer")
        text = choice.get("message", {}).get("content")
        if not isinstance(text, str) or not text.strip():
            raise ServiceError("writer returned empty content")
        return text

    async def decide(
        self, state: dict, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceResult]:
        with span("model.decision", role="decision"):
            return await self._decide(state, questions)

    async def _decide(
        self, state: dict, questions: dict[str, tuple[str, dict[str, str]]]
    ) -> dict[str, ChoiceResult]:
        if not 1 <= len(questions) <= 16 or len(json.dumps(state)) > 80_000:
            raise ValueError("decision request exceeds bounded pilot limits")
        result = await self._invoke(
            "decision",
            "/providers/microsoft/v1/systemone",
            {
                "model": self.session.resources.decision_deployment,
                "state": state,
                "questions": {
                    key: {"type": "choice", "instructions": instructions, "criteria": criteria}
                    for key, (instructions, criteria) in questions.items()
                },
            },
        )
        with span("model.decision.parse", role="decision"):
            return parse_choices(result, {key: value[1] for key, value in questions.items()})
