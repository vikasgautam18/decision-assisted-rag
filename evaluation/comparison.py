from __future__ import annotations

import asyncio
import random
import re
import time
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, model_validator

from evaluation.dataset import Case
from ingestion.corpus import DOSSIERS
from pipelines.arms import Arm


class CitationSpan(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    document: str
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_span(self):
        if (
            self.document not in DOSSIERS
            or self.page_end < self.page_start
            or self.page_end > DOSSIERS[self.document][1]
        ):
            raise ValueError("citation span outside pinned corpus")
        return self


class PublicAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str = Field(min_length=1)
    citations: tuple[CitationSpan, ...]
    figure_keys: tuple[str, ...]


def normalize_response(arm: Arm, payload: dict) -> PublicAnswer:
    answer = payload.get("answer")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("missing application answer")
    if answer.lstrip().startswith("ERROR") or (arm != "A" and payload.get("status") != "completed"):
        raise ValueError("application reported an error")
    citations = payload.get("citations")
    figures = payload.get("figures")
    if not isinstance(citations, list) or not isinstance(figures, list):
        raise ValueError("invalid application citation or figure list")
    spans = []
    for item in citations:
        if not isinstance(item, dict):
            raise ValueError("invalid citation record")
        spans.append(
            CitationSpan.model_validate(
                {
                    "document": item.get("doc" if arm == "A" else "document"),
                    "page_start": item.get("page_start"),
                    "page_end": item.get("page_end"),
                }
            )
        )
    keys = []
    pattern = (
        r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-fig\d{2}"
        if arm == "A"
        else r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*-figure-\d+"
    )
    for item in figures:
        if not isinstance(item, dict):
            raise ValueError("invalid figure record")
        key = item.get("key")
        if (
            not isinstance(key, str)
            or not re.fullmatch(pattern, key)
            or item.get("url") != f"/api/figures/{key}"
        ):
            raise ValueError("invalid original-figure identity or route")
        keys.append(key)
    return PublicAnswer(answer=answer, citations=tuple(spans), figure_keys=tuple(keys))


def paired_schedule(cases: list[Case], *, seed: int) -> list[tuple[str, str]]:
    if not cases or len({case.id for case in cases}) != len(cases):
        raise ValueError("comparison cases must be nonempty and unique")
    rng = random.Random(seed)
    ordered = rng.sample(cases, len(cases))
    first_arms = ["A"] * (len(cases) // 2) + ["F"] * (len(cases) - len(cases) // 2)
    rng.shuffle(first_arms)
    return [
        (case.id, arm)
        for case, first in zip(ordered, first_arms, strict=True)
        for arm in (first, "F" if first == "A" else "A")
    ]


def validate_origin(url: str) -> str:
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.path not in ("", "/")
        or parsed.query
        or parsed.fragment
        or parsed.scheme == "http"
        and parsed.hostname not in ("127.0.0.1", "localhost")
    ):
        raise ValueError("comparison needs an HTTPS origin or loopback HTTP origin")
    return url.rstrip("/")


async def capture_public_answer(
    arm: Arm,
    origin: str,
    case: Case,
    *,
    timeout_seconds: float,
    transport: httpx.AsyncBaseTransport | None = None,
    correlation_headers: dict[str, str] | None = None,
) -> dict:
    origin = validate_origin(origin)
    if timeout_seconds <= 0:
        raise ValueError("comparison timeout must be positive")
    async with httpx.AsyncClient(
        base_url=origin,
        headers={"Origin": origin, **(correlation_headers or {})},
        timeout=timeout_seconds,
        follow_redirects=False,
        trust_env=False,
        transport=transport,
    ) as client:
        started = time.monotonic()
        try:
            async with asyncio.timeout(timeout_seconds):
                response = await client.post(
                    "/api/chat", json={"question": case.question, "conversation_id": None}
                )
        except (httpx.RequestError, TimeoutError) as exc:
            return {
                "status": "error",
                "error_type": type(exc).__name__,
                "application_seconds": time.monotonic() - started,
                "http_status": None,
                "usage": None,
            }
        seconds = time.monotonic() - started
        record = {
            "application_seconds": seconds,
            "http_status": response.status_code,
            "raw_body": response.text,
            "usage": None,
            "correlation": {
                key: response.headers.get(key)
                for key in ("x-observation-id", "x-experiment-trace-id")
            },
        }
        if response.status_code != 200:
            return {**record, "status": "error", "error_type": "http_error"}
        try:
            payload = response.json()
            if not isinstance(payload, dict):
                raise ValueError("application response is not an object")
            answer = normalize_response(arm, payload)
        except ValueError as exc:
            return {
                **record,
                "status": "error",
                "error_type": "invalid_application_output",
                "error_detail": str(exc),
            }
        return {
            **record,
            "status": "completed",
            "public_answer": answer.model_dump(mode="json"),
        }
