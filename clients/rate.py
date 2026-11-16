from __future__ import annotations

import asyncio
import json
import logging
import math
import time
from collections import deque
from collections.abc import Mapping

import tiktoken

from clients.telemetry import span
from pipelines.budget import ServiceError

logger = logging.getLogger(__name__)


class TokenWindow:
    """Single-process admission, not inference retries or shared-account quota guarantees."""

    def __init__(self, tokens: int, requests: int, *, clock=time.monotonic, sleep=asyncio.sleep):
        self.tokens = tokens
        self.requests = requests
        self.clock = clock
        self.sleep = sleep
        self.window = 65.0
        self.entries: deque[tuple[float, int]] = deque()
        self.lock = asyncio.Lock()
        self.blocked_until = 0.0

    async def acquire(self, tokens: int) -> None:
        with span(
            "admission.window", estimated_tokens=tokens,
            token_capacity=self.tokens, request_capacity=self.requests,
        ):
            await self._acquire(tokens)

    async def _acquire(self, tokens: int) -> None:
        if tokens > self.tokens:
            raise ServiceError("estimated input exceeds deployment admission token window")
        async with self.lock:
            while True:
                now = self.clock()
                while self.entries and self.entries[0][0] <= now - self.window:
                    self.entries.popleft()
                wait = max(0.0, self.blocked_until - now)
                if self.entries and (
                    len(self.entries) >= self.requests
                    or sum(item[1] for item in self.entries) + tokens > self.tokens
                ):
                    wait = max(wait, self.entries[0][0] + self.window - now)
                if wait <= 0:
                    self.entries.append((now, tokens))
                    return
                logger.info("deployment_admission_wait seconds=%.2f", wait)
                with span(
                    "admission.wait", wait_seconds=wait,
                    wait_reason="throttle" if self.blocked_until > now else "capacity",
                ):
                    await self.sleep(wait)

    def throttle(self, seconds: float) -> None:
        self.blocked_until = max(self.blocked_until, self.clock() + seconds)


class ModelAdmission:
    def __init__(
        self, *, deployment_capacities: Mapping[str, tuple[int, int]] | None = None
    ):
        """Capacities are observed TPM/RPM; the existing 80% token headroom still applies."""
        self.deployment_capacities = dict(deployment_capacities or {})
        for deployment, capacity in self.deployment_capacities.items():
            if (
                not isinstance(deployment, str) or not deployment
                or not isinstance(capacity, (tuple, list)) or len(capacity) != 2
                or any(type(value) is not int or value <= 0 for value in capacity)
            ):
                raise ValueError("deployment capacity must specify positive integer TPM and RPM")
        self.encoding = tiktoken.get_encoding("o200k_base")
        self.windows: dict[str, TokenWindow] = {}

    async def acquire(self, path: str, body: dict) -> TokenWindow | None:
        if path == "/providers/microsoft/v1/systemone":
            key = body.get("model", "decision")
            capacity, requests, output = 100_000, 100, 256
        elif path.startswith("/openai/deployments/"):
            key = path.split("/")[3]
            capacity = 30_000 if key == "decision-rag-evaluator" else 100_000
            requests = 100
            output = body.get("max_completion_tokens", 0)
        else:
            return None
        capacity, requests = self.deployment_capacities.get(key, (capacity, requests))
        estimate = self.estimate(body, output_tokens=output)
        window = self.windows.setdefault(key, TokenWindow(int(capacity * 0.8), requests))
        await window.acquire(estimate)
        return window

    def estimate(self, body: dict, *, output_tokens: int) -> int:
        """Use the same conservative estimate for placement and actual admission."""
        estimated_body = dict(body)
        image_reserve = 0
        if "messages" in body:
            messages = []
            for message in body["messages"]:
                content = message.get("content")
                if isinstance(content, list):
                    image_reserve += 16_384 * sum(
                        item.get("type") == "image_url" for item in content
                    )
                    content = [
                        {"type": "image_url", "image_url": {"url": "image-reserved-separately"}}
                        if item.get("type") == "image_url"
                        else item
                        for item in content
                    ]
                messages.append({**message, "content": content})
            estimated_body["messages"] = messages
        text = json.dumps(estimated_body, ensure_ascii=False)
        return (
            max(len(self.encoding.encode(text, disallowed_special=())), math.ceil(len(text) / 3))
            + output_tokens
            + image_reserve
            + 256
        )
