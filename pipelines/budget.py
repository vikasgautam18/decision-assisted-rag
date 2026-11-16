from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Literal, TypeVar

from pydantic import Field

from pipelines.contracts import Artifact

T = TypeVar("T")


class BudgetExhausted(RuntimeError):
    pass


class ServiceError(RuntimeError):
    """A service adapter must translate expected transport failures to this type."""

    def __init__(
        self, message: str, *, status_code: int | None = None, retry_after: float | None = None
    ):
        super().__init__(message)
        self.status_code = status_code
        self.retry_after = retry_after


class BudgetLimits(Artifact):
    timeout_seconds: float = Field(default=5.0, gt=0, le=300)
    max_calls: int = Field(default=2, ge=1, le=100)


@dataclass(frozen=True)
class Operation:
    stage: str
    elapsed_seconds: float
    status: Literal["completed", "failed", "cancelled"]
    usage_status: Literal["synthetic_not_billable"] = "synthetic_not_billable"


@dataclass
class RequestBudget:
    limits: BudgetLimits
    clock: Callable[[], float] = time.monotonic
    calls: int = field(default=0, init=False)
    operations: list[Operation] = field(default_factory=list, init=False)
    started: float = field(init=False)

    def __post_init__(self) -> None:
        self.started = self.clock()

    def check(self) -> float:
        remaining = self.limits.timeout_seconds - (self.clock() - self.started)
        if remaining <= 0:
            raise BudgetExhausted("request deadline exhausted")
        return remaining

    async def call(self, stage: str, operation: Callable[[], Awaitable[T]]) -> T:
        remaining = self.check()
        if self.calls >= self.limits.max_calls:
            raise BudgetExhausted("request call budget exhausted")
        self.calls += 1
        started = self.clock()
        status: Literal["completed", "failed", "cancelled"] = "failed"
        try:
            async with asyncio.timeout(remaining):
                result = await operation()
            self.check()
            status = "completed"
            return result
        except TimeoutError as exc:
            raise BudgetExhausted("service call exceeded request deadline") from exc
        except asyncio.CancelledError:
            status = "cancelled"
            raise
        finally:
            self.operations.append(Operation(stage, self.clock() - started, status))
