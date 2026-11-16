import asyncio

import pytest

from clients.rate import ModelAdmission, TokenWindow
from pipelines.budget import ServiceError


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def __call__(self):
        return self.now

    async def sleep(self, seconds):
        self.waits.append(seconds)
        self.now += seconds


def test_token_window_accounts_tokens_and_expires_before_dispatch():
    async def run():
        clock = Clock()
        window = TokenWindow(100, 10, clock=clock, sleep=clock.sleep)
        await window.acquire(60)
        await window.acquire(40)
        assert clock.waits == []
        await window.acquire(1)
        assert clock.waits == [65]
        assert list(window.entries) == [(65, 1)]

    asyncio.run(run())


def test_request_count_cooldown_and_single_request_overflow_are_explicit():
    async def run():
        clock = Clock()
        window = TokenWindow(100, 1, clock=clock, sleep=clock.sleep)
        await window.acquire(1)
        window.throttle(90)
        await window.acquire(1)
        assert clock.waits == [90]
        with pytest.raises(ServiceError, match="exceeds"):
            await window.acquire(101)
        assert len(window.entries) == 1

    asyncio.run(run())


def test_deployments_have_separate_token_windows_and_bounded_completion_reserve():
    async def run():
        gate = ModelAdmission()
        decision = await gate.acquire("/providers/microsoft/v1/systemone", {"state": "test"})
        grader = await gate.acquire(
            "/openai/deployments/decision-rag-evaluator/chat/completions",
            {"max_completion_tokens": 4000, "messages": []},
        )
        assert decision.tokens == 80_000 and grader.tokens == 24_000
        assert grader.entries[0][1] > 4000
        assert decision is not grader

    asyncio.run(run())


def test_original_image_payload_is_not_mistaken_for_base64_language_tokens():
    async def run():
        gate = ModelAdmission()
        window = await gate.acquire(
            "/openai/deployments/writer/chat/completions",
            {
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {"url": "data:image/png;base64," + "x" * 500_000},
                            },
                        ],
                    }
                ],
                "max_completion_tokens": 6000,
            },
        )
        assert 22_384 < window.entries[0][1] < 30_000

    asyncio.run(run())


def test_placement_estimate_matches_the_actual_admission_reservation():
    async def run():
        gate = ModelAdmission()
        body = {"messages": [{"role": "user", "content": "Evidence " * 8000}],
                "max_completion_tokens": 4000}
        estimate = gate.estimate(body, output_tokens=4000)
        window = await gate.acquire("/openai/deployments/writer/chat/completions", body)
        assert window.entries[0][1] == estimate

    asyncio.run(run())
