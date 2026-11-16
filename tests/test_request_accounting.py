import asyncio
from types import SimpleNamespace

import httpx
import pytest

from clients.models import Models, request_calls
from ingestion.checkpoint import CheckpointQuota, CheckpointQuotaExhausted


class Session:
    resources = SimpleNamespace(embedding_dimensions=3, embedding_deployment="test")

    async def request(self, *args, **kwargs):
        return httpx.Response(
            200,
            json={
                "data": [{"index": 0, "embedding": [0.1, 0.2, 0.3]}],
                "usage": {"total_tokens": 1},
                "model": "fixture",
            },
        )


def test_per_request_limits_preserve_durable_usage_and_private_call_ids(tmp_path):
    models = Models(Session(), CheckpointQuota(tmp_path / "quota.db"))

    async def run():
        with request_calls() as calls:
            for _ in range(5):
                await models.embed(["fixture"])
            with pytest.raises(CheckpointQuotaExhausted, match="per-request"):
                await models.embed(["fixture"])
            assert len(calls.records) == 5
            assert all(record["usage"] == {"total_tokens": 1} for record in calls.records)
            assert len({record["ledger_id"] for record in calls.records}) == 5
        with request_calls():
            await models.embed(["new request"])

    asyncio.run(run())
    assert len(models.quota.snapshot()) == 6


def test_failed_transport_still_consumes_call_and_records_reservation(tmp_path):
    class BrokenSession(Session):
        async def request(self, *args, **kwargs):
            raise TimeoutError("fixture")

    models = Models(BrokenSession(), CheckpointQuota(tmp_path / "quota.db"))

    async def run():
        with request_calls() as calls:
            with pytest.raises(TimeoutError):
                await models.embed(["fixture"])
            assert len(calls.records) == 1
            assert calls.records[0]["state"] == "reserved"

    asyncio.run(run())
    assert models.quota.snapshot()[0]["state"] == "reserved"
