import asyncio
import json
import sqlite3
import threading
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from azure.core.exceptions import ClientAuthenticationError

from app.resources import ExperimentResources
from clients import telemetry
from clients.azure import AzureSession
from clients.models import Models, request_calls
from clients.rate import ModelAdmission, TokenWindow
from clients.search import Search
from clients.telemetry import observation
from ingestion.checkpoint import CheckpointQuota, CheckpointQuotaExhausted
from pipelines.budget import ServiceError

USAGE = {
    "prompt_tokens": 12, "completion_tokens": 9, "total_tokens": 21,
    "prompt_tokens_details": {"cached_tokens": 0},
    "completion_tokens_details": {"reasoning_tokens": 4, "accepted_prediction_tokens": 1},
}
ANSWER = {
    "id": "response-1", "model": "fixture-version", "usage": USAGE,
    "choices": [{"finish_reason": "stop", "message": {"content": "yes"}}],
}


class FixtureSession:
    resources = SimpleNamespace(
        writer_deployment="writer", embedding_deployment="embedding",
        embedding_dimensions=2, decision_deployment="decision", index_name="index",
    )

    def __init__(self, response=None):
        self.response = response if response is not None else httpx.Response(200, json=ANSWER)
        self.calls = []

    async def request(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.response


def summaries(quota):
    with sqlite3.connect(quota.path) as connection:
        return [
            json.loads(row[0])
            for row in connection.execute("SELECT summary FROM calls ORDER BY id")
        ]


def rows(recorder):
    return [json.loads(line) for line in recorder.path.read_text().splitlines()]


def test_semantic_decisions_bill_model_without_consuming_writer_allowance(tmp_path):
    quota = CheckpointQuota(tmp_path / "calls.db")
    session = FixtureSession()
    models = Models(session, quota)

    async def scenario():
        with observation(tmp_path / "traces", {"arm": "G"}) as recorder, request_calls() as calls:
            for _ in range(20):
                assert await models.semantic_complete("private system", "private query") == "yes"
            for _ in range(5):
                assert await models.complete("private system", "private query") == "yes"
            with pytest.raises(CheckpointQuotaExhausted):
                await models.semantic_complete("private system", "private query")
            with pytest.raises(CheckpointQuotaExhausted):
                await models.complete("private system", "private query")
            assert calls.counts == {"decision": 20, "model": 5}
            assert calls.records[0]["role"] == "semantic_decision"
            assert calls.records[0]["trace_id"] == recorder.trace_id
            assert calls.records[0]["service"] == "ai"
            assert len(calls.records) == 25
            assert calls.records[-1]["state"] == "completed"
            return recorder

    recorder = asyncio.run(scenario())
    assert len(session.calls) == 25
    assert all(call[1]["params"] == {"api-version": "2024-10-21"} for call in session.calls)
    assert len(quota.snapshot()) == 25
    assert {item["category"] for item in quota.snapshot()} == {"model"}
    assert summaries(quota)[0]["usage"] == USAGE
    assert summaries(quota)[0]["trace_id"] == recorder.trace_id
    assert "private system" not in recorder.path.read_text()
    assert "private query" not in recorder.path.read_text()


def test_native_and_semantic_decisions_share_twenty_invocation_ceiling(tmp_path):
    quota = CheckpointQuota(tmp_path / "calls.db")
    session = FixtureSession()
    models = Models(session, quota)

    async def scenario():
        with request_calls() as calls:
            await models._invoke(
                "decision", "/providers/microsoft/v1/systemone", {"model": "decision"}
            )
            for _ in range(19):
                await models.semantic_complete("system", "question")
            with pytest.raises(CheckpointQuotaExhausted):
                await models.semantic_complete("system", "question")
            assert calls.counts == {"decision": 20}
            assert calls.records[0]["role"] == "decision"
            assert calls.records[1]["role"] == "semantic_decision"

    asyncio.run(scenario())
    assert [row["category"] for row in quota.snapshot()] == ["decision"] + ["model"] * 19


@pytest.mark.parametrize("failure", ["json", "shape", "transport", "cancel", "quota"])
def test_every_failure_preserves_terminal_trace_and_call_record(tmp_path, failure):
    quota = CheckpointQuota(tmp_path / "calls.db")
    session = FixtureSession()
    if failure == "json":
        session.response = httpx.Response(200, content=b"secret malformed json")
    elif failure == "shape":
        session.response = httpx.Response(200, json=[])
    elif failure in ("transport", "cancel"):
        async def fail(*args, **kwargs):
            if failure == "cancel":
                raise asyncio.CancelledError("secret cancellation")
            raise ServiceError("safe error", status_code=429, retry_after=65)
        session.request = fail
    elif failure == "quota":
        quota.limits = {**quota.limits, "model": 0}
    models = Models(session, quota)

    async def scenario():
        with observation(tmp_path / "traces", {}) as recorder, request_calls() as calls:
            with pytest.raises((ServiceError, asyncio.CancelledError, CheckpointQuotaExhausted)):
                await models.complete("private system", "private query")
            assert calls.records[0]["state"] == ("cancelled" if failure == "cancel" else "failed")
            assert calls.records[0]["usage_status"] == "unknown"
            assert calls.records[0]["input_tokens"] is None
            return recorder

    recorder = asyncio.run(scenario())
    ends = [row for row in rows(recorder) if row["event"] == "span_end"]
    attempt = next(row for row in ends if row["name"] == "model.attempt")
    assert attempt["status"] == ("cancelled" if failure == "cancel" else "error")
    if failure != "quota":
        assert summaries(quota)[0]["state"] in ("failed", "cancelled")
    assert "secret" not in recorder.path.read_text()


@pytest.mark.parametrize("after_dispatch", [False, True])
def test_model_export_failure_is_never_swallowed(tmp_path, monkeypatch, after_dispatch):
    quota = CheckpointQuota(tmp_path / "calls.db")
    session = FixtureSession()
    original = telemetry.Recorder.emit

    def fail(recorder, event):
        if event["name"] == "model.attempt" and event["event"] == (
            "span_end" if after_dispatch else "span_start"
        ):
            raise OSError("export unavailable")
        original(recorder, event)

    monkeypatch.setattr(telemetry.Recorder, "emit", fail)

    async def scenario():
        with observation(tmp_path / "traces", {}):
            with pytest.raises(OSError, match="export unavailable"):
                await Models(session, quota).complete("system", "query")

    asyncio.run(scenario())
    assert len(session.calls) == int(after_dispatch)


def test_cancelled_reservation_is_joined_and_finalized_before_return(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class SlowQuota(CheckpointQuota):
        def reserve(self, category):
            entered.set()
            assert release.wait(5)
            return super().reserve(category)

    quota = SlowQuota(tmp_path / "calls.db")
    session = FixtureSession()

    async def scenario():
        with observation(tmp_path / "traces", {}) as recorder, request_calls() as calls:
            task = asyncio.create_task(Models(session, quota).complete("system", "question"))
            assert await asyncio.to_thread(entered.wait, 5)
            task.cancel()
            release.set()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert calls.records[0]["ledger_id"] == 1
            return recorder

    recorder = asyncio.run(scenario())
    assert session.calls == []
    assert summaries(quota)[0]["state"] == "cancelled"
    assert any(row["status"] == "cancelled" for row in rows(recorder))


def test_export_failure_after_reservation_stops_dispatch_and_keeps_ledger_join(
    tmp_path, monkeypatch
):
    quota = CheckpointQuota(tmp_path / "calls.db")
    session = FixtureSession()
    original = telemetry.Recorder.emit

    def fail(recorder, event):
        if event["name"] == "model.reservation" and event["event"] == "span_end":
            raise OSError("reservation export unavailable")
        original(recorder, event)

    monkeypatch.setattr(telemetry.Recorder, "emit", fail)

    async def scenario():
        with observation(tmp_path / "traces", {}):
            with pytest.raises(OSError):
                await Models(session, quota).complete("system", "question")

    asyncio.run(scenario())
    assert session.calls == []
    assert summaries(quota)[0]["ledger_id"] == 1
    assert summaries(quota)[0]["state"] == "failed"


def test_token_window_records_actual_capacity_wait(tmp_path):
    clock = [0.0]

    async def sleep(seconds):
        clock[0] += seconds

    async def scenario():
        window = TokenWindow(100, 1, clock=lambda: clock[0], sleep=sleep)
        with observation(tmp_path, {}) as recorder:
            await window.acquire(1)
            await window.acquire(1)
        return recorder

    recorder = asyncio.run(scenario())
    waits = [row for row in rows(recorder)
             if row["name"] == "admission.wait" and row["event"] == "span_end"]
    assert len(waits) == 1
    assert waits[0]["attributes"]["wait_seconds"] == 65.0
    assert waits[0]["attributes"]["wait_reason"] == "capacity"


def test_observed_deployment_capacities_override_legacy_defaults_and_isolate_pools():
    async def scenario():
        gate = ModelAdmission(deployment_capacities={
            "evaluator-replica": (60_000, 60), "writer-replica": (100_000, 100),
            "decision-replica": (100_000, 120), "decision-rag-evaluator": (30_000, 30),
        })
        evaluator = await gate.acquire(
            "/openai/deployments/evaluator-replica/chat/completions", {}
        )
        original = await gate.acquire(
            "/openai/deployments/decision-rag-evaluator/chat/completions", {}
        )
        writer = await gate.acquire(
            "/openai/deployments/writer-replica/chat/completions", {}
        )
        decision = await gate.acquire(
            "/providers/microsoft/v1/systemone", {"model": "decision-replica"}
        )
        assert (evaluator.tokens, evaluator.requests) == (48_000, 60)
        assert (original.tokens, original.requests) == (24_000, 30)
        assert (writer.tokens, writer.requests) == (80_000, 100)
        assert (decision.tokens, decision.requests) == (80_000, 120)
        assert len({id(window) for window in (evaluator, original, writer, decision)}) == 4
        assert set(gate.windows) == {
            "evaluator-replica", "decision-rag-evaluator", "writer-replica", "decision-replica"
        }

    asyncio.run(scenario())


@pytest.mark.parametrize("capacity", [(0, 100), (100, 0), (True, 1), (10.0, 1), (1,), None])
def test_invalid_observed_capacities_are_rejected(capacity):
    with pytest.raises(ValueError):
        ModelAdmission(deployment_capacities={"fixture": capacity})


def test_azure_session_accepts_explicit_observed_capacities():
    session = AzureSession(
        SimpleNamespace(), SimpleNamespace(), SimpleNamespace(),
        deployment_capacities={"fixture-evaluator": (60_000, 60)},
    )
    assert session.admission.deployment_capacities == {"fixture-evaluator": (60_000, 60)}


def test_search_calls_hash_inputs_and_capture_result_count(tmp_path):
    session = FixtureSession(httpx.Response(200, json={"value": [{"id": "chunk-1"}]}))

    async def scenario():
        with observation(tmp_path, {}) as recorder:
            assert await Search(session).query("secret question", [0.1, 0.2]) == [{"id": "chunk-1"}]
        return recorder

    recorder = asyncio.run(scenario())
    end = next(row for row in rows(recorder)
               if row["name"] == "search.query" and row["event"] == "span_end")
    assert end["attributes"]["result_count"] == 1
    assert len(end["attributes"]["question_hash"]) == 64
    assert "secret question" not in recorder.path.read_text()


@pytest.mark.parametrize("failure", [None, "auth", "http", "transport"])
def test_azure_auth_transport_ids_and_sanitized_errors(tmp_path, failure):
    resources = ExperimentResources.model_validate_json(
        (Path(__file__).parents[1] / "manifests/azure-resources.json").read_bytes()
    )

    class Credential:
        async def get_token(self, *scopes):
            if failure == "auth":
                raise ClientAuthenticationError(message="secret credential detail")
            return SimpleNamespace(token="secret-token")

    def handler(request):
        assert len(request.headers["x-ms-client-request-id"]) == 32
        if failure == "transport":
            raise httpx.ConnectError("secret URL", request=request)
        return httpx.Response(
            429 if failure == "http" else 200, json={"secret": "provider body"},
            headers={"x-ms-request-id": "allowed-id", "set-cookie": "secret-cookie"},
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with observation(tmp_path / "traces", {}) as recorder:
                session = AzureSession(resources, Credential(), client)
                if failure:
                    with pytest.raises(ServiceError) as error:
                        await session.request("search", "POST", "/test", json={"query": "secret"})
                    assert "secret" not in str(error.value)
                    assert error.value.__cause__ is None
                else:
                    await session.request("search", "POST", "/test", json={"query": "secret"})
        return recorder

    recorder = asyncio.run(scenario())
    assert "secret" not in recorder.path.read_text()
    names = {row["name"] for row in rows(recorder)}
    assert {"azure.request", "azure.admission", "azure.authentication"} <= names
    if failure != "auth":
        assert "azure.transport" in names
    if failure in (None, "http"):
        assert "allowed-id" in recorder.path.read_text()
