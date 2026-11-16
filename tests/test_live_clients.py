import asyncio
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.resources import ExperimentResources
from clients.azure import AzureSession
from clients.decision import parse_choices
from clients.models import Models, request_calls
from ingestion.checkpoint import CheckpointQuota
from pipelines.budget import ServiceError

OPTIONS = {"test": {"yes": "Supported", "no": "Not supported"}}


def test_decision_validation_retains_distribution():
    payload = {
        "answers": {
            "test": {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.9, "no": 0.1}}
        }
    }
    result = parse_choices(payload, OPTIONS)["test"]
    assert result.choice == "yes" and result.probabilities == {"yes": 0.9, "no": 0.1}


@pytest.mark.parametrize(
    "answer",
    [
        {"type": "noul", "choice": "yes"},
        {"type": "choice", "choice": "foreign"},
        {"type": "choice", "choice": "yes", "probabilities": {"yes": 1.0}},
        {"type": "choice", "choice": "yes", "probabilities": {"yes": True, "no": False}},
        {"type": "choice", "choice": "yes", "probabilities": {"yes": float("nan"), "no": 0.0}},
        {"type": "choice", "choice": "yes", "probabilities": {"yes": 0.3, "no": 0.1}},
    ],
)
def test_invalid_decision_outputs_rejected(answer):
    with pytest.raises(ServiceError):
        parse_choices({"answers": {"test": answer}}, OPTIONS)


def test_wrong_decision_question_ids_rejected():
    with pytest.raises(ServiceError, match="IDs"):
        parse_choices({"answers": {}}, OPTIONS)


def test_checkpoint_calls_remain_consumed_across_processes_and_failures(tmp_path):
    path = tmp_path / "quota.db"
    first = CheckpointQuota(path)
    for _ in range(first.limits["decision"]):
        first.reserve("decision")
    second = CheckpointQuota(path)
    with pytest.raises(RuntimeError, match="exhausted"):
        second.reserve("decision")
    assert second.snapshot()[0]["state"] == "reserved"


def test_model_quota_is_aggregate_and_completion_recorded(tmp_path):
    quota = CheckpointQuota(tmp_path / "quota.db")
    for _ in range(quota.limits["model"]):
        call_id = quota.reserve("model")
        quota.complete(call_id, {"usage_status": "unknown"})
    with pytest.raises(RuntimeError, match="exhausted"):
        quota.reserve("model")
    assert all(call["state"] == "completed" for call in quota.snapshot())


def test_approved_retest_and_review_caps():
    assert CheckpointQuota.limits == {"di_analysis": 6, "decision": 4800, "model": 1200}


def test_five_aggregate_model_calls_allowed_but_sixth_is_blocked(tmp_path):
    class FakeSession:
        async def request(self, *args, **kwargs):
            return httpx.Response(200, json={"model": "fixture"})

    quota = CheckpointQuota(tmp_path / "quota.db")
    models = Models(FakeSession(), quota)

    async def scenario():
        with request_calls() as calls:
            for _ in range(5):
                await models._invoke("model", "/fixture", {})
            with pytest.raises(RuntimeError, match="per-request call budget exhausted"):
                await models._invoke("model", "/fixture", {})
            assert calls.counts["model"] == 5

    asyncio.run(scenario())
    assert len(quota.snapshot()) == 5


def test_operation_location_cannot_redirect_server_credentials():
    resources = ExperimentResources.model_validate_json(
        (Path(__file__).parents[1] / "manifests/azure-resources.json").read_bytes()
    )

    class FakeCredential:
        async def get_token(self, *scopes):
            return SimpleNamespace(token="not-a-real-token")

    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda r: httpx.Response(200))
        ) as c:
            session = AzureSession(resources, FakeCredential(), c)
            path = "/documentintelligence/documentModels/prebuilt-layout/analyzeResults/123"
            assert session.operation_path("di", resources.di_endpoint + path) == path
            for url in (
                "https://foreign.invalid" + path,
                resources.di_endpoint + "/unrelated",
                resources.di_endpoint + path + "#fragment",
            ):
                with pytest.raises(ServiceError):
                    session.operation_path("di", url)

    asyncio.run(scenario())


def test_transport_does_not_leak_error_body_or_retry(caplog):
    resources = ExperimentResources.model_validate_json(
        (Path(__file__).parents[1] / "manifests/azure-resources.json").read_bytes()
    )
    calls = []

    class FakeCredential:
        async def get_token(self, *scopes):
            return SimpleNamespace(token="not-a-real-token")

    def handler(request):
        calls.append(request)
        return httpx.Response(429, json={"error": "private provider detail"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            session = AzureSession(resources, FakeCredential(), client)
            with pytest.raises(ServiceError, match="HTTP 429") as error:
                await session.request("ai", "POST", "/test", json={})
            assert "private provider detail" not in str(error.value)

    asyncio.run(scenario())
    assert len(calls) == 1
    assert "private provider detail" not in caplog.text
