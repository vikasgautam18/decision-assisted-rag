import pytest
from fastapi.testclient import TestClient

from app.main import create_app
from clients.synthetic import SyntheticWriter
from pipelines.budget import BudgetLimits, RequestBudget, ServiceError
from pipelines.control import ControlPipeline

QUESTION = "What is the declared combined range of the DEMO-1 Electric variant?"


def test_health_and_synthetic_lookup():
    with TestClient(create_app(), headers={"origin": "http://testserver"}) as client:
        assert client.get("/health").json()["live_services"] is False
        response = client.post("/api/lookup", json={"question": QUESTION})
        assert response.status_code == 200
        assert response.json()["behavior"] == "answer"
        assert response.json()["semantic_verification"] == "not_run"
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize(
    "body",
    [
        {"question": " "},
        {"question": 123},
        {"question": "a" * 2001},
        {"question": QUESTION, "endpoint": "https://arbitrary.invalid"},
        {"question": QUESTION, "conversation_id": "foreign"},
        {"question": QUESTION, "arm": "F"},
    ],
)
def test_reject_invalid_request_or_unimplemented_controls(body):
    with TestClient(create_app(), headers={"origin": "http://testserver"}) as client:
        assert client.post("/api/lookup", json=body).status_code == 422


def test_cross_origin_rejected_and_same_origin_allowed():
    with TestClient(create_app()) as client:
        response = client.post(
            "/api/lookup",
            json={"question": QUESTION},
            headers={"origin": "https://foreign.invalid"},
        )
        assert response.status_code == 403
        response = client.post(
            "/api/lookup", json={"question": QUESTION}, headers={"origin": "http://testserver"}
        )
        assert response.status_code == 200
        assert (
            client.post(
                "/api/lookup",
                json={"question": QUESTION},
                headers={"sec-fetch-site": "cross-site"},
            ).status_code
            == 403
        )


def test_budget_failure_has_error_http_status():
    with TestClient(
        create_app(budget_factory=lambda: RequestBudget(BudgetLimits(max_calls=1))),
        headers={"origin": "http://testserver"},
    ) as client:
        response = client.post("/api/lookup", json={"question": QUESTION})
        assert response.status_code == 503
        assert response.json()["error_code"] == "budget_exhausted"
        assert response.json()["behavior"] is None


def test_service_failure_has_error_http_status_and_no_private_details():
    class FailingSearch:
        async def search(self, plan):
            raise ServiceError("private detail")

    with TestClient(
        create_app(ControlPipeline(FailingSearch(), SyntheticWriter())),
        headers={"origin": "http://testserver"},
    ) as client:
        response = client.post("/api/lookup", json={"question": QUESTION})
        assert response.status_code == 502
        assert response.json()["error_code"] == "service_error"
        assert "private detail" not in response.text


def test_mutation_requires_explicit_same_origin_signal():
    with TestClient(create_app()) as client:
        assert client.post("/api/lookup", json={"question": QUESTION}).status_code == 403
        response = client.post(
            "/api/lookup", json={"question": QUESTION}, headers={"sec-fetch-site": "same-origin"}
        )
        assert response.status_code == 200
