import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from app.resources import ExperimentResources
from clients.azure import AzureSession
from clients.search import PROJECT_MARKER, SEARCH_API_VERSION, Search, index_definition
from clients.storage import Storage
from pipelines.budget import ServiceError


class FakeCredential:
    async def get_token(self, *scopes):
        return SimpleNamespace(token="not-a-real-token")


@pytest.fixture
def resources():
    return ExperimentResources.model_validate_json(
        (Path(__file__).parents[1] / "manifests/azure-resources.json").read_bytes()
    )


def test_search_creation_is_isolated_and_conditional(resources):
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(404)
        assert request.headers["if-none-match"] == "*"
        assert request.url.params["api-version"] == "2025-09-01"
        return httpx.Response(201, json={"name": resources.index_name})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await Search(AzureSession(resources, FakeCredential(), client)).prepare()

    asyncio.run(scenario())
    assert len(seen) == 2
    assert all("/indexes/decision-rag-chunks-v1" in str(request.url) for request in seen)
    assert SEARCH_API_VERSION == "2025-09-01"
    assert index_definition(resources.index_name, 3072)["description"] == PROJECT_MARKER


def test_existing_unowned_index_is_never_overwritten(resources):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, json={"description": "another experiment"})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ServiceError, match="ownership"):
                await Search(AzureSession(resources, FakeCredential(), client)).prepare()

    asyncio.run(scenario())
    assert len(seen) == 1 and seen[0].method == "GET"


def test_storage_creation_is_private_and_annotated(resources):
    seen = []

    def handler(request):
        seen.append(request)
        if request.method == "GET":
            return httpx.Response(404)
        assert request.headers["x-ms-meta-project"] == "decision-assisted-rag"
        assert "x-ms-blob-public-access" not in request.headers
        assert request.headers["if-none-match"] == "*"
        return httpx.Response(201)

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            await Storage(AzureSession(resources, FakeCredential(), client)).prepare(
                resources.documents_container
            )

    asyncio.run(scenario())
    assert len(seen) == 2


def test_immutable_storage_artifacts_refuse_overwrite(resources):
    seen = []

    def handler(request):
        seen.append(request)
        return httpx.Response(200, content=b"existing different bytes")

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            with pytest.raises(ServiceError, match="different bytes"):
                await Storage(AzureSession(resources, FakeCredential(), client)).upload(
                    resources.documents_container,
                    "NOVA-3/hash/original.pdf",
                    b"new bytes",
                    "application/pdf",
                )

    asyncio.run(scenario())
    assert len(seen) == 1


def test_baseline_container_name_never_reaches_network(resources):
    async def scenario():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: pytest.fail("unexpected network call"))
        ) as client:
            with pytest.raises(ValueError, match="outside"):
                await Storage(AzureSession(resources, FakeCredential(), client)).prepare("docs")

    asyncio.run(scenario())


def test_history_reserves_historical_evidence_and_current_comparison_context(resources):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        historical = body["filter"] == "historical eq true"
        return httpx.Response(
            200, json={"value": [{"id": "previous" if historical else "current"}]}
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            search = Search(AzureSession(resources, FakeCredential(), client))
            results = await search.query(
                "What changed from the previous revision?",
                [0.0] * resources.embedding_dimensions,
                historical=True,
            )
            assert [item["id"] for item in results] == ["previous", "current"]
            bodies.clear()
            await search.query("Current value?", [0.0] * resources.embedding_dimensions)
            assert len(bodies) == 1 and bodies[0]["filter"] == "historical eq false"

    asyncio.run(scenario())


def test_figure_request_reserves_original_figures_separately_from_prose(resources):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        figure = "kind eq 'figure'" in body["filter"]
        return httpx.Response(
            200, json={"value": [{"id": "original" if figure else "table-context"}]}
        )

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            results = await Search(AzureSession(resources, FakeCredential(), client)).query(
                "Show the original warranty flowchart",
                [0.0] * resources.embedding_dimensions,
                include_figures=True,
            )
            assert [item["id"] for item in results] == ["original", "table-context"]

    asyncio.run(scenario())
    assert [(body["filter"], body["top"]) for body in bodies] == [
        ("historical eq false and kind eq 'figure'", 2),
        ("historical eq false and kind ne 'figure'", 3),
    ]


def test_historical_figure_request_preserves_both_requirements(resources):
    bodies = []

    def handler(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json={"value": [{"id": body["filter"]}]})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            results = await Search(AzureSession(resources, FakeCredential(), client)).query(
                "Show the superseded diagram and current replacement",
                [0.0] * resources.embedding_dimensions,
                historical=True,
                include_figures=True,
            )
            assert len(results) == 4

    asyncio.run(scenario())
    assert [(body["filter"], body["top"]) for body in bodies] == [
        ("historical eq true and kind eq 'figure'", 1),
        ("historical eq true and kind ne 'figure'", 2),
        ("historical eq false and kind eq 'figure'", 1),
        ("historical eq false and kind ne 'figure'", 1),
    ]
