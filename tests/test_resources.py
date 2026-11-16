import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.resources import ExperimentResources


@pytest.fixture
def resources():
    path = Path(__file__).parents[1] / "manifests" / "azure-resources.json"
    return ExperimentResources.model_validate_json(path.read_bytes())


def test_experiment_writes_do_not_target_baseline(resources):
    resources.require_write_target("index", "decision-rag-chunks-v1")
    resources.require_write_target("container", "decision-rag-docs")
    for kind, name in (
        ("index", "dossier-chunks"),
        ("container", "docs"),
        ("container", "figures"),
        ("index", "decision-rag-unconfigured"),
    ):
        with pytest.raises(ValueError, match="outside"):
            resources.require_write_target(kind, name)


@pytest.mark.parametrize("name", ["docs", "dossier-chunks", "decision-rag-a--b", "decision-rag-"])
def test_non_namespaced_or_invalid_resources_rejected(resources, name):
    with pytest.raises(ValidationError):
        ExperimentResources.model_validate({**resources.model_dump(), "index_name": name})


def test_containers_must_be_separate(resources):
    with pytest.raises(ValidationError, match="separate"):
        ExperimentResources.model_validate(
            {**resources.model_dump(), "figures_container": resources.documents_container}
        )


@pytest.mark.parametrize(
    "endpoint",
    [
        "http://example.test",
        "https://user:password@example.test",
        "https://example.test?api-key=secret",
        "https://example.test/indexes/dossier-chunks",
        "https://example.test:8443",
    ],
)
def test_resource_endpoints_are_nonsecret_service_roots(resources, endpoint):
    with pytest.raises(ValidationError):
        ExperimentResources.model_validate({**resources.model_dump(), "search_endpoint": endpoint})


def test_manifest_contains_no_credentials(resources):
    data = json.loads(resources.model_dump_json())
    assert not any("key" in key or "secret" in key or "token" in key for key in data)
