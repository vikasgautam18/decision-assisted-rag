import pytest
from pydantic import ValidationError

from clients.synthetic import SYNTHETIC_SOURCE
from pipelines.budget import BudgetLimits
from pipelines.contracts import EvidenceBundle, EvidenceSource, PipelineResult, QueryContext


def test_contract_json_roundtrip():
    original = EvidenceBundle(sources=(SYNTHETIC_SOURCE,), selection_provenance="fixture")
    assert EvidenceBundle.model_validate_json(original.model_dump_json()) == original


@pytest.mark.parametrize(
    "changes",
    [
        {"page_start": 0},
        {"page_end": 0},
        {"page_start": 2, "page_end": 1},
        {"page_start": "1"},
        {"unexpected": True},
        {"text": " "},
    ],
)
def test_invalid_sources(changes):
    with pytest.raises(ValidationError):
        EvidenceSource.model_validate({**SYNTHETIC_SOURCE.model_dump(), **changes})


def test_duplicate_and_missing_context_sources():
    with pytest.raises(ValidationError, match="unique"):
        EvidenceBundle(sources=(SYNTHETIC_SOURCE, SYNTHETIC_SOURCE), selection_provenance="fixture")
    source = EvidenceSource.model_validate(
        {**SYNTHETIC_SOURCE.model_dump(), "context_source_ids": ("missing",)}
    )
    with pytest.raises(ValidationError, match="supporting context"):
        EvidenceBundle(sources=(source,), selection_provenance="fixture")


@pytest.mark.parametrize(
    "fields",
    [
        {"status": "completed", "behavior": "answer", "answer": "uncited"},
        {"status": "completed", "behavior": "abstain"},
        {"status": "error", "error_code": "service_error", "behavior": "abstain"},
        {"status": "error", "error_code": "service_error", "semantic_verification": "passed"},
    ],
)
def test_invalid_terminal_states(fields):
    with pytest.raises(ValidationError):
        PipelineResult(trace_id="test", **fields)


@pytest.mark.parametrize(
    "fields",
    [
        {"timeout_seconds": 0.0},
        {"timeout_seconds": float("inf")},
        {"timeout_seconds": float("nan")},
        {"max_calls": 0},
        {"max_calls": True},
    ],
)
def test_invalid_budgets(fields):
    with pytest.raises(ValidationError):
        BudgetLimits(**fields)


def test_question_rejects_blank_and_unknown_fields():
    with pytest.raises(ValidationError):
        QueryContext(original_question=" ", retrieval_question="valid")
    with pytest.raises(ValidationError):
        QueryContext(original_question="valid", retrieval_question="valid", endpoint="arbitrary")
