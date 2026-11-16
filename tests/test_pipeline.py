import asyncio

import pytest

from clients.synthetic import SYNTHETIC_SOURCE, SyntheticSearch, SyntheticWriter
from pipelines.budget import BudgetLimits, RequestBudget, ServiceError
from pipelines.contracts import AnswerArtifact, EvidenceBundle, MaterialClaim, QueryContext
from pipelines.control import ControlPipeline

QUESTION = "What is the declared combined range of the DEMO-1 Electric variant?"


def run(search=None, writer=None, limits=None, question=QUESTION, **context):
    budget = RequestBudget(limits or BudgetLimits())
    pipeline = ControlPipeline(
        search if search is not None else SyntheticSearch(),
        writer if writer is not None else SyntheticWriter(),
    )
    result = asyncio.run(
        pipeline.run(
            QueryContext(original_question=question, retrieval_question=question, **context), budget
        )
    )
    return result, budget


def test_grounded_synthetic_lookup():
    result, budget = run()
    assert result.status == "completed" and result.behavior == "answer"
    assert result.answer == SYNTHETIC_SOURCE.text
    assert result.citations[0].source_id == SYNTHETIC_SOURCE.source_id
    assert result.citations[0].revision == "A" and result.citations[0].page_start == 1
    assert result.semantic_verification == "not_run"
    assert result.execution_mode == "synthetic"
    assert [operation.stage for operation in budget.operations] == ["search", "writer"]
    assert all(
        operation.usage_status == "synthetic_not_billable" for operation in budget.operations
    )


def test_no_evidence_is_not_an_operational_failure():
    result, budget = run(question="What is the price?")
    assert result.status == "completed" and result.behavior == "abstain"
    assert not result.citations and result.error_code is None
    assert budget.calls == 1


def test_unresolved_reference_clarifies_without_calls():
    result, budget = run(unresolved_references=("which variant",))
    assert result.behavior == "clarify" and budget.calls == 0


class FailingSearch:
    async def search(self, plan):
        raise ServiceError("private endpoint and credential detail must never reach users")


def test_service_error_not_converted_to_abstention(caplog):
    result, budget = run(search=FailingSearch())
    assert result.status == "error" and result.error_code == "service_error"
    assert result.behavior is None and result.answer is None
    assert "private endpoint" not in result.model_dump_json() + caplog.text
    assert result.trace_id in caplog.text
    assert budget.operations[0].status == "failed"


class MalformedWriter:
    async def write(self, plan, evidence):
        return {"draft": "missing claims"}


class InventingWriter:
    async def write(self, plan, evidence):
        return AnswerArtifact(
            draft="Unsupported claim",
            claims=(
                MaterialClaim(claim_id="c1", text="Unsupported claim", source_ids=("foreign",)),
            ),
        )


@pytest.mark.parametrize("writer", [MalformedWriter(), InventingWriter()])
def test_invalid_output_is_not_delivered(writer):
    result, _ = run(writer=writer)
    assert result.error_code == "invalid_output"
    assert result.answer is None and not result.citations


def test_call_limit_prevents_writer():
    result, budget = run(limits=BudgetLimits(max_calls=1))
    assert result.error_code == "budget_exhausted"
    assert budget.calls == 1


class IncompleteSearch:
    async def search(self, plan):
        return EvidenceBundle(
            sources=(SYNTHETIC_SOURCE,),
            missing_slots=("qualification",),
            selection_provenance="fixture",
        )


def test_partial_evidence_does_not_produce_an_unqualified_answer():
    result, budget = run(search=IncompleteSearch())
    assert result.behavior == "abstain" and budget.calls == 1


class ConflictingSearch:
    async def search(self, plan):
        return EvidenceBundle(
            sources=(SYNTHETIC_SOURCE,),
            conflicts=("unresolved discrepancy",),
            selection_provenance="fixture",
        )


def test_conflict_not_silently_resolved():
    result, budget = run(search=ConflictingSearch())
    assert result.behavior == "conflict" and budget.calls == 1


def test_deadline_cancels_inflight_call():
    cancelled = []

    class SlowSearch:
        async def search(self, plan):
            try:
                await asyncio.sleep(10)
            finally:
                cancelled.append(True)

    result, budget = run(search=SlowSearch(), limits=BudgetLimits(timeout_seconds=0.01))
    assert result.error_code == "budget_exhausted"
    assert cancelled == [True] and budget.calls == 1


def test_external_cancellation_propagates():
    async def scenario():
        started = asyncio.Event()

        async def operation():
            started.set()
            await asyncio.sleep(10)

        budget = RequestBudget(BudgetLimits())
        task = asyncio.create_task(budget.call("search", operation))
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert budget.operations[0].status == "cancelled"

    asyncio.run(scenario())


def test_elapsed_deadline_prevents_call():
    clock = [0.0]
    budget = RequestBudget(BudgetLimits(), clock=lambda: clock[0])
    clock[0] = 6.0
    result = asyncio.run(
        ControlPipeline(SyntheticSearch(), SyntheticWriter()).run(
            QueryContext(original_question=QUESTION, retrieval_question=QUESTION), budget
        )
    )
    assert result.error_code == "budget_exhausted" and budget.calls == 0


def test_unexpected_programming_errors_are_not_silently_hidden():
    class BrokenSearch:
        async def search(self, plan):
            raise RuntimeError("programming bug")

    with pytest.raises(RuntimeError, match="programming bug"):
        run(search=BrokenSearch())
