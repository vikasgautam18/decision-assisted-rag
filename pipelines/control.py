from __future__ import annotations

import logging
import uuid

from pydantic import ValidationError

from clients.interfaces import SearchClient, WriterClient
from pipelines.budget import BudgetExhausted, RequestBudget, ServiceError
from pipelines.contracts import (
    AnswerArtifact,
    Citation,
    EvidenceBundle,
    PipelineResult,
    QueryContext,
    RetrievalPlan,
)

logger = logging.getLogger(__name__)


class InvalidEvidence(RuntimeError):
    pass


class InvalidOutput(RuntimeError):
    pass


class ControlPipeline:
    """Offline lookup slice; semantic decision stages are not implemented yet."""

    def __init__(self, search: SearchClient, writer: WriterClient) -> None:
        self.search = search
        self.writer = writer

    async def run(self, context: QueryContext, budget: RequestBudget) -> PipelineResult:
        trace_id = uuid.uuid4().hex
        try:
            budget.check()
            if context.conversation_id is not None:
                raise InvalidOutput(
                    "conversation resolution is not implemented in the offline slice"
                )
            if context.unresolved_references:
                return PipelineResult(
                    trace_id=trace_id,
                    status="completed",
                    behavior="clarify",
                    answer="Please specify the subject of the unresolved reference.",
                )
            plan = RetrievalPlan(question=context.retrieval_question, platforms=context.platforms)
            raw_evidence = await budget.call("search", lambda: self.search.search(plan))
            evidence = EvidenceBundle.model_validate(raw_evidence)
            if any(source.historical for source in evidence.sources):
                raise InvalidEvidence("current lookup received historical evidence")
            if evidence.conflicts:
                return PipelineResult(
                    trace_id=trace_id,
                    status="completed",
                    behavior="conflict",
                    answer="The evidence contains unresolved conflicts; no answer was generated.",
                )
            supplied = {slot for source in evidence.sources for slot in source.slots}
            if (
                not evidence.sources
                or evidence.missing_slots
                or not set(plan.required_slots) <= supplied
            ):
                return PipelineResult(
                    trace_id=trace_id,
                    status="completed",
                    behavior="abstain",
                    answer="The synthetic fixture does not contain the required evidence.",
                )
            raw_answer = await budget.call("writer", lambda: self.writer.write(plan, evidence))
            answer = AnswerArtifact.model_validate(raw_answer)
            if answer.figure_ids:
                raise InvalidOutput("figure delivery is not implemented in the offline slice")
            sources = {source.source_id: source for source in evidence.sources}
            cited = tuple(dict.fromkeys(sid for claim in answer.claims for sid in claim.source_ids))
            if not set(cited) <= sources.keys():
                raise InvalidOutput("answer cites a source outside the supplied evidence")
            budget.check()
            return PipelineResult(
                trace_id=trace_id,
                status="completed",
                behavior="answer",
                answer=answer.draft,
                citations=tuple(
                    Citation(
                        source_id=sid,
                        document=sources[sid].document,
                        revision=sources[sid].revision,
                        page_start=sources[sid].page_start,
                        page_end=sources[sid].page_end,
                    )
                    for sid in cited
                ),
            )
        except (
            BudgetExhausted,
            ServiceError,
            ValidationError,
            InvalidEvidence,
            InvalidOutput,
        ) as exc:
            if isinstance(exc, BudgetExhausted):
                code = "budget_exhausted"
            elif isinstance(exc, ServiceError):
                code = "service_error"
            elif isinstance(exc, InvalidEvidence):
                code = "invalid_evidence"
            else:
                code = "invalid_output"
            logger.warning("pipeline_failed trace_id=%s error_code=%s", trace_id, code)
            return PipelineResult(trace_id=trace_id, status="error", error_code=code)
