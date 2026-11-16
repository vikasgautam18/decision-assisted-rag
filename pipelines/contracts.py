from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=200)]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
Behavior = Literal["answer", "clarify", "conflict", "abstain"]


class Artifact(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True, allow_inf_nan=False)

    schema_version: Literal["1"] = "1"


class QueryContext(Artifact):
    original_question: Text
    retrieval_question: Text
    platforms: tuple[Identifier, ...] = ()
    unresolved_references: tuple[Text, ...] = ()
    conversation_id: Identifier | None = None


class RetrievalPlan(Artifact):
    strategy: Literal["lookup", "comparison", "history", "procedure", "calculation"] = "lookup"
    question: Text
    platforms: tuple[Identifier, ...] = ()
    temporal_scope: Literal["current", "historical"] = "current"
    required_slots: tuple[Identifier, ...] = ("requested_fact",)
    primary_section: Literal["overview"] | None = None


class EvidenceSource(Artifact):
    source_id: Identifier
    document: Identifier
    revision: Identifier
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)
    text: Text
    slots: tuple[Identifier, ...]
    context_source_ids: tuple[Identifier, ...] = ()
    historical: bool = False

    @model_validator(mode="after")
    def ordered_pages(self) -> Self:
        if self.page_end < self.page_start:
            raise ValueError("page_end must not precede page_start")
        return self


class EvidenceBundle(Artifact):
    sources: tuple[EvidenceSource, ...] = ()
    missing_slots: tuple[Identifier, ...] = ()
    conflicts: tuple[Text, ...] = ()
    selection_provenance: Text

    @model_validator(mode="after")
    def valid_source_graph(self) -> Self:
        ids = {source.source_id for source in self.sources}
        if len(ids) != len(self.sources):
            raise ValueError("source IDs must be unique")
        for source in self.sources:
            if source.source_id in source.context_source_ids:
                raise ValueError("source must not depend on itself")
            if not set(source.context_source_ids) <= ids:
                raise ValueError("supporting context must be included in the bundle")
        return self


class MaterialClaim(Artifact):
    claim_id: Identifier
    text: Text
    source_ids: tuple[Identifier, ...] = Field(min_length=1)


class AnswerArtifact(Artifact):
    draft: Text
    claims: tuple[MaterialClaim, ...] = Field(min_length=1)
    figure_ids: tuple[Identifier, ...] = ()

    @model_validator(mode="after")
    def unique_claims(self) -> Self:
        if len({claim.claim_id for claim in self.claims}) != len(self.claims):
            raise ValueError("claim IDs must be unique")
        return self


class VerificationCheck(Artifact):
    check_id: Identifier
    outcome: Literal["supported", "contradicted", "insufficient_evidence", "invalid"]
    claim_ids: tuple[Identifier, ...] = ()
    source_ids: tuple[Identifier, ...] = ()


class VerificationReport(Artifact):
    checks: tuple[VerificationCheck, ...]
    semantic_status: Literal["not_run", "passed", "failed", "uncertain"]
    whole_draft_status: Literal["not_run", "supported", "contradicted", "insufficient_evidence"]


class Citation(Artifact):
    source_id: Identifier
    document: Identifier
    revision: Identifier
    page_start: int = Field(ge=1)
    page_end: int = Field(ge=1)


class PipelineResult(Artifact):
    trace_id: Identifier
    execution_mode: Literal["synthetic"] = "synthetic"
    status: Literal["completed", "error"]
    behavior: Behavior | None = None
    answer: Text | None = None
    citations: tuple[Citation, ...] = ()
    semantic_verification: Literal["not_run", "passed"] = "not_run"
    error_code: (
        Literal["service_error", "invalid_output", "invalid_evidence", "budget_exhausted"] | None
    ) = None

    @model_validator(mode="after")
    def consistent_terminal_state(self) -> Self:
        if self.status == "error":
            if (
                self.error_code is None
                or self.behavior is not None
                or self.answer is not None
                or self.citations
                or self.semantic_verification != "not_run"
            ):
                raise ValueError("operational errors must not contain a semantic answer or pass")
        elif self.error_code is not None or self.behavior is None or self.answer is None:
            raise ValueError("completed results require behavior and answer, without an error")
        if self.behavior == "answer" and not self.citations:
            raise ValueError("answers require citations")
        return self
