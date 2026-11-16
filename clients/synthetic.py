from pipelines.contracts import (
    AnswerArtifact,
    EvidenceBundle,
    EvidenceSource,
    MaterialClaim,
    RetrievalPlan,
)

SYNTHETIC_SOURCE = EvidenceSource(
    source_id="demo-1-rev-a-p1",
    document="DEMO-1",
    revision="A",
    page_start=1,
    page_end=1,
    text="The DEMO-1 Electric variant has a declared combined range of 120 km.",
    slots=("requested_fact",),
)


class SyntheticSearch:
    async def search(self, plan: RetrievalPlan) -> EvidenceBundle:
        question = plan.question.casefold()
        known_question = (
            "demo-1" in question
            and "electric" in question
            and "range" in question
            and not any(word in question for word in ("diesel", "previous", "compare"))
        )
        return EvidenceBundle(
            sources=(SYNTHETIC_SOURCE,) if known_question else (),
            missing_slots=() if known_question else plan.required_slots,
            selection_provenance="synthetic fixture; not a search engine or real dossier",
        )


class SyntheticWriter:
    async def write(self, plan: RetrievalPlan, evidence: EvidenceBundle) -> AnswerArtifact:
        source = evidence.sources[0]
        return AnswerArtifact(
            draft=source.text,
            claims=(
                MaterialClaim(claim_id="claim-1", text=source.text, source_ids=(source.source_id,)),
            ),
        )
