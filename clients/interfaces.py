from typing import Protocol

from pipelines.contracts import AnswerArtifact, EvidenceBundle, RetrievalPlan


class SearchClient(Protocol):
    async def search(self, plan: RetrievalPlan) -> EvidenceBundle: ...


class WriterClient(Protocol):
    async def write(self, plan: RetrievalPlan, evidence: EvidenceBundle) -> AnswerArtifact: ...
