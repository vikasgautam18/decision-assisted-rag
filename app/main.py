from collections.abc import Callable

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.boundary import install_api_boundary
from clients.synthetic import SyntheticSearch, SyntheticWriter
from pipelines.budget import BudgetLimits, RequestBudget
from pipelines.contracts import PipelineResult, QueryContext
from pipelines.control import ControlPipeline


class LookupRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    question: str = Field(min_length=1, max_length=2000)

    @field_validator("question")
    @classmethod
    def nonblank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("question must not be blank")
        return value.strip()


def create_app(
    pipeline: ControlPipeline | None = None,
    budget_factory: Callable[[], RequestBudget] | None = None,
) -> FastAPI:
    application = FastAPI(title="Decision-assisted RAG: synthetic foundation", version="0.1.0")
    runner = (
        pipeline if pipeline is not None else ControlPipeline(SyntheticSearch(), SyntheticWriter())
    )
    make_budget = budget_factory or (lambda: RequestBudget(BudgetLimits()))

    install_api_boundary(application)

    @application.get("/health")
    async def health():
        return {"status": "ok", "execution_mode": "synthetic", "live_services": False}

    @application.post(
        "/api/lookup",
        response_model=PipelineResult,
        responses={502: {"model": PipelineResult}, 503: {"model": PipelineResult}},
    )
    async def lookup(body: LookupRequest):
        result = await runner.run(
            QueryContext(original_question=body.question, retrieval_question=body.question),
            make_budget(),
        )
        if result.status == "error":
            status = 503 if result.error_code == "budget_exhausted" else 502
            return JSONResponse(status_code=status, content=result.model_dump(mode="json"))
        return result

    return application


app = create_app()
