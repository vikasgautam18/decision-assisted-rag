from __future__ import annotations

import json
from typing import Annotated, Literal, Self
from urllib.parse import urlsplit

from pydantic import Field, StringConstraints, field_validator, model_validator

from pipelines.contracts import Artifact

ExperimentName = Annotated[
    str, StringConstraints(pattern=r"^decision-rag-[a-z0-9]+(?:-[a-z0-9]+)*$", max_length=63)
]


class ExperimentResources(Artifact):
    subscription_id: str = Field(
        pattern=r"^[a-f0-9]{8}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{4}-[a-f0-9]{12}$"
    )
    resource_group: str = Field(min_length=1)
    search_endpoint: str
    di_endpoint: str
    ai_endpoint: str
    storage_account: str = Field(pattern=r"^[a-z0-9]{3,24}$")
    index_name: ExperimentName = "decision-rag-chunks-v1"
    documents_container: ExperimentName = "decision-rag-docs"
    layouts_container: ExperimentName = "decision-rag-layout"
    figures_container: ExperimentName = "decision-rag-figures"
    traces_container: ExperimentName = "decision-rag-traces"
    writer_deployment: str = Field(min_length=1)
    embedding_deployment: str = Field(min_length=1)
    embedding_dimensions: int = Field(ge=1)
    decision_deployment: str = Field(min_length=1)

    @field_validator("search_endpoint", "di_endpoint", "ai_endpoint")
    @classmethod
    def service_root(cls, value: str) -> str:
        parsed = urlsplit(value)
        if (
            parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
            or parsed.path not in ("", "/")
            or parsed.port is not None
        ):
            raise ValueError("service endpoint must be an HTTPS root without credentials or query")
        return value.rstrip("/")

    @model_validator(mode="after")
    def distinct_containers(self) -> Self:
        containers = (
            self.documents_container,
            self.layouts_container,
            self.figures_container,
            self.traces_container,
        )
        if len(set(containers)) != len(containers):
            raise ValueError("documents, layouts, figures, and traces require separate containers")
        return self

    def require_write_target(self, kind: Literal["index", "container"], name: str) -> None:
        permitted = (
            {self.index_name}
            if kind == "index"
            else {
                self.documents_container,
                self.layouts_container,
                self.figures_container,
                self.traces_container,
            }
        )
        if kind not in ("index", "container") or name not in permitted:
            raise ValueError("write target is outside the configured experiment resources")


def main() -> None:
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Validate a non-secret resource plan; no Azure writes."
    )
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    resources = ExperimentResources.model_validate_json(args.config.read_bytes())
    print(json.dumps({"status": "plan_only", "resources": resources.model_dump()}, indent=2))


if __name__ == "__main__":
    main()
