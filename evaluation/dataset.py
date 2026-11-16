from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ingestion.corpus import DOSSIERS, REFERENCE_COMMIT, REFERENCE_REPOSITORY

DATA = Path(__file__).parent / "data"
PINNED_HASHES = {
    "benchmark.jsonl": "2ec3f388f724e92e85b4c143b1f2ea29fb251b81042f29bb6a1641863157f035",
    "heldout.jsonl": "eba5343940acedea356781216dba8d28f3a0e49c689983980ab447b784dfa682",
}
CATEGORIES = {
    "variant_confusion",
    "revision_superseded",
    "transposed_table",
    "cross_page_table",
    "conditional_table",
    "figure_diagram",
    "unanswerable",
    "ambiguous",
    "fleet_wide",
    "reasoning_trap",
    "arithmetic",
    "multi_hop",
    "negation",
    "cross_document",
}


class GoldSource(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    doc: str
    page: int = Field(ge=1)

    @model_validator(mode="after")
    def valid_page(self):
        if self.doc not in DOSSIERS or self.page > DOSSIERS[self.doc][1]:
            raise ValueError("gold source is outside the pinned corpus")
        return self


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: str = Field(pattern=r"^q\d{2,}$")
    category: str
    question: str = Field(min_length=1, max_length=2000)
    gold_answer: str
    gold_sources: tuple[GoldSource, ...]
    requires_figure: bool
    expected_behavior: Literal["answer", "flag_conflict", "clarify", "abstain"]
    notes: str = Field(min_length=1)

    @model_validator(mode="after")
    def valid_semantics(self):
        if self.category not in CATEGORIES or not self.question.strip() or not self.notes.strip():
            raise ValueError("invalid category or empty text")
        if self.expected_behavior == "abstain":
            if self.gold_sources or self.category != "unanswerable":
                raise ValueError("abstentions require empty sources and unanswerable category")
        elif not self.gold_sources or not self.gold_answer.strip():
            raise ValueError("other behaviors require gold answer and sources")
        if self.category == "figure_diagram" and not self.requires_figure:
            raise ValueError("figure cases require figure delivery")
        if self.category == "ambiguous" and self.expected_behavior != "clarify":
            raise ValueError("ambiguous cases require clarification")
        return self


def load_cases(directory: Path = DATA) -> list[Case]:
    cases = []
    for filename, count in (("benchmark.jsonl", 57), ("heldout.jsonl", 15)):
        if (
            hashlib.sha256((directory / filename).read_bytes()).hexdigest()
            != (PINNED_HASHES[filename])
        ):
            raise ValueError(f"{filename} differs from the original pinned dataset")
        rows = [
            Case.model_validate_json(line)
            for line in (directory / filename).read_text().splitlines()
            if line.strip()
        ]
        if len(rows) != count:
            raise ValueError(f"{filename} must preserve its original {count} cases")
        cases.extend(rows)
    if len({case.id for case in cases}) != 72:
        raise ValueError("case IDs must be unique across both original splits")
    return cases


def audit(directory: Path = DATA) -> dict:
    cases = load_cases(directory)
    return {
        "reference_repository": REFERENCE_REPOSITORY,
        "reference_commit": REFERENCE_COMMIT,
        "dataset_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("benchmark.jsonl", "heldout.jsonl")
        },
        "cases": len(cases),
        "categories": dict(Counter(case.category for case in cases)),
        "behaviors": dict(Counter(case.expected_behavior for case in cases)),
        "figure_cases": sum(case.requires_figure for case in cases),
        "exposure": "All 72 are existing exposed development/regression cases; heldout is "
        "the historical filename, not newly blind evidence.",
        "gold_source_semantics": "Original lists mix corroborating/alternative pages and "
        "multi-source dependencies; all-page recall is diagnostic, "
        "not an invented mandatory-all-pages gold rule.",
        "coverage_gaps": [
            "Single-turn cases do not establish conversation or scope-switch success.",
            "No dataset labels for malicious retrieved instructions, service faults, budgets, "
            "concurrency or cancellation; existing unit/fault/browser tests cover these.",
            "Gold provides doc/page, not exact figure ID or per-claim citation support.",
            "Runtime score calibration, load and repeated instability require separate protocols.",
        ],
        "new_questions_added": 0,
    }


if __name__ == "__main__":
    print(json.dumps(audit(), indent=2))
