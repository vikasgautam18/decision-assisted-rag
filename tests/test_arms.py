import asyncio
import json

from clients.decision import ChoiceResult
from pipelines.live import LivePipeline
from pipelines.planning import deterministic_plan

SOURCE = {
    "id": "NOVA-3-table-6",
    "document": "NOVA-3",
    "revision": "D",
    "page_start": 4,
    "page_end": 4,
    "kind": "table",
    "historical": False,
    "text": "Dedicated CNG combined range: 310 km",
    "figure_id": "",
}


class FakeModels:
    def __init__(self):
        self.decision_requests = []
        self.semantic_calls = []
        self.writer_calls = 0

    async def embed(self, _texts):
        return [[0.1, 0.2, 0.3]]

    async def complete(self, _system, content, *, structured=False):
        self.writer_calls += 1
        return json.dumps(
            {
                "behavior": "answer",
                "draft": "Range 310 km [NOVA-3-table-6]",
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "Range 310 km",
                        "source_ids": ["NOVA-3-table-6"],
                    }
                ],
                "figure_ids": [],
                "calculations": [],
            }
        )

    async def decide(self, state, questions):
        self.decision_requests.append((state, questions))
        if "intent" in questions:
            return {
                "intent": ChoiceResult("lookup", None),
                "temporal_scope": ChoiceResult("current", None),
                "figure_requirement": ChoiceResult("not_required", None),
                "comparison_basis": ChoiceResult("declared_specifications", None),
                **{
                    key: ChoiceResult("include" if key == "scope_NOVA-3" else "exclude", None)
                    for key in questions
                    if key.startswith("scope_")
                },
            }
        if "coverage" in questions:
            return {
                key: ChoiceResult("complete" if key == "coverage" else "sufficient", None)
                for key in questions
            }
        if any(key.startswith("evidence_") for key in questions):
            return {key: ChoiceResult("relevant", None) for key in questions}
        return {key: ChoiceResult("supported", None) for key in questions}

    async def semantic_complete(self, _system, content):
        request = json.loads(content)
        self.semantic_calls.append(request)
        answers = {}
        for key, question in request["questions"].items():
            labels = question["allowed_labels"]
            if key == "intent":
                choice = "lookup"
            elif key == "temporal_scope":
                choice = "current"
            elif key == "figure_requirement":
                choice = "not_required"
            elif key == "comparison_basis":
                choice = "declared_specifications"
            elif key.startswith("scope_"):
                choice = "include" if key == "scope_NOVA-3" else "exclude"
            elif key == "coverage":
                choice = "complete"
            elif key.startswith("expand_"):
                choice = "sufficient"
            elif key.startswith("evidence_"):
                choice = "relevant"
            else:
                choice = "supported"
            assert choice in labels
            answers[key] = choice
        return json.dumps({"answers": answers})


class FakeSearch:
    async def query(self, *_args, **_kwargs):
        return [SOURCE.copy()]


def pipeline(tmp_path, models, arm):
    catalogue = [SOURCE.copy()]
    sections = {"NOVA-3": {"overview": (4,)}}
    return LivePipeline(
        models, FakeSearch(), tmp_path, catalogue, sections, arm=arm
    )


def run(tmp_path, models, arm):
    return asyncio.run(
        pipeline(tmp_path, models, arm).run("What is the NOVA-3 range?", [])
    )


def test_deterministic_planner_uses_general_scope_history_and_figure_rules():
    current = deterministic_plan("Show the original NOVA-3 figure.", [])
    assert current.require_figure
    assert current.routing["intent"].choice == "figure"
    assert current.routing["scope_NOVA-3"].choice == "include"
    inherited = deterministic_plan("What is its range?", ["What is the NOVA-3 range?"])
    assert inherited.routing["scope_NOVA-3"].choice == "include"
    historical = deterministic_plan("What value applied on 10 January?", ["NOVA-3"])
    assert historical.routing["temporal_scope"].choice == "history"


def test_b_uses_no_decisions_and_marks_verification_not_applicable(tmp_path):
    models = FakeModels()
    result = run(tmp_path, models, "B")
    trace = json.loads((tmp_path / f"{result['trace_id']}.json").read_text())
    assert result["status"] == "completed"
    assert result["semantic_verification"] == "not_applicable"
    assert models.decision_requests == []
    assert trace["stages"][0]["stage"] == "planning-deterministic-v1"
    assert trace["candidate_selection"]["retained_ids"] == [SOURCE["id"]]


def test_c_runs_assessment_without_verification(tmp_path):
    models = FakeModels()
    result = run(tmp_path, models, "C")
    trace = json.loads((tmp_path / f"{result['trace_id']}.json").read_text())
    question_sets = [set(questions) for _, questions in models.decision_requests]
    assert result["semantic_verification"] == "not_applicable"
    assert any("evidence_0" in keys for keys in question_sets)
    assert any("coverage" in keys for keys in question_sets)
    assert not any("whole_draft" in keys for keys in question_sets)
    assert trace["pre_assessment_batches"][0]["sources"] == [SOURCE]


def test_d_runs_verification_without_assessment_or_coverage(tmp_path):
    models = FakeModels()
    result = run(tmp_path, models, "D")
    question_sets = [set(questions) for _, questions in models.decision_requests]
    assert result["semantic_verification"] == "passed"
    assert len(question_sets) == 1
    assert "whole_draft" in question_sets[0]
    assert "applicability" in question_sets[0]
    assert not any("evidence_0" in keys or "coverage" in keys for keys in question_sets)


def test_g_uses_ordinary_categorical_decisions_and_no_decision_client(tmp_path):
    models = FakeModels()
    result = run(tmp_path, models, "G")
    assert result["semantic_verification"] == "passed"
    assert models.decision_requests == []
    assert any("whole_draft" in call["questions"] for call in models.semantic_calls)
    assert all(
        isinstance(result_item, str)
        for call in models.semantic_calls
        for result_item in call["questions"]
    )
