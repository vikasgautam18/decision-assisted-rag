import asyncio
import json

import pytest

from clients.decision import ChoiceResult
from pipelines.live import (
    APPLICABILITY_RULE,
    CLAIM_SUPPORT,
    SUPPORT,
    LivePipeline,
    requests_visual_evidence,
)

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
    def __init__(self, *, reject=False, invented=False, intent="lookup", irrelevant=False):
        self.reject = reject
        self.invented = invented
        self.intent = intent
        self.irrelevant = irrelevant
        self.calls = []
        self.writer_calls = 0
        self.writer_systems = []
        self.writer_payloads = []
        self.decision_requests = []

    async def embed(self, texts):
        self.calls.append("embed")
        return [[0.1, 0.2, 0.3]]

    async def complete(self, system, content, *, structured=False):
        self.writer_calls += 1
        self.writer_systems.append(system)
        self.writer_payloads.append(json.loads(content))
        self.calls.append("writer")
        return json.dumps(
            {
                "behavior": "answer",
                "draft": "Range 310 km [NOVA-3-table-6]",
                "claims": [
                    {
                        "claim_id": "c1",
                        "text": "Range 310 km",
                        "source_ids": ["foreign" if self.invented else "NOVA-3-table-6"],
                    }
                ],
                "figure_ids": [],
            }
        )

    async def decide(self, state, questions):
        self.decision_requests.append((state, questions))
        self.calls.append(tuple(questions))
        if "intent" in questions:
            return {
                "intent": ChoiceResult(self.intent, None),
                "figure_requirement": ChoiceResult(
                    "required" if self.intent == "figure" else "not_required", None
                ),
                "temporal_scope": ChoiceResult(
                    "history" if self.intent == "history" else "current", None
                ),
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
            return {
                key: ChoiceResult("irrelevant" if self.irrelevant else "uncertain", None)
                for key in questions
            }
        assert questions["whole_draft"][1] is SUPPORT
        assert questions["c1"][1] is CLAIM_SUPPORT
        return {
            key: ChoiceResult("insufficient" if self.reject else "supported", None)
            for key in questions
        }


class FakeSearch:
    async def query(
        self, question, vector, *, historical=False, include_figures=False, platforms=()
    ):
        return [SOURCE.copy()]


def execute(tmp_path, models):
    return asyncio.run(
        LivePipeline(models, FakeSearch(), tmp_path).run(
            "What is the NOVA-3 Dedicated CNG range?", []
        )
    )


def test_full_live_flow_can_be_exercised_offline(tmp_path):
    models = FakeModels()
    result = execute(tmp_path, models)
    assert result["status"] == "completed" and result["behavior"] == "answer"
    assert result["semantic_verification"] == "passed"
    assert result["citations"][0]["source_id"] == SOURCE["id"]
    assert "scope_ATLAS-6" in models.calls[0]
    assert models.calls[1:] == [
        "embed",
        ("evidence_0",),
        ("coverage", "expand_NOVA-3"),
        "writer",
        ("whole_draft", "applicability", "c1"),
    ]
    assert (tmp_path / f"{result['trace_id']}.json").exists()


def test_candidate_profile_uses_cache_first_prefix_and_one_repair(tmp_path):
    models = FakeModels(reject=True)
    result = asyncio.run(
        LivePipeline(
            models,
            FakeSearch(),
            tmp_path,
            pilot_profile="candidate",
        ).run("What is the NOVA-3 Dedicated CNG range?", [])
    )
    assert result["status"] == "completed"
    assert result["behavior"] == "abstain"
    assert result["semantic_verification"] == "not_run"
    assert models.writer_calls == 2

    first_payload = models.writer_payloads[0]
    keys = list(first_payload.keys())
    assert keys[:3] == ["required_output", "calculation_rules", "sources"]
    assert keys[-1] == "repair_failed_checks_only"
    assert "current_question" in keys[3:]
    assert "available_platforms" not in first_payload
    assert "document_title_data" not in first_payload


def test_candidate_profile_rejects_unknown_profile(tmp_path):
    with pytest.raises(ValueError, match="Unknown Arm"):
        LivePipeline(
            FakeModels(),
            FakeSearch(),
            tmp_path,
            pilot_profile="unknown",
        )


def test_candidate_profile_rejects_non_arm_f(tmp_path):
    with pytest.raises(ValueError, match="only for Arm F"):
        LivePipeline(
            FakeModels(),
            FakeSearch(),
            tmp_path,
            arm="E",
            pilot_profile="candidate",
        )


def test_whole_draft_and_claim_criteria_are_not_conflated():
    assert "complete current question" in SUPPORT["supported"]
    assert "complete current question" not in CLAIM_SUPPORT["supported"]


def test_repair_is_bounded_and_unverified_draft_never_delivered(tmp_path):
    models = FakeModels(reject=True)
    result = execute(tmp_path, models)
    assert result["behavior"] == "abstain"
    assert result["semantic_verification"] == "not_run"
    assert result["citations"] == []
    assert models.writer_calls == 4
    assert "310" not in result["answer"]


def test_invented_source_stops_before_semantic_verification(tmp_path):
    models = FakeModels(invented=True)
    result = execute(tmp_path, models)
    assert result["status"] == "error" and result["answer"] is None
    assert not any(isinstance(call, tuple) and "whole_draft" in call for call in models.calls)


@pytest.mark.parametrize(
    "answer",
    [
        "For the Plug-in Hybrid only, before the change use 62 Nem, no angle stage. "
        "Which variant is it?",
        "For the explicitly specified Plug-in Hybrid before the change, use 62 Nem, "
        "no angle stage.",
        "For the explicitly specified Plug-in Hybrid from the change date, "
        "use the current bushing's documented schedule.",
        "The stated Turbo Petrol variant omits this component; do not apply the hybrid torque.",
        "The stated Diesel variant omits this component; do not apply the hybrid torque.",
    ],
)
def test_applicability_failure_repairs_even_when_other_checks_pass(tmp_path, answer):
    class ApplicabilityModels(FakeModels):
        async def complete(self, system, content, *, structured=False):
            output = json.loads(await super().complete(system, content, structured=structured))
            assert APPLICABILITY_RULE in system
            assert "use behavior 'answer'" in system
            assert "clarification-only responses do not deliver factual drafts" in system
            if self.writer_calls > 1:
                feedback = json.loads(content)["repair_failed_checks_only"]
                assert feedback["failed_check_ids"] == ["applicability"]
                assert feedback["claim_source_ids"] == {"c1": ["NOVA-3-table-6"]}
                output["draft"] = answer + " [NOVA-3-table-6]"
                output["claims"][0]["text"] = answer
            return json.dumps(output)

        async def decide(self, state, questions):
            result = await super().decide(state, questions)
            if "applicability" in questions:
                assert APPLICABILITY_RULE in questions["applicability"][0]
                result["applicability"] = ChoiceResult(
                    "insufficient" if self.writer_calls == 1 else "supported", None
                )
            return result

    models = ApplicabilityModels()
    result = execute(tmp_path, models)
    assert result["semantic_verification"] == "passed"
    assert result["answer"] == answer + " [NOVA-3-table-6]"
    assert models.writer_calls == 2


def test_unresolved_applicability_never_delivers_otherwise_supported_answer(tmp_path):
    class ApplicabilityModels(FakeModels):
        async def decide(self, state, questions):
            result = await super().decide(state, questions)
            if "applicability" in questions:
                result["applicability"] = ChoiceResult("contradicted", None)
            return result

    models = ApplicabilityModels()
    result = execute(tmp_path, models)
    assert result["behavior"] == "abstain"
    assert result["citations"] == []
    assert models.writer_calls == 4


@pytest.mark.parametrize("malformed", [True, False])
@pytest.mark.parametrize("failed_attempts", [1, 2, 3])
def test_bounded_repairs_can_fix_schema_or_deterministic_source_errors(
    tmp_path, malformed, failed_attempts
):
    class RepairModels(FakeModels):
        async def complete(self, system, content, *, structured=False):
            output = await super().complete(system, content, structured=structured)
            if self.writer_calls <= failed_attempts:
                if malformed:
                    return "not JSON"
                payload = json.loads(output)
                payload["claims"][0]["source_ids"] = ["foreign"]
                return json.dumps(payload)
            feedback = json.loads(content)["repair_failed_checks_only"]
            assert (
                "schema_validation" in feedback
                if malformed
                else "deterministic_validation" in feedback
            )
            return output

    models = RepairModels()
    result = execute(tmp_path, models)
    assert result["status"] == "completed" and result["semantic_verification"] == "passed"
    assert models.writer_calls == failed_attempts + 1


@pytest.mark.parametrize("failed_attempts", [2, 3])
def test_later_semantic_repairs_can_pass(tmp_path, failed_attempts):
    class RepairModels(FakeModels):
        async def decide(self, state, questions):
            self.reject = self.writer_calls <= failed_attempts
            return await super().decide(state, questions)

    models = RepairModels()
    result = execute(tmp_path, models)
    assert result["semantic_verification"] == "passed"
    assert models.writer_calls == failed_attempts + 1


def test_clarification_does_not_search_or_write(tmp_path):
    models = FakeModels(intent="clarify")
    result = execute(tmp_path, models)
    assert result["behavior"] == "clarify"
    assert len(models.calls) == 1 and "intent" in models.calls[0]


def test_clarification_does_not_present_inferred_scope_as_user_specified():
    assert "NOVA-3?" not in LivePipeline.clarification(
        {
            "current_question": "What is the kerb mass of the 2.2 L Diesel AT?",
            "retrieval_plan": {"platforms": ["NOVA-3"]},
        }
    )
    assert "Which vehicle platform" in LivePipeline.clarification(
        {
            "current_question": "What is its mass?",
        }
    )


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("Show the original NOVA-3 figure.", True),
        ("What does Figure 6.1 depict?", True),
        ("Is engine power the engine-only figure or system-combined figure?", False),
        ("What is the declared combined figure for range?", False),
    ],
)
def test_visual_request_detection_distinguishes_numeric_figure_from_image(question, expected):
    assert requests_visual_evidence(question) is expected


def test_writer_prompt_covers_human_adjudicated_source_requirements(tmp_path):
    models = FakeModels()
    execute(tmp_path, models)
    system = models.writer_systems[0]
    assert "compare the named component in each" in system
    assert "report populated distance and time columns separately" in system
    assert "give both numeric ratings" in system
    assert "cite that row rather than infer the value" in system
    assert "non-dispositive discrepancy" in system


def test_historical_lookup_does_not_require_a_current_flowchart_image(tmp_path):
    class OvereagerFigureModels(FakeModels):
        async def decide(self, state, questions):
            decisions = await super().decide(state, questions)
            if "intent" in questions:
                decisions["figure_requirement"] = ChoiceResult("required", None)
            return decisions

    models = OvereagerFigureModels(intent="history")
    result = asyncio.run(
        LivePipeline(models, FakeSearch(), tmp_path).run(
            "Which threshold applied to the claim on 10 January?", []
        )
    )
    assert result["status"] == "completed"
    assert models.writer_payloads[0]["original_figure_required"] is False


def test_historical_procedure_does_not_require_a_current_flowchart_image(tmp_path):
    class HistoricalProcedureModels(FakeModels):
        async def decide(self, state, questions):
            decisions = await super().decide(state, questions)
            if "intent" in questions:
                decisions["figure_requirement"] = ChoiceResult("required", None)
                decisions["temporal_scope"] = ChoiceResult("history", None)
            return decisions

    models = HistoricalProcedureModels(intent="procedure")
    result = asyncio.run(
        LivePipeline(models, FakeSearch(), tmp_path).run(
            "Which step applied in the superseded procedure?", []
        )
    )
    assert result["status"] == "completed"
    assert models.writer_payloads[0]["original_figure_required"] is False


def test_irrelevant_candidates_do_not_reach_writer(tmp_path):
    models = FakeModels(irrelevant=True)
    result = execute(tmp_path, models)
    assert result["behavior"] == "abstain" and models.writer_calls == 0


def test_low_confidence_irrelevance_is_retained_as_uncertain(tmp_path):
    class UncertainModels(FakeModels):
        async def decide(self, state, questions):
            if any(key.startswith("evidence_") for key in questions):
                return {
                    key: ChoiceResult(
                        "irrelevant",
                        {
                            "irrelevant": 0.5,
                            "uncertain": 0.3,
                            "relevant": 0.2,
                        },
                    )
                    for key in questions
                }
            return await super().decide(state, questions)

    result = execute(tmp_path, UncertainModels())
    assert result["semantic_verification"] == "passed"


def test_planned_performance_context_survives_a_full_later_expansion(tmp_path):
    class AnchorModels(FakeModels):
        async def decide(self, state, questions):
            if "coverage" in questions:
                return {
                    "coverage": ChoiceResult("expand", None),
                    "expand_NOVA-3": ChoiceResult("thermal", None),
                }
            result = await super().decide(state, questions)
            if "intent" in questions:
                result["comparison_basis"] = ChoiceResult("declared_specifications", None)
            return result

    thermal = [
        {
            **SOURCE,
            "id": f"thermal-{index}",
            "page_start": 11,
            "page_end": 11,
            "text": "Other thermal context",
        }
        for index in range(24)
    ]
    result = asyncio.run(
        LivePipeline(
            AnchorModels(intent="comparison"),
            FakeSearch(),
            tmp_path,
            [SOURCE] + thermal,
            {"NOVA-3": {"overview": (4,), "thermal": (11,)}},
        ).run("Compare declared NOVA-3 range", [])
    )
    assert result["status"] == "completed" and result["semantic_verification"] == "passed"
    assert SOURCE["id"] in {item["source_id"] for item in result["citations"]}


def test_engine_vs_system_power_question_anchors_declared_specifications(tmp_path):
    class PowerModels(FakeModels):
        async def decide(self, state, questions):
            decisions = await super().decide(state, questions)
            if "comparison_basis" in questions:
                decisions["comparison_basis"] = ChoiceResult("declared_specifications", None)
            return decisions

    result = asyncio.run(
        LivePipeline(
            PowerModels(),
            FakeSearch(),
            tmp_path,
            [SOURCE.copy()],
            {"NOVA-3": {"overview": (4,)}},
        ).run("Is engine power engine-only or system-combined?", [])
    )
    trace = json.loads((tmp_path / f"{result['trace_id']}.json").read_text())
    routing = next(stage for stage in trace["stages"] if stage["stage"] == "routing-v2")
    assert routing["plan"]["primary_section"] == "overview"


def test_programming_bug_is_not_a_quota_error(tmp_path):
    class BrokenModels(FakeModels):
        async def decide(self, state, questions):
            raise RuntimeError("implementation bug")

    with pytest.raises(RuntimeError, match="implementation bug"):
        execute(tmp_path, BrokenModels())


@pytest.mark.parametrize("behavior", ["clarify", "abstain"])
def test_writer_nonanswer_cannot_bypass_verification_with_factual_prose(tmp_path, behavior):
    class NonanswerModels(FakeModels):
        async def complete(self, system, content, *, structured=False):
            return json.dumps(
                {
                    "behavior": behavior,
                    "draft": "Unverified range is 999 km [invented-source]",
                    "claims": [],
                    "figure_ids": [],
                }
            )

    result = execute(tmp_path, NonanswerModels())
    assert result["behavior"] == behavior
    assert "999" not in result["answer"] and "invented-source" not in result["answer"]
    assert result["citations"] == [] and result["figures"] == []


def test_missing_original_figure_stops_before_writer(tmp_path):
    models = FakeModels(intent="figure")
    result = execute(tmp_path, models)
    assert result["behavior"] == "abstain"
    assert models.writer_calls == 0
    assert "original figure" in result["answer"]


@pytest.mark.parametrize("intent", ["figure", "lookup", "history"])
def test_figure_request_reserves_retrieval_and_delivers_verified_original(tmp_path, intent):
    figure = {
        **SOURCE,
        "id": "NOVA-3-figure-2",
        "kind": "figure",
        "figure_id": "NOVA-3-figure-2",
    }

    class FigureSearch:
        async def query(
            self, question, vector, *, historical=False, include_figures=False, platforms=()
        ):
            assert include_figures and historical == (intent == "history")
            return [figure, SOURCE.copy()]

    class FigureModels(FakeModels):
        async def decide(self, state, questions):
            decisions = await super().decide(state, questions)
            if "intent" in questions:
                decisions["figure_requirement"] = ChoiceResult("required", None)
            return decisions

        async def complete(self, system, content, *, structured=False):
            draft = json.loads(await super().complete(system, content, structured=structured))
            draft["figure_ids"] = ["NOVA-3-figure-2"]
            draft["claims"][0]["source_ids"].append("NOVA-3-figure-2")
            return json.dumps(draft)

    models = FigureModels(intent=intent)
    result = asyncio.run(
        LivePipeline(models, FigureSearch(), tmp_path).run("Show the original NOVA-3 figure.", [])
    )
    assert result["semantic_verification"] == "passed"
    assert result["figures"][0]["url"] == "/api/figures/NOVA-3-figure-2"
    figure_check = next(
        questions["selected_figures"]
        for _, questions in models.decision_requests
        if "selected_figures" in questions
    )
    assert "relevance/identity only" in figure_check[0]
    assert "independently proves" in figure_check[0]


def test_missing_fact_expands_correct_section_even_when_platform_is_present(tmp_path):
    revision = {
        **SOURCE,
        "id": "revision-record",
        "page_start": 2,
        "page_end": 2,
        "text": "Revision changes; petrol range is not present",
    }

    class RevisionSearch(FakeSearch):
        async def query(self, *args, **kwargs):
            return [revision.copy()]

    class MissingFactModels(FakeModels):
        async def decide(self, state, questions):
            result = await super().decide(state, questions)
            if "coverage" in questions:
                result["coverage"] = ChoiceResult("expand", None)
                result["expand_NOVA-3"] = ChoiceResult("overview", None)
            return result

    result = asyncio.run(
        LivePipeline(
            MissingFactModels(),
            RevisionSearch(),
            tmp_path,
            [revision, SOURCE.copy()],
            {"NOVA-3": {"overview": (4,)}},
        ).run("NOVA-3 Dedicated CNG range?", [])
    )
    assert result["semantic_verification"] == "passed"
    trace = json.loads((tmp_path / f"{result['trace_id']}.json").read_text())
    expansions = [stage for stage in trace["stages"] if stage["stage"] == "expansion-v1"]
    assert len(expansions) == 1
    assert expansions[0]["added_candidate_ids"] == [SOURCE["id"]]
