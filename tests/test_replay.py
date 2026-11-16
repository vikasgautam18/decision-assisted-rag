import copy

import pytest

from evaluation.replay import build_fixture, replay_verification
from pipelines.budget import ServiceError


def fixture():
    return {
        "question": "What is the NOVA-3 range?",
        "sources": [
            {
                "id": "NOVA-3-table-6",
                "document": "NOVA-3",
                "revision": "D",
                "page_start": 4,
                "page_end": 4,
                "kind": "table",
                "figure_id": "",
                "text": "Range 310 km",
            }
        ],
        "initial_draft": {
            "behavior": "answer",
            "draft": "310 km [NOVA-3-table-6]",
            "claims": [{"claim_id": "c1", "text": "310 km", "source_ids": ["NOVA-3-table-6"]}],
            "figure_ids": [],
            "calculations": [],
        },
        "initial_verification": {
            "whole_draft": {
                "choice": "supported",
                "probabilities": {
                    "supported": 0.79,
                    "contradicted": 0.2,
                    "insufficient": 0.01,
                },
            },
            "c1": {"choice": "supported", "probabilities": None},
        },
        "require_figure": False,
        "require_calculation": False,
        "routing_plan": {"platforms": ["NOVA-3"]},
    }


def test_identical_replay_input_changes_only_gate_and_keeps_threshold():
    data = fixture()
    before = copy.deepcopy(data)
    assert replay_verification(data, enabled=False)["action"] == (
        "eligible_without_semantic_verification"
    )
    result = replay_verification(data, enabled=True)
    assert result["action"] == "requires_repair"
    assert result["failed_checks"] == ["whole_draft"]
    assert data == before
    data["initial_verification"]["whole_draft"]["probabilities"] = {
        "supported": 0.8,
        "contradicted": 0.19,
        "insufficient": 0.01,
    }
    assert replay_verification(data, enabled=True)["action"] == (
        "eligible_after_semantic_verification"
    )


@pytest.mark.parametrize("enabled", [False, True])
def test_deterministic_safety_checks_still_apply_to_control(enabled):
    data = fixture()
    data["initial_draft"]["claims"][0]["source_ids"] = ["foreign"]
    assert replay_verification(data, enabled=enabled)["action"] == "deterministic_rejection"


def test_missing_recorded_decisions_are_not_silently_supported():
    data = fixture()
    del data["initial_verification"]["whole_draft"]
    with pytest.raises(ServiceError):
        replay_verification(data, enabled=True)


def test_new_applicability_gate_replays_without_changing_legacy_fixtures():
    data = fixture()
    data["initial_verification"]["whole_draft"]["probabilities"] = None
    assert replay_verification(data, enabled=True)["action"] == (
        "eligible_after_semantic_verification"
    )
    data["initial_verification"]["applicability"] = {
        "choice": "insufficient",
        "probabilities": None,
    }
    result = replay_verification(data, enabled=True)
    assert result["action"] == "requires_repair"
    assert result["failed_checks"] == ["applicability"]


def test_trace_fixture_requires_matched_initial_boundary_not_repaired_draft():
    data = fixture()
    trace = {
        "question": data["question"],
        "evidence": data["sources"],
        "stages": [
            {
                "stage": "routing-v2",
                "decisions": {
                    "figure_requirement": {"choice": "not_required"},
                    "intent": {"choice": "lookup"},
                },
                "plan": {"platforms": ["NOVA-3"], "temporal_scope": "current"},
            },
            {"stage": "draft-v1", "attempt": 0, "artifact": data["initial_draft"]},
            {"stage": "verification-v1", "attempt": 0, "decisions": data["initial_verification"]},
            {"stage": "draft-v1", "attempt": 1, "artifact": {"draft": "do not select"}},
        ],
    }
    result = build_fixture(trace, trace_sha256="a" * 64, origin="fixture")
    assert result["initial_draft"] == data["initial_draft"]
    assert result["source_trace_sha256"] == "a" * 64
    trace["stages"] = [s for s in trace["stages"] if s.get("attempt") != 0]
    with pytest.raises(ValueError, match="boundary"):
        build_fixture(trace, trace_sha256="a" * 64, origin="fixture")
