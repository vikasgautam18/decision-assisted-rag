import json

import pytest

from clients.telemetry import observation, span
from evaluation.observed_spans import analyze_trace, collect, union_seconds


def test_child_overlap_is_unioned_not_double_subtracted():
    assert union_seconds([(1, 3), (2, 4), (5, 6)]) == 4
    with pytest.raises(ValueError):
        union_seconds([(3, 1)])


def test_observed_tree_and_usage_extraction_are_exactly_correlated(tmp_path):
    with observation(tmp_path, {"observation_id": "a" * 32, "arm": "F"}) as recorder:
        with span("pipeline"):
            with span(
                "model.attempt",
                call_id="known-call",
                role="writer",
                usage={"input_tokens": 10, "output_tokens": 1},
            ):
                pass
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    result = analyze_trace(events)
    assert result["status"] == "complete"
    assert result["observation_id"] == "a" * 32
    assert result["calls"][0]["call_id"] == "known-call"
    assert result["calls"][0]["usage"]["input_tokens"] == 10
    assert all(row["exclusive_seconds"] >= 0 for row in result["spans"])
    assert collect([tmp_path])["trace_count"] == 1
    with pytest.raises(ValueError, match="twice"):
        collect([tmp_path, tmp_path])


def test_incomplete_trace_remains_visible_not_successful(tmp_path):
    with observation(tmp_path, {"observation_id": "a" * 32}) as recorder:
        pass
    events = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    result = analyze_trace(events[:1])
    assert result["status"] == "incomplete"
    assert len(result["incomplete_span_ids"]) == 1
    with pytest.raises(ValueError, match="without"):
        analyze_trace(events[1:])
