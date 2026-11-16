import pytest

from evaluation.experiment import Evaluator
from evaluation.observed_grade import evaluator_assignments


def pool():
    return tuple(
        Evaluator(
            deployment=f"grader-{index}",
            tokens_per_minute=capacity,
            requests_per_minute=100,
        )
        for index, capacity in enumerate((30_000, 60_000, 60_000))
    )


def test_small_requests_preserve_existing_balanced_ordinal_assignment():
    records = [{"observation_id": str(index)} for index in range(6)]
    assigned = evaluator_assignments(
        pool(), [(index, record, 10_000) for index, record in enumerate(records)]
    )
    assert assigned == [[records[0], records[3]], [records[1], records[4]],
                        [records[2], records[5]]]


def test_pilot_large_clarification_payload_uses_only_capable_approved_replicas():
    records = [{"observation_id": str(index)} for index in range(2)]
    assigned = evaluator_assignments(
        pool(), [(index, record, 29_866) for index, record in enumerate(records)]
    )
    assert assigned == [[], [records[0]], [records[1]]]
    assert evaluator_assignments(pool(), [(0, records[0], 24_000)])[0] == [records[0]]
    assert evaluator_assignments(pool(), [(0, records[0], 24_001)])[0] == []


def test_no_fitting_replica_rejects_all_placement_before_dispatch():
    records = [{"observation_id": "small"}, {"observation_id": "too-large"}]
    with pytest.raises(ValueError, match="no approved evaluator fits 48001"):
        evaluator_assignments(
            pool(), [(0, records[0], 1000), (1, records[1], 48_001)]
        )
    assert records == [{"observation_id": "small"}, {"observation_id": "too-large"}]
