import hashlib
import json

import pytest

from evaluation.outcome import consolidate, latency_summary
from evaluation.run import persist


@pytest.mark.parametrize("values", [[], [-1], [float("nan")], [float("inf")]])
def test_invalid_latency_samples_fail(values):
    with pytest.raises(ValueError):
        latency_summary(values)


def test_nearest_rank_p95_and_mean():
    result = latency_summary(list(range(1, 21)))
    assert result["p95_seconds"] == 19
    assert result["mean_seconds"] == 10.5


def test_consolidation_counts_retry_time_and_usage_once_and_checks_review_hash(tmp_path):
    source, retry = tmp_path / "first", tmp_path / "retry"
    for folder in (source, retry):
        (folder / "responses").mkdir(parents=True)
        (folder / "judgments").mkdir()
    persist(source / "manifest.json", {"fixture": True})
    persist(
        retry / "retry-provenance.json",
        {
            "source_manifest_sha256": hashlib.sha256(
                (source / "manifest.json").read_bytes()
            ).hexdigest()
        },
    )
    rows = []
    for number in range(1, 73):
        for arm in ("A", "F"):
            name = f"q{number:02}-{arm}.json"
            original = {"application_seconds": 2}
            selected = (
                {"application_seconds": 3}
                if number == 1 and arm == "F"
                else {"application_seconds": 2, "reused_from": "first"}
            )
            persist(source / "responses" / name, original)
            persist(retry / "responses" / name, selected)
            rows.append(
                {
                    "id": f"q{number:02}",
                    "arm": arm,
                    "status": "completed",
                    "grading_status": "completed",
                }
            )
    report = {"rows": rows, "by_arm": {"A": {}, "F": {}}}
    for folder in (source, retry):
        persist(folder / "report.json", report)
    persist(retry / "human-review-queue.json", {"items": [{"human_status": "reviewed"}]})
    persist(
        retry / "human-adjudicated-summary.json",
        {
            "source_review_sha256": hashlib.sha256(
                (retry / "human-review-queue.json").read_bytes()
            ).hexdigest(),
            "by_arm": {"A": {}, "F": {}},
            "paired_counts": {},
            "reviewed_question_pairs": 24,
            "unreviewed_question_pairs": 48,
            "remaining_corrections": [],
            "limitations": [],
        },
    )
    persist(
        source / "judgments/q01-A.json", {"usage": {"prompt_tokens": 10, "completion_tokens": 2}}
    )
    persist(
        retry / "judgments/q01-A.json",
        {"reused_from": "first", "usage": {"prompt_tokens": 10, "completion_tokens": 2}},
    )
    persist(source / "judgments/q01-F.json", {"status": "error"})
    persist(
        retry / "judgments/q01-F.json", {"usage": {"prompt_tokens": 20, "completion_tokens": 3}}
    )
    result = consolidate(source, retry)
    assert result["by_arm"]["F"]["application_retries"] == 1
    assert (
        result["by_arm"]["F"]["application_attempt_sum_latency"]["mean_seconds"]
        == (71 * 2 + 5) / 72
    )
    assert result["grading_usage"]["attempts"] == 3
    assert result["grading_usage"]["reported_prompt_tokens"] == 30
    assert result["grading_usage"]["unknown_usage_attempts"] == 1
    queue = retry / "human-review-queue.json"
    queue.write_text(json.dumps({"items": []}))
    with pytest.raises(ValueError, match="review changed"):
        consolidate(source, retry)
