import asyncio
import json

import httpx
import pytest
from test_experiment import approval, protocol

from app.live import create_live_app
from evaluation.dataset import load_cases
from evaluation.experiment import CostScope, ExperimentLedger, schedule
from evaluation.observed_grade import review_case, selected_records
from evaluation.observed_report import clustered_difference, distribution, review_queue, summarize
from evaluation.observed_run import collect_one, materialize_cases


def test_distribution_and_cluster_resampling_keep_questions_as_unit():
    assert distribution([])["mean"] is None
    assert distribution([1, 2, 3])["median"] == 2
    with pytest.raises(ValueError):
        distribution([float("nan")])
    pairs = [("q01", 3, 1)] * 5 + [("q02", 2, 1)] * 5
    result = clustered_difference(pairs, samples=100)
    assert result["question_clusters"] == 2
    assert result["matched_observations"] == 10
    assert result["mean_question_difference"] == 1.5
    assert result == clustered_difference(pairs, samples=100)


def test_report_preserves_missing_observations_and_ungraded_answers():
    p = protocol("pilot")
    row = schedule(p)[0]
    record = {
        **row.model_dump(),
        "status": "completed",
        "application_seconds": 2.0,
        "public_checks": {"completed": True},
    }
    report = summarize(p, [record], {})
    assert report["by_arm"][row.arm]["grading_unknown_delivered"] == 1
    assert report["by_arm"][row.arm]["pass_fraction_all_scheduled"] == 0
    assert report["by_arm"][row.arm]["not_captured"] == 5
    with pytest.raises(ValueError, match="duplicate"):
        summarize(p, [record, record], {})


@pytest.mark.parametrize("phase", ["pilot", "load"])
def test_all_report_phases_disclose_the_same_user_cost_exclusions(phase):
    p = protocol(phase).model_copy(
        update={
            "cost_scope": CostScope(
                excluded_categories=("search_retrieval", "defender_ai"),
                exclusion_reference="User: ignore these two for both implementations",
            )
        }
    )
    report = summarize(p, [], {})
    assert report["cost_scope"] == p.cost_scope.report()
    assert report["cost_scope"]["applies_to_arms"] == list("ABCDEFG")
    assert not report["cost_scope"]["total_serving_cost_claim"]


def test_collection_has_exact_correlation_and_never_redispatches_intent(tmp_path):
    p = protocol("pilot")
    row = next(item for item in schedule(p) if item.arm == "A")
    case = next(item for item in load_cases() if item.id == row.case_id)
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p))
    for folder in ("responses", "runner-spans", "private"):
        (tmp_path / folder).mkdir()
    posts = []

    def handle(request):
        if request.url.path == "/health":
            return httpx.Response(
                200,
                json={
                    "arm": "A",
                    "experiment_id": p.experiment_id,
                    "implementation_sha256": p.targets["A"].application_sha256,
                },
            )
        posts.append(request)
        assert request.headers["x-observation-id"] == row.observation_id
        assert json.loads(request.content)["question"] == case.question
        return httpx.Response(
            200,
            json={
                "answer": "Test answer.",
                "citations": [],
                "figures": [],
            },
            headers={
                "X-Observation-ID": row.observation_id,
                "X-Experiment-Trace-ID": "f" * 32,
            },
        )

    for _ in range(2):
        asyncio.run(
            collect_one(
                row,
                case,
                p,
                tmp_path,
                ledger,
                {},
                transport=httpx.MockTransport(handle),
            )
        )
    assert len(posts) == 1
    result = json.loads((tmp_path / "responses" / f"{row.observation_id}.json").read_text())
    assert result["correlation"]["x-observation-id"] == row.observation_id
    assert ledger.records()[0]["state"] == "recorded"


def test_grading_cannot_overlap_incomplete_serving_and_overlay_preserves_gold(tmp_path):
    p = protocol("pilot")
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p))
    with pytest.raises(ValueError, match="complete serving"):
        selected_records(p, ledger)
    case = load_cases()[0]
    updated = review_case(
        case,
        {
            "cases": {
                case.id: {
                    "guidance": "Check the actual variant.",
                    "review_reference": "offline-fixture",
                }
            },
        },
    )
    assert updated.gold_answer == case.gold_answer
    assert updated.notes != case.notes


def test_experiment_api_identity_is_not_an_arm_selector(tmp_path):
    from fastapi.testclient import TestClient

    class Runner:
        async def run(self, question, history):
            return {
                "status": "completed",
                "behavior": "abstain",
                "answer": "Fixture.",
                "citations": [],
                "figures": [],
                "semantic_verification": "not_run",
                "trace_id": "b" * 32,
            }

    p = protocol("pilot")
    row = next(item for item in schedule(p) if item.arm == "F")
    app = create_live_app(
        tmp_path,
        runner=Runner(),
        arm="F",
        experiment_id=p.experiment_id,
        telemetry_directory=tmp_path / "spans",
    )
    with TestClient(app, headers={"Origin": "http://testserver"}) as client:
        assert client.post("/api/chat", json={"question": "test"}).status_code == 400
        result = client.post("/api/chat", json={"question": "test"}, headers=row.headers)
        assert result.status_code == 200
        assert result.headers["x-observation-id"] == row.observation_id
        assert len(result.headers["x-experiment-trace-id"]) == 32
        wrong = {**row.headers, "X-Experiment-Arm": "B"}
        assert client.post("/api/chat", json={"question": "test"}, headers=wrong).status_code == 400
        assert client.get("/health").json()["arm"] == "F"
    events = [
        json.loads(line)
        for path in (tmp_path / "spans").glob("*.jsonl")
        for line in path.read_text().splitlines()
    ]
    api_end = next(
        event for event in events if event["event"] == "span_end" and event["name"] == "api.chat"
    )
    assert api_end["pass_number"] == str(row.pass_number)
    assert api_end["attempt"] == "1"
    assert api_end["attributes"]["response_body_sent_seconds"] >= 0
    root_end = next(
        event for event in events if event["event"] == "span_end" and event["name"] == "observation"
    )
    assert root_end["attributes"]["process_cpu_interval_seconds"] >= 0


def test_current_provider_cache_write_detail_is_not_lost():
    from clients.telemetry import normalized_usage

    assert (
        normalized_usage(
            {
                "input_tokens": 100,
                "input_tokens_details": {"cache_write_tokens": 80},
            }
        )["cache_write_tokens"]
        == 80
    )


def test_load_results_report_achieved_overlap_without_reusing_quality_totals():
    p = protocol("load")
    row = schedule(p)[0]
    record = {
        **row.model_dump(),
        "status": "completed",
        "application_seconds": 2.0,
        "started_utc": "2026-11-04T00:00:00+00:00",
        "completed_utc": "2026-11-04T00:00:02+00:00",
    }
    report = summarize(p, [record], {})
    cell = next(
        item
        for item in report["cells"]
        if item["arm"] == row.arm and item["target_concurrency"] == row.concurrency
    )
    assert cell["client_observed_peak_in_flight"] == 1
    assert cell["completed_per_second"] == 0.5
    assert "by_arm" not in report


def test_review_deduplication_does_not_transfer_an_old_human_decision():
    p = protocol("pilot")
    first, second = schedule(p)[:2]
    records = [
        {
            **row.model_dump(),
            "status": "completed",
            "public_answer": {"answer": "Identical.", "citations": []},
        }
        for row in (first, second)
    ]
    queue = review_queue(p, records, {})
    assert len(queue["groups"]) == 1
    assert queue["groups"][0]["human_status"] == "pending"
    assert queue["groups"][0]["observation_ids"] == [
        first.observation_id,
        second.observation_id,
    ]


def test_synthetic_pilot_question_text_requires_its_own_freeze_hash():
    p = protocol("pilot")
    original = load_cases()[0]
    synthetic = original.model_copy(update={"id": "q7102", "question": "Synthetic variant."})
    p = p.model_copy(update={"case_ids": (*p.case_ids, synthetic.id)})
    with pytest.raises(ValueError, match="frozen protocol"):
        materialize_cases(p, [synthetic])
