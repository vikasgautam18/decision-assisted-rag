import asyncio
import json

import httpx
import pytest

from evaluation.dataset import DATA, Case, audit, load_cases
from evaluation.judge import GradingQuota, Judgment, grade
from evaluation.report import summarize
from evaluation.run import capture, check_server
from evaluation.scoring import deterministic, trace_checks
from ingestion.checkpoint import CheckpointQuotaExhausted
from pipelines.budget import ServiceError

SOURCE = {
    "id": "NOVA-3-table-6",
    "document": "NOVA-3",
    "revision": "D",
    "page_start": 4,
    "page_end": 4,
    "text": "Range 310 km",
    "historical": False,
}


def answer():
    return {
        "status": "completed",
        "behavior": "answer",
        "semantic_verification": "passed",
        "answer": "310 km [NOVA-3-table-6]",
        "citations": [
            {
                "source_id": SOURCE["id"],
                **{key: SOURCE[key] for key in ("document", "revision", "page_start", "page_end")},
            }
        ],
        "figures": [],
    }


def test_original_datasets_audited_without_rewriting_labels():
    cases = load_cases()
    report = audit()
    assert len(cases) == 72 and len({case.id for case in cases}) == 72
    assert report["behaviors"] == {"answer": 64, "flag_conflict": 1, "abstain": 4, "clarify": 3}
    assert report["figure_cases"] == 7 and len(report["categories"]) == 14
    assert report["new_questions_added"] == 0


def test_invalid_gold_page_and_extra_fields_fail_strictly():
    raw = json.loads((DATA / "benchmark.jsonl").read_text().splitlines()[0])
    raw["gold_sources"][0]["page"] = True
    with pytest.raises(ValueError):
        Case.model_validate_json(json.dumps(raw))
    raw["gold_sources"][0]["page"] = 4
    raw["extra"] = "not in original schema"
    with pytest.raises(ValueError):
        Case.model_validate_json(json.dumps(raw))


def test_matching_page_is_not_enough_for_forged_citation_identity():
    case = load_cases()[0]
    good = deterministic(case, answer(), {SOURCE["id"]: SOURCE}, {})
    assert good["pass"]
    response = answer()
    response["citations"][0]["source_id"] = "invented"
    assert not deterministic(case, response, {SOURCE["id"]: SOURCE}, {})["pass"]


def test_behavior_error_and_unverified_answer_never_pass():
    case = load_cases()[0]
    for changed in (
        {"status": "error"},
        {"behavior": "abstain"},
        {"semantic_verification": "not_run"},
        {"answer": "310 km [foreign]"},
    ):
        assert not deterministic(case, {**answer(), **changed}, {SOURCE["id"]: SOURCE}, {})["pass"]


def test_original_source_alternatives_do_not_require_all_pages():
    case = load_cases()[4]
    source = {**SOURCE, "id": "ZENITH-11-table-6", "document": "ZENITH-11", "revision": "B"}
    response = answer()
    response["answer"] = "Source [ZENITH-11-table-6]"
    response["citations"] = [
        {
            "source_id": source["id"],
            **{key: source[key] for key in ("document", "revision", "page_start", "page_end")},
        }
    ]
    result = deterministic(case, response, {source["id"]: source}, {})
    assert result["checks"]["gold_page_match"] and result["gold_page_recall"] == 0.5


def test_figure_identity_page_and_original_delivery_url_are_required():
    case = next(case for case in load_cases() if case.id == "q31")
    response = answer()
    response["figures"] = [{"key": "NOVA-3-figure-2", "url": "/api/figures/NOVA-3-figure-2"}]
    result = deterministic(
        case, response, {SOURCE["id"]: SOURCE}, {"NOVA-3-figure-2": ("NOVA-3", 10)}
    )
    assert result["checks"]["original_figure"]
    response["figures"][0]["url"] = "https://foreign.invalid/image.png"
    result = deterministic(
        case, response, {SOURCE["id"]: SOURCE}, {"NOVA-3-figure-2": ("NOVA-3", 10)}
    )
    assert not result["checks"]["original_figure"] and result["wrong_figures"]


def test_grading_validation_does_not_coerce_boolean_strings():
    payload = {
        "behavior": "answer",
        "correct": "true",
        "complete": True,
        "claims_supported": True,
        "citations_supported": True,
        "gold_issue": False,
        "uncertain": False,
        "reason": "fixture",
    }
    with pytest.raises(ValueError):
        Judgment.model_validate_json(json.dumps(payload))


def test_grader_never_receives_runtime_identity_verdict_or_probabilities(tmp_path):
    seen = []

    class Session:
        async def request(self, *args, **kwargs):
            seen.append(kwargs["json"])
            return httpx.Response(
                200,
                json={
                    "model": "gpt-5.2-2025-12-11",
                    "usage": {"total_tokens": 1},
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {
                                "content": json.dumps(
                                    {
                                        "behavior": "answer",
                                        "correct": True,
                                        "complete": True,
                                        "claims_supported": True,
                                        "citations_supported": True,
                                        "gold_issue": False,
                                        "uncertain": False,
                                        "reason": "fixture",
                                    }
                                )
                            },
                        }
                    ],
                },
            )

    response = {
        **answer(),
        "execution_mode": "secret-implementation",
        "trace_id": "private-trace",
        "probabilities": {"supported": 0.99},
    }
    result = asyncio.run(
        grade(Session(), GradingQuota(tmp_path / "quota.db"), load_cases()[0], response, [SOURCE])
    )
    user = seen[0]["messages"][1]["content"]
    assert "secret-implementation" not in user and "private-trace" not in user
    assert "probabilities" not in user and "semantic_verification" not in user
    assert result["model"] == "gpt-5.2-2025-12-11"


def test_grading_cap_counts_failed_reservations_across_instances(tmp_path):
    path = tmp_path / "quota.db"
    first = GradingQuota(path)
    for _ in range(first.limits["grading"]):
        first.reserve("grading")
    with pytest.raises(CheckpointQuotaExhausted):
        GradingQuota(path).reserve("grading")


def test_resume_does_not_retry_existing_failed_attempt(tmp_path):
    path = tmp_path / "responses/q01.json"
    path.parent.mkdir()
    path.write_text('{"response":{"status":"error"}}')
    asyncio.run(
        capture(
            load_cases()[0],
            tmp_path,
            "http://127.0.0.1:1",
            "snapshot",
            {},
            {},
            asyncio.Semaphore(1),
        )
    )
    assert path.read_text() == '{"response":{"status":"error"}}'


def test_trace_usage_unknown_is_not_zero_or_reported():
    result = trace_checks(
        {"behavior": "abstain"},
        {
            "calls": [
                {"category": "decision", "usage": None},
                {"category": "model", "usage": {"total_tokens": 10}},
            ]
        },
    )
    assert result["usage_unknown_calls"] == result["usage_reported_calls"] == 1


@pytest.mark.parametrize(
    "payload",
    [
        [],
        {"choices": [None]},
        {"choices": [{"finish_reason": "stop", "message": None}]},
        {
            "model": "wrong-model",
            "choices": [{"finish_reason": "stop", "message": {"content": "{}"}}],
        },
    ],
)
def test_malformed_grader_is_explicit_and_consumes_one_reservation(tmp_path, payload):
    class Session:
        async def request(self, *args, **kwargs):
            return httpx.Response(200, json=payload)

    quota = GradingQuota(tmp_path / "quota.db")
    with pytest.raises(ServiceError):
        asyncio.run(grade(Session(), quota, load_cases()[0], answer(), [SOURCE]))
    assert len(quota.snapshot()) == 1


def test_loaded_server_snapshot_must_match():
    async def execute():
        transport = httpx.MockTransport(
            lambda request: httpx.Response(200, json={"implementation_sha256": "old"})
        )
        async with httpx.AsyncClient(base_url="http://localhost", transport=transport) as client:
            await check_server(client, "expected")

    with pytest.raises(ValueError, match="snapshot"):
        asyncio.run(execute())


def test_attempt_is_durable_before_post_and_crash_is_not_retried(tmp_path, monkeypatch):
    case = load_cases()[0]
    target = tmp_path / "responses" / f"{case.id}.json"
    target.parent.mkdir()
    original_client = httpx.AsyncClient
    requests = []

    def handle(request):
        requests.append(request.method)
        if request.method == "GET":
            return httpx.Response(200, json={"implementation_sha256": "snapshot"})
        assert json.loads(target.read_text())["response"]["error_code"] == (
            "interrupted_or_not_captured"
        )
        raise RuntimeError("simulated process failure")

    monkeypatch.setattr(
        "evaluation.run.httpx.AsyncClient",
        lambda **kwargs: original_client(**kwargs, transport=httpx.MockTransport(handle)),
    )
    with pytest.raises(RuntimeError, match="simulated process failure"):
        asyncio.run(
            capture(case, tmp_path, "http://localhost", "snapshot", {}, {}, asyncio.Semaphore(1))
        )
    asyncio.run(
        capture(case, tmp_path, "http://localhost", "snapshot", {}, {}, asyncio.Semaphore(1))
    )
    assert requests == ["GET", "POST"]


def test_report_keeps_all_72_cases_and_existing_human_notes(tmp_path):
    for name in ("benchmark.jsonl", "heldout.jsonl"):
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "frozen": {"application_sha256": "snapshot", "protocol": {}, "audit": {}},
            }
        )
    )
    report = summarize(tmp_path)
    assert report["cases_expected"] == 72 and report["cases_captured"] == 0
    assert report["strict_provisional_passes"] == 0
    assert report["grading_errors_or_missing"] == 72
    queue = tmp_path / "human-review-queue.json"
    payload = json.loads(queue.read_text())
    payload["items"][0]["human_notes"] = "preserve the review"
    queue.write_text(json.dumps(payload))
    summarize(tmp_path)
    assert json.loads(queue.read_text())["items"][0]["human_notes"] == "preserve the review"
    (tmp_path / "responses").mkdir()
    (tmp_path / "judgments").mkdir()
    (tmp_path / "responses/q01.json").write_text(
        json.dumps(
            {
                "response": {"status": "completed", "behavior": "answer"},
                "deterministic": {"pass": True},
                "application_seconds": 1,
                "trace_checks": {"available": True},
            }
        )
    )
    (tmp_path / "judgments/q01.json").write_text(
        json.dumps(
            {
                "status": "completed",
                "judgment": {
                    "behavior": "answer",
                    "correct": True,
                    "complete": True,
                    "claims_supported": True,
                    "citations_supported": True,
                    "gold_issue": False,
                    "uncertain": False,
                    "reason": "ok",
                },
            }
        )
    )
    summarize(tmp_path)
    preserved = next(item for item in json.loads(queue.read_text())["items"] if item["id"] == "q01")
    assert preserved["human_notes"] == "preserve the review"


def test_dataset_drift_from_pinned_reference_is_rejected(tmp_path):
    for name in ("benchmark.jsonl", "heldout.jsonl"):
        (tmp_path / name).write_bytes((DATA / name).read_bytes() + b"\n")
    with pytest.raises(ValueError, match="pinned dataset"):
        load_cases(tmp_path)


def test_targeted_development_report_does_not_claim_a_72_case_run(tmp_path):
    for name in ("benchmark.jsonl", "heldout.jsonl"):
        (tmp_path / name).write_bytes((DATA / name).read_bytes())
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "frozen": {
                    "application_sha256": "snapshot",
                    "protocol": {"selected_case_ids": ["q18", "q24"]},
                    "audit": {},
                },
            }
        )
    )
    report = summarize(tmp_path)
    assert report["cases_expected"] == 2 and len(report["rows"]) == 2
    assert set(row["id"] for row in report["rows"]) == {"q18", "q24"}
