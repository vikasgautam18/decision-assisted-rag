import asyncio
import hashlib
import json
import subprocess
from types import SimpleNamespace

import httpx
import pytest

from evaluation import paired_pilot
from evaluation.baseline_copy import REFERENCE_ENV, reference_checkout
from evaluation.comparison import (
    capture_public_answer,
    normalize_response,
    paired_schedule,
    validate_origin,
)
from evaluation.dataset import load_cases
from evaluation.paired_pilot import common_evidence, public_checks
from evaluation.paired_run import paired_results, retry_needed
from evaluation.run import persist


def baseline_answer():
    return {
        "answer": "310 km [NOVA-3 p.4]",
        "citations": [{"doc": "NOVA-3", "page_start": 4, "page_end": 4}],
        "figures": [{"key": "NOVA-3-fig01", "url": "/api/figures/NOVA-3-fig01"}],
    }


def test_normalization_preserves_native_answer_without_inventing_runtime_verdicts():
    result = normalize_response("A", baseline_answer())
    assert result.answer == "310 km [NOVA-3 p.4]"
    assert result.citations[0].document == "NOVA-3"
    assert "semantic_verification" not in result.model_dump()
    assert result.figure_keys == ("NOVA-3-fig01",)
    result = normalize_response(
        "F",
        {
            "status": "completed",
            "answer": "310 km [NOVA-3-table-6]",
            "citations": [{"document": "NOVA-3", "page_start": 4, "page_end": 4}],
            "figures": [],
        },
    )
    assert result.citations[0].page_start == 4


@pytest.mark.parametrize(
    "change",
    [
        {"answer": "ERROR provider unavailable"},
        {"answer": None},
        {"citations": [{"doc": "NOVA-3", "page_start": True, "page_end": 4}]},
        {"citations": [{"doc": "FOREIGN", "page_start": 1, "page_end": 1}]},
        {"citations": [{"doc": "NOVA-3", "page_start": 5, "page_end": 4}]},
        {"figures": [{"key": "NOVA-3-fig01", "url": "https://foreign.invalid/image"}]},
    ],
)
def test_invalid_baseline_output_fails_explicitly(change):
    with pytest.raises(ValueError):
        normalize_response("A", {**baseline_answer(), **change})


def test_seeded_schedule_is_paired_balanced_and_reproducible():
    cases = load_cases()
    schedule = paired_schedule(cases, seed=7)
    assert schedule == paired_schedule(cases, seed=7)
    assert schedule != paired_schedule(cases, seed=8)
    assert len(schedule) == 144
    assert sum(arm == "A" for _, arm in schedule[::2]) == 36
    for first, second in zip(schedule[::2], schedule[1::2], strict=True):
        assert first[0] == second[0] and {first[1], second[1]} == {"A", "F"}


@pytest.mark.parametrize(
    "url",
    [
        "http://remote.invalid",
        "https://user:secret@example.invalid",
        "https://example.invalid/path",
        "https://example.invalid?query=value",
        "https://example.invalid#fragment",
    ],
)
def test_unsafe_origins_rejected(url):
    with pytest.raises(ValueError):
        validate_origin(url)


def test_public_adapter_uses_fresh_sessions_without_retries():
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["origin"] == "http://127.0.0.1:8000"
        assert "cookie" not in request.headers
        return httpx.Response(
            200, json=baseline_answer(), headers={"set-cookie": "sandbox_session=fixture"}
        )

    async def scenario():
        for _ in range(2):
            result = await capture_public_answer(
                "A",
                "http://127.0.0.1:8000",
                load_cases()[0],
                timeout_seconds=180,
                transport=httpx.MockTransport(handler),
            )
            assert result["status"] == "completed" and result["usage"] is None

    asyncio.run(scenario())
    assert len(requests) == 2


@pytest.mark.parametrize("failure", ["http", "transport", "body"])
def test_operational_failures_are_not_answers_or_retried(failure):
    requests = []

    def handler(request):
        requests.append(request)
        if failure == "transport":
            raise httpx.ConnectTimeout("fixture", request=request)
        if failure == "http":
            return httpx.Response(503)
        return httpx.Response(200, json={"answer": "ERROR unavailable"})

    result = asyncio.run(
        capture_public_answer(
            "A",
            "http://127.0.0.1:8000",
            load_cases()[0],
            timeout_seconds=180,
            transport=httpx.MockTransport(handler),
        )
    )
    assert result["status"] == "error"
    assert len(requests) == 1


def test_common_checks_do_not_require_private_verifier_or_calculation_records():
    case = load_cases()[0]
    answer = normalize_response("A", {**baseline_answer(), "figures": []})
    record = {
        "status": "completed",
        "public_answer": answer.model_dump(mode="json"),
        "images": [],
    }
    assert all(public_checks(case, record).values())
    assert not all(public_checks(case, {"status": "error"}).values())


def test_required_figure_must_be_delivered_and_match_gold_page():
    case = next(case for case in load_cases() if case.id == "q28")
    answer = normalize_response(
        "A",
        {
            "answer": "No fault found [VOLTA-4 p.10]",
            "citations": [{"doc": "VOLTA-4", "page_start": 10, "page_end": 10}],
            "figures": [],
        },
    )
    record = {"status": "completed", "public_answer": answer.model_dump(mode="json"), "images": []}
    assert not public_checks(case, record)["required_figure"]
    record["images"] = [{"valid_location": True, "matches_gold_page": True, "delivered_png": True}]
    assert public_checks(case, record)["required_figure"]
    record["images"][0]["delivered_png"] = False
    assert not public_checks(case, record)["required_figure"]


def test_common_judge_evidence_includes_cited_and_gold_pages_without_private_ids():
    case = load_cases()[0]
    answer = normalize_response("A", {**baseline_answer(), "figures": []})
    catalogue = {
        "source": {"id": "source", "document": "NOVA-3", "page_start": 4, "page_end": 4},
        "other": {"id": "other", "document": "NOVA-3", "page_start": 9, "page_end": 9},
    }
    assert common_evidence(case, answer, catalogue) == [catalogue["source"]]


@pytest.mark.parametrize(
    ("application", "grading", "expected"),
    [
        ("error", "not_run", (True, True)),
        ("completed", "error", (False, True)),
        ("completed", "missing", (False, True)),
        ("completed", "completed", (False, False)),
    ],
)
def test_retry_selection_never_resamples_completed_semantic_judgments(
    application, grading, expected
):
    assert (
        retry_needed(
            {"status": application},
            {"status": grading, "judgment": {"correct": False}},
        )
        == expected
    )


def test_paired_counts_keep_unknown_grading_distinct_from_semantic_failures():
    rows = []
    for case_id, a, f in (
        ("q01", True, True),
        ("q02", True, False),
        ("q03", False, True),
        ("q39", False, False),
    ):
        for arm, passed in (("A", a), ("F", f)):
            rows.append(
                {
                    "id": case_id,
                    "arm": arm,
                    "status": "completed",
                    "strict_provisional_pass": passed,
                    "grading_status": "error" if case_id == "q39" else "completed",
                }
            )
    result = paired_results({"rows": rows})
    assert result["counts"] == {
        "both_pass": 1,
        "A_only_pass": 1,
        "F_only_pass": 1,
        "neither_pass": 1,
    }
    assert result["pairs"][-1]["grading_unknown_arms"] == ["A", "F"]


def test_retry_execution_preserves_originals_and_resumes_without_spending_again(
    tmp_path, monkeypatch
):
    from evaluation import paired_run

    cases = [case for case in load_cases() if case.id in ("q01", "q39")]
    monkeypatch.setattr(paired_run, "ROOT", tmp_path)
    monkeypatch.setattr(paired_run, "load_cases", lambda: cases)
    monkeypatch.setattr(paired_run, "catalogue_at", lambda root: ({}, {}))
    source = tmp_path / "comparison"
    for folder in ("responses", "judgments"):
        (source / folder).mkdir(parents=True)
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests/azure-resources.json").write_text("{}")
    schedule = [(case.id, arm) for case in cases for arm in ("A", "F")]
    persist(
        source / "manifest.json",
        {
            "schedule": schedule,
            "origins": {"A": "http://127.0.0.1:8001", "F": "http://127.0.0.1:8765"},
            "timeouts": {"A": 420, "F": 195},
        },
    )
    public = normalize_response("A", {**baseline_answer(), "figures": []}).model_dump(mode="json")
    completed = {
        "status": "completed",
        "public_answer": public,
        "public_checks": {"completed": True},
        "application_seconds": 1,
    }
    judgment = {
        "behavior": "answer",
        "correct": False,
        "complete": True,
        "claims_supported": True,
        "citations_supported": True,
        "gold_issue": False,
        "uncertain": False,
        "reason": "fixture",
    }
    for case_id, arm in schedule:
        persist(
            source / f"responses/{case_id}-{arm}.json",
            {"status": "error"} if (case_id, arm) == ("q01", "A") else completed,
        )
        persist(
            source / f"judgments/{case_id}-{arm}.json",
            {"status": "error"} if arm == "A" else {"status": "completed", "judgment": judgment},
        )
    persist(source / "report.json", {"by_arm": {}})
    originals = {path: path.read_bytes() for path in source.rglob("*.json")}
    calls = []

    async def preflight(manifest):
        pass

    async def capture(arm, origin, case, **kwargs):
        calls.append(("application", case.id, arm))
        return completed.copy()

    async def images(*args):
        return []

    async def grade(*args):
        case = args[2]
        calls.append(("grading", case.id))
        return {"judgment": {**judgment, "correct": True, "behavior": case.expected_behavior}}

    class Credential:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

    monkeypatch.setattr(paired_run, "check_snapshot", preflight)
    monkeypatch.setattr(paired_run, "capture_public_answer", capture)
    monkeypatch.setattr(paired_run, "check_images", images)
    monkeypatch.setattr(paired_run, "grade", grade)
    monkeypatch.setattr(paired_run, "AzureCliCredential", Credential)
    monkeypatch.setattr(paired_run, "AzureSession", lambda *args: None)
    monkeypatch.setattr(paired_run, "GradingQuota", lambda *args: None)
    monkeypatch.setattr(
        paired_run, "ExperimentResources", SimpleNamespace(model_validate_json=lambda data: None)
    )
    asyncio.run(paired_run.retry_errors(source))
    assert calls == [("application", "q01", "A"), ("grading", "q01"), ("grading", "q39")]
    asyncio.run(paired_run.retry_errors(source))
    assert len(calls) == 3
    assert all(path.read_bytes() == contents for path, contents in originals.items())
    report = json.loads((tmp_path / "comparison-retry1/report.json").read_text())
    assert report["first_attempt_by_arm"] == {}
    assert len(report["rows"]) == 4


def fixture_reference(path):
    git = ["git", "-C", str(path)]
    subprocess.run(["git", "init", "-q", str(path)], check=True)
    (path / "backend.py").write_text("print('baseline')\n")
    subprocess.run([*git, "add", "backend.py"], check=True)
    commit = ["-c", "user.name=fixture", "-c", "user.email=fixture@example.invalid"]
    subprocess.run(
        [*git, *commit, "-c", "commit.gpgsign=false", "commit", "-q", "-m", "x"], check=True
    )
    return subprocess.check_output([*git, "rev-parse", "HEAD"], text=True).strip()


def test_reference_checkout_reads_only_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv(REFERENCE_ENV, raising=False)
    assert reference_checkout() is None
    monkeypatch.setenv(REFERENCE_ENV, "")
    assert reference_checkout() is None
    monkeypatch.setenv(REFERENCE_ENV, str(tmp_path))
    assert reference_checkout() == tmp_path


def test_baseline_snapshot_requires_a_configured_reference(monkeypatch):
    monkeypatch.delenv(REFERENCE_ENV, raising=False)
    with pytest.raises(ValueError, match=REFERENCE_ENV):
        paired_pilot.baseline_snapshot()


def test_baseline_snapshot_hashes_the_configured_clean_checkout(monkeypatch, tmp_path):
    commit = fixture_reference(tmp_path)
    monkeypatch.setenv(REFERENCE_ENV, str(tmp_path))
    monkeypatch.setattr(paired_pilot, "REFERENCE_COMMIT", commit)
    snapshot = paired_pilot.baseline_snapshot()
    assert snapshot["commit"] == commit
    assert snapshot["tracked_files"] == {
        "backend.py": hashlib.sha256(b"print('baseline')\n").hexdigest()
    }


def test_baseline_snapshot_rejects_a_dirty_or_unpinned_checkout(monkeypatch, tmp_path):
    commit = fixture_reference(tmp_path)
    monkeypatch.setenv(REFERENCE_ENV, str(tmp_path))
    monkeypatch.setattr(paired_pilot, "REFERENCE_COMMIT", "0" * 40)
    with pytest.raises(ValueError, match="clean at the pinned revision"):
        paired_pilot.baseline_snapshot()
    monkeypatch.setattr(paired_pilot, "REFERENCE_COMMIT", commit)
    (tmp_path / "backend.py").write_text("print('edited')\n")
    with pytest.raises(ValueError, match="clean at the pinned revision"):
        paired_pilot.baseline_snapshot()
