import json
from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest

from evaluation.experiment import (
    Approval,
    CostScope,
    ExperimentLedger,
    Protocol,
    schedule,
    schedule_audit,
)
from evaluation.observed_run import ROOT, code_hash, prepare, retry_rows
from pipelines.arms import ARMS, policy_for


def protocol(phase="main"):
    count = 12 if phase == "load" else 72 if phase == "main" else 6
    return Protocol.model_validate_json(
        json.dumps(
            {
                "experiment_id": "test-instrumented",
                "phase": phase,
                "seed": 71,
                "passes": 5 if phase == "main" else 1,
                "case_ids": [f"q{index:02}" for index in range(1, count + 1)],
                "targets": {
                    arm: {
                        "origin": f"http://127.0.0.1:{8800 + index}",
                        "application_sha256": "a" * 64,
                        "timeout_seconds": 195,
                        "placement": "pinned-original-placement",
                    }
                    for index, arm in enumerate(ARMS)
                },
                "code_sha256": code_hash(ROOT),
                "dataset_sha256": "c" * 64,
                "rate_card_sha256": "d" * 64,
                "review_overlay_sha256": "e" * 64,
                "baseline_visibility_accepted": True,
            }
        )
    )


def approval(p, attempts=2520, spend="100"):
    return Approval(
        protocol_sha256=p.sha256,
        approval_reference="offline-test-fixture-not-real-approval",
        user_approved_paid_execution=True,
        max_application_attempts=attempts,
        max_grading_attempts=10,
        max_reserved_usd=Decimal(spend),
        reservation_usd_per_application=Decimal("1"),
        reservation_usd_per_grade=Decimal("0.5"),
        cost_ceiling_kind="conservative_reservation_not_invoice_guarantee",
    )


def test_main_schedule_exact_counts_identity_and_determinism():
    p = protocol()
    rows = schedule(p)
    assert rows == schedule(p)
    audit = schedule_audit(rows)
    assert audit["observations"] == audit["unique_observation_ids"] == 2520
    assert audit["by_arm"] == dict.fromkeys(ARMS, 360)
    assert len({(r.case_id, r.arm, r.pass_number) for r in rows}) == 2520
    assert all(r.attempt == 1 and r.concurrency == 1 for r in rows)
    assert len({r.logical_request_id for r in rows}) == 2520
    for arm in ARMS:
        counts = audit["arm_positions"][arm].values()
        assert max(counts) - min(counts) <= 2


def test_cost_exclusions_require_user_reference_and_are_hash_frozen_for_all_arms(tmp_path):
    p = protocol("pilot")
    scope = CostScope(
        excluded_categories=("search_retrieval", "defender_ai"),
        exclusion_reference="User: ignore these two for both implementations",
    )
    scoped = p.model_copy(update={"cost_scope": scope})
    assert scoped.sha256 != p.sha256
    directory = tmp_path / "scoped"
    prepare(directory, scoped)
    persisted = Protocol.model_validate_json((directory / "protocol.json").read_bytes())
    assert persisted.cost_scope == scope
    assert persisted.cost_scope.report()["applies_to_arms"] == list(ARMS)
    assert not persisted.cost_scope.report()["excluded_charges_are_zero"]
    assert not persisted.cost_scope.report()["total_serving_cost_claim"]
    assert not json.loads((directory / "status.json").read_text())["paid_calls_made"]
    with pytest.raises(ValueError, match="user decision"):
        CostScope(excluded_categories=("search_retrieval",))
    with pytest.raises(ValueError, match="duplicate"):
        CostScope(
            excluded_categories=("defender_ai", "defender_ai"),
            exclusion_reference="offline fixture",
        )
    with pytest.raises(ValueError):
        CostScope.model_validate_json(
            '{"excluded_categories":["writer"],"exclusion_reference":"not permitted"}'
        )


def test_load_schedule_separates_levels():
    rows = schedule(protocol("load"))
    assert len(rows) == 252
    assert len({row.observation_id for row in rows}) == 252
    assert {row.concurrency for row in rows} == {1, 2, 4}


def test_invalid_scope_and_custom_baseline_are_rejected():
    with pytest.raises(ValueError, match="72 cases"):
        Protocol.model_validate_json(
            protocol().model_dump_json().replace('"passes":5', '"passes":1')
        )
    with pytest.raises(ValueError):
        policy_for("A")
    assert not policy_for("B").verification
    assert policy_for("G").ordinary_decisions


def test_prepare_is_no_spend_and_never_overwrites(tmp_path):
    p = protocol()
    target = tmp_path / "new"
    prepare(target, p)
    assert json.loads((target / "status.json").read_text())["paid_calls_made"] == 0
    assert not (target / "accounting.sqlite3").exists()
    assert code_hash(target / "source") == p.code_sha256
    snapshot = target / "source/evaluation/experiment.py"
    original = snapshot.read_bytes()
    snapshot.write_bytes(original + b"\n# changed archived source\n")
    assert code_hash(target / "source") != p.code_sha256
    with pytest.raises(ValueError, match="exists"):
        prepare(target, p)


def test_ledger_keeps_ambiguous_intent_and_enforces_ceiling(tmp_path):
    p = protocol()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p, attempts=1))
    assert ledger.reserve("one", "application")
    assert not ledger.reserve("one", "application")
    with pytest.raises(ValueError, match="ceiling"):
        ledger.reserve("two", "application")
    assert ledger.records()[0]["state"] == "reserved"
    with pytest.raises(ValueError, match="already"):
        ledger.complete("missing", {})
    ledger.complete("one", {"status": "error"})
    assert not ledger.reserve("one", "application")
    with pytest.raises(ValueError, match="different"):
        ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p, attempts=2))


def test_preparation_rejects_source_drift_before_creating_a_run(tmp_path):
    p = protocol("pilot").model_copy(update={"code_sha256": "0" * 64})
    target = tmp_path / "not-created"
    with pytest.raises(ValueError, match="source changed before preparation"):
        prepare(target, p)
    assert not target.exists()


def test_spend_reservations_cannot_race(tmp_path):
    p = protocol()
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p, spend="2"))

    def reserve(key):
        try:
            return ledger.reserve(str(key), "application")
        except ValueError:
            return False

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(reserve, range(20)))
    assert sum(results) == 2
    assert len(ledger.records()) == 2


def test_retry_requires_reconciled_intent_and_never_retries_semantic_failure(tmp_path):
    p = protocol("pilot")
    row = schedule(p)[0]
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p))
    ledger.reserve(row.observation_id, "application")
    with pytest.raises(ValueError, match="uncertain dispatch"):
        retry_rows([row], ledger)
    ledger.complete(row.observation_id, {"status": "completed", "behavior": "abstain"})
    assert retry_rows([row], ledger) == []


def test_operational_retry_uses_new_identity(tmp_path):
    p = protocol("pilot")
    row = schedule(p)[0]
    ledger = ExperimentLedger(tmp_path / "ledger.sqlite3", p, approval(p))
    ledger.reserve(row.observation_id, "application")
    ledger.complete(row.observation_id, {"status": "error"})
    retry = retry_rows([row], ledger)[0]
    assert retry.attempt == 2
    assert retry.observation_id != row.observation_id
    assert retry.logical_request_id == row.logical_request_id
