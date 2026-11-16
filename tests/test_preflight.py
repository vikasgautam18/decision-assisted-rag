import pytest

from evaluation.preflight import candidate_inventory, replica_capacity
from ingestion.checkpoint import CheckpointQuota


def test_replica_capacity_does_not_invent_new_quota_or_throughput():
    rows = replica_capacity(
        [{"name": "model", "used": 100, "limit": 250, "unit": "Count"}],
        {"model": 200, "missing": 30},
    )
    assert rows[0]["status"] == "insufficient"
    assert rows[0]["available_units"] == 150
    assert rows[1]["status"] == "missing_or_ambiguous_quota"


def test_existing_replicas_are_not_reported_as_uncreated_or_silently_reconfigured():
    candidate = {
        "name": "replica",
        "model": "gpt-test",
        "version": "version",
        "sku": "GlobalStandard",
        "capacity": 60,
        "upgrade": "NoAutoUpgrade",
    }
    assert candidate_inventory([candidate], [])[0]["deployment_status"] == "not_created"
    deployed = {**candidate, "state": "Succeeded"}
    assert (
        candidate_inventory([candidate], [deployed])[0]["deployment_status"]
        == "existing_matches_proposal"
    )
    assert (
        candidate_inventory([candidate], [{**deployed, "version": "different"}])[0][
            "deployment_status"
        ]
        == "existing_configuration_mismatch"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        candidate_inventory([candidate], [deployed, deployed])


def test_explicit_cumulative_caps_keep_all_old_reservations(tmp_path):
    path = tmp_path / "existing.sqlite3"
    quota = CheckpointQuota(path, limits={"di_analysis": 6, "decision": 1, "model": 1})
    first = quota.reserve("decision")
    with pytest.raises(RuntimeError, match="exhausted"):
        quota.reserve("decision")
    increased = CheckpointQuota(path, limits={"di_analysis": 6, "decision": 2, "model": 1})
    second = increased.reserve("decision")
    assert second != first
    assert len(increased.snapshot()) == 2
    assert CheckpointQuota.limits["decision"] == 4800
    with pytest.raises(ValueError):
        CheckpointQuota(path, limits={"decision": 100})
