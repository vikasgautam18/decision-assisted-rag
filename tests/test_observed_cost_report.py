import json
import subprocess
import sys
from decimal import Decimal

import pytest

from clients.telemetry import observation, span
from evaluation.experiment import CostScope, Evaluator, Protocol, identity, schedule
from evaluation.observed_cost_report import (
    Inventory,
    RateCard,
    build_report,
    file_hash,
    hydrate_call,
    load_frozen,
)
from evaluation.observed_cost_template import (
    ACTIVE_AGENT,
    FINDINGS,
    MODERN,
    RESOLUTION,
    RETAIL,
    create_templates,
)
from evaluation.observed_costs import Meter, price_usage, select_meter
from evaluation.observed_spans import collect


def dump(path, value):
    path.write_text(json.dumps(value))


@pytest.fixture
def frozen(tmp_path):
    evidence = tmp_path / "fixture-evidence.json"
    dump(evidence, {"source": "synthetic offline test only; not an official tier threshold"})
    binding = {
        "binding_id": "writer",
        "arms": ["B"],
        "call_kinds": ["model.attempt"],
        "purpose": "serving",
        "application_sha256_by_arm": {"B": "a" * 64},
        "resource_id": "https://fixture.example",
        "endpoint_host": "fixture.example",
        "deployment": "writer-deployment",
        "model": "fixture-model",
        "model_version": "v1",
        "reported_models": ["fixture-model", "fixture-model-v1"],
        "role": "writer",
        "region": "fixture-region",
        "sku": "GlobalStandard",
        "context_tiering": True,
        "configuration_evidence": evidence.name,
    }
    baseline = {
        **binding,
        "binding_id": "baseline",
        "arms": ["A"],
        "application_sha256_by_arm": {"A": "a" * 64},
        "call_kinds": ["baseline.sdk_create", "baseline.sdk_http_attempt"],
        "deployment": "baseline-exact-configured-deployment",
        "allow_unobserved_deployment": True,
    }
    grading = [
        {
            **binding,
            "binding_id": f"grader-{number}",
            "arms": list("ABCDEFG"),
            "call_kinds": ["evaluation.provider"],
            "application_sha256_by_arm": {},
            "purpose": "evaluation",
            "role": "grading",
            "deployment": f"grader-{number}",
        }
        for number in range(1, 4)
    ]
    inventory = {
        "schema_version": "1",
        "status": "frozen",
        "experiment_id": "fixture-pilot",
        "bindings": [binding, baseline, *grading],
    }
    inventory_path = tmp_path / "inventory.json"
    dump(inventory_path, inventory)
    meters = []
    for item in inventory["bindings"]:
        for tier, low, high, rates in (
            ("short", 0, 100, ("4", "0.4", "5", "20")),
            ("long", 101, None, ("8", "0.8", "10", "30")),
        ):
            meters.append(
                {
                    **{
                        key: item[key]
                        for key in (
                            "model",
                            "model_version",
                            "resource_id",
                            "deployment",
                            "role",
                            "region",
                            "sku",
                        )
                    },
                    "source": "offline-fixture",
                    "effective_date": "fixture",
                    "meter_ids": ["input", "cache-read", "cache-write", "output"],
                    "component_meter_ids": dict(
                        zip(
                            (
                                "input_tokens",
                                "cached_tokens",
                                "cache_write_tokens",
                                "output_tokens",
                            ),
                            ("input", "cache-read", "cache-write", "output"),
                            strict=True,
                        )
                    ),
                    "price_kind": "official_retail",
                    "input_per_million": rates[0],
                    "cached_per_million": rates[1],
                    "cache_write_per_million": rates[2],
                    "output_per_million": rates[3],
                    "cache_semantics": "disjoint_input_subsets",
                    "semantics_evidence": evidence.name,
                    "evidence_refs": [evidence.name],
                    "context_tier": tier,
                    "input_min_tokens": low,
                    "input_max_tokens": high,
                    "context_evidence": evidence.name,
                }
            )
    card_path = tmp_path / "rate-card.json"
    dump(
        card_path,
        {
            "schema_version": "1",
            "status": "frozen",
            "inventory_sha256": file_hash(inventory_path),
            "evidence_sha256": {evidence.name: file_hash(evidence)},
            "meters": meters,
        },
    )
    protocol = Protocol(
        experiment_id="fixture-pilot",
        phase="pilot",
        seed=1,
        passes=1,
        case_ids=("q01",),
        targets={
            arm: {
                "origin": f"http://127.0.0.1:{8800 + index}",
                "application_sha256": "a" * 64,
                "timeout_seconds": 195,
                "placement": "offline",
            }
            for index, arm in enumerate("ABCDEFG")
        },
        code_sha256="b" * 64,
        dataset_sha256="c" * 64,
        rate_card_sha256=file_hash(card_path),
        review_overlay_sha256="d" * 64,
        evaluator_pool=tuple(
            Evaluator(
                deployment=f"grader-{number}", tokens_per_minute=60000, requests_per_minute=600
            )
            for number in range(1, 4)
        ),
        cost_scope=CostScope(
            excluded_categories=("search_retrieval", "defender_ai"),
            exclusion_reference="User: ignore these two for both implementations",
        ),
    )
    protocol_path = tmp_path / "protocol.json"
    dump(protocol_path, protocol.model_dump(mode="json"))
    return protocol, protocol_path, inventory_path, card_path, tmp_path


def metadata(protocol, arm, *, grading=False, retry=False):
    item = next(row for row in schedule(protocol) if row.arm == arm)
    return {
        "experiment_id": protocol.experiment_id,
        "observation_id": identity(item.logical_request_id, 2) if retry else item.observation_id,
        "arm": None if grading else arm,
        "phase": "grading" if grading else protocol.phase,
        "attempt": 2 if retry else 1,
    }


def usage(tokens=100, cached=40, written=20):
    return {
        "input_tokens": tokens,
        "output_tokens": 10,
        "input_tokens_details": {"cached_tokens": cached, "cache_write_tokens": written},
        "output_tokens_details": {"reasoning_tokens": 7},
    }


def report(frozen, spans):
    _, protocol, inventory, card, root = frozen
    return build_report(protocol, [spans], inventory, card, root)


def model_call(directory, protocol, *, call_id="writer-call", retry=False, **attributes):
    with observation(directory, metadata(protocol, "B", retry=retry)):
        with span(
            "model.attempt",
            call_id=call_id,
            role="writer",
            deployment="writer-deployment",
            resource_id="https://fixture.example",
        ) as state:
            state.update(usage=usage())
            state.update(attributes)


def test_real_terminal_spans_hydrate_missing_model_metadata_and_price(frozen):
    protocol, _, _, _, root = frozen
    spans = root / "spans"
    model_call(spans, protocol)
    result = report(frozen, spans)
    row = result["calls"][0]
    assert row["observed_model"] is None
    assert row["model_version"] == "v1"
    assert row["binding_id"] == "writer"
    assert Decimal(row["cost_usd"]) == Decimal("0.000476")
    assert result["by_arm"]["B"]["serving"]["priced_calls"] == 1
    assert result["by_arm_model_role"][0]["role"] == "writer"
    assert result["observable_calls_fully_priced"]
    assert result["evaluation"]["complete_observable_usage_total_usd"] is None
    assert not result["cost_scope"]["total_serving_cost_claim"]
    assert result["cost_scope"]["excluded_categories"] == ["search_retrieval", "defender_ai"]
    assert not result["cost_scope"]["excluded_charges_are_zero"]


def test_retries_and_three_grader_deployments_are_retained_separately(frozen):
    protocol, _, _, _, root = frozen
    spans = root / "spans"
    model_call(spans, protocol, call_id="original")
    model_call(spans, protocol, call_id="external-retry", retry=True)
    for number in range(1, 4):
        with observation(spans, metadata(protocol, "B", grading=True)):
            with span(
                "evaluation.provider",
                call_id=f"grade-{number}",
                role="grading",
                deployment=f"grader-{number}",
                resource_id="https://fixture.example",
                usage=usage(),
                model="fixture-model-v1",
            ):
                pass
    result = report(frozen, spans)
    assert result["observable_calls"] == 5
    assert result["serving"]["observable_calls"] == 2
    assert result["evaluation"]["observable_calls"] == 3
    assert result["by_arm"]["B"]["evaluation"]["observable_calls"] == 3
    assert {row["deployment"] for row in result["calls"] if row["purpose"] == "evaluation"} == {
        "grader-1",
        "grader-2",
        "grader-3",
    }


def sdk_call(spans, protocol, *, successful):
    headers = metadata(protocol, "A")
    headers["attempt_id"] = str(headers.pop("attempt"))
    with observation(spans, headers):
        try:
            with span("baseline.sdk_create", call_id="native-parent") as state:
                with span(
                    "baseline.sdk_http_attempt",
                    call_id="native-parent",
                    transport_call_id="failed-root-http",
                    http_status=429,
                    endpoint_host="fixture.example",
                ):
                    pass
                if not successful:
                    raise RuntimeError("offline SDK retry exhaustion")
                with span(
                    "baseline.sdk_http_attempt",
                    call_id="native-parent",
                    transport_call_id="success-http",
                    http_status=200,
                    endpoint_host="fixture.example",
                ):
                    pass
                state.update(model="fixture-model-v1", usage=usage())
        except RuntimeError:
            pass


@pytest.mark.parametrize("successful", [True, False])
def test_failed_sdk_http_attempts_including_root_are_unknown_without_parent_double_count(
    frozen, successful
):
    protocol, _, _, _, root = frozen
    spans = root / "spans"
    sdk_call(spans, protocol, successful=successful)
    result = report(frozen, spans)
    ids = {row["call_id"] for row in result["calls"]}
    assert ids == ({"failed-root-http", "native-parent"} if successful else {"failed-root-http"})
    failed = next(row for row in result["calls"] if row["call_id"] == "failed-root-http")
    assert failed["deployment"] == "baseline-exact-configured-deployment"
    assert failed["cost_usd"] is None
    assert failed["normalized_usage"]["input_tokens"] is None
    assert result["unpriced_calls"] == 1
    assert result["complete_observable_usage_total_usd"] is None
    assert not result["observable_calls_fully_priced"]


@pytest.mark.parametrize(
    "change",
    [
        {"model": "wrong-model"},
        {"model_version": "wrong-version"},
        {"resource_id": "https://other.example"},
        {"region": "other-region"},
        {"sku": "DataZoneStandard"},
        {"service_tier": "priority"},
    ],
)
def test_observed_identity_conflicts_remain_unpriced_not_silently_overwritten(frozen, change):
    protocol, _, _, _, root = frozen
    spans = root / "spans"
    model_call(spans, protocol, **change)
    result = report(frozen, spans)
    assert result["calls"][0]["status"] == "unpriced_observed_inventory_mismatch"
    assert result["unpriced_calls"] == 1
    assert result["priced_subtotal_usd"] == "0"


def test_ambiguous_baseline_mapping_not_resolved_by_response_model(frozen):
    protocol, _, inventory_path, _, _ = frozen
    raw = json.loads(inventory_path.read_text())
    raw["bindings"].append(
        {
            **raw["bindings"][1],
            "binding_id": "other-baseline",
            "deployment": "other-configured-deployment",
        }
    )
    inventory = Inventory.model_validate_json(json.dumps(raw))
    row = hydrate_call(
        {
            **metadata(protocol, "A"),
            "span_name": "baseline.sdk_create",
            "call_id": "call",
            "model": "fixture-model",
            "usage": usage(),
        },
        inventory,
        protocol,
    )
    assert row["identity_status"] == "unpriced_ambiguous_inventory"


def test_protocol_unknown_observation_and_mismatched_metadata_are_unpriced(frozen):
    protocol, _, inventory_path, _, _ = frozen
    inventory = Inventory.model_validate_json(inventory_path.read_bytes())
    base = {
        **metadata(protocol, "B"),
        "span_name": "model.attempt",
        "call_id": "call",
        "deployment": "writer-deployment",
        "role": "writer",
    }
    for changes in ({"experiment_id": "other"}, {"observation_id": "unknown"}):
        assert hydrate_call({**base, **changes}, inventory, protocol)["identity_status"] == (
            "unpriced_unknown_protocol_observation"
        )
    assert hydrate_call({**base, "arm": "C"}, inventory, protocol)["identity_status"] == (
        "unpriced_protocol_metadata_mismatch"
    )


def test_context_boundary_unknown_tier_and_no_assumed_short_default(frozen):
    _, _, _, card_path, _ = frozen
    meters = list(RateCard.model_validate_json(card_path.read_bytes()).meters[:2])
    call = {
        key: getattr(meters[0], key)
        for key in ("model", "deployment", "role", "region", "sku", "resource_id", "model_version")
    }
    selected, _ = select_meter({**call, "usage": usage(100)}, meters)
    assert selected.context_tier == "short"
    selected, _ = select_meter({**call, "usage": usage(101)}, meters)
    assert selected.context_tier == "long"
    assert select_meter({**call, "usage": None}, meters)[0] is None
    unbounded = [
        item.model_copy(
            update={"input_min_tokens": None, "input_max_tokens": None, "context_evidence": None}
        )
        for item in meters
    ]
    assert select_meter({**call, "usage": usage()}, unbounded)[0] is None
    assert (
        select_meter(
            {**call, "usage": usage(101), "context_tier": "short", "context_evidence": "fixture"},
            meters,
        )[0]
        is None
    )
    assert (
        select_meter(
            {**call, "usage": usage(), "context_tier": "short", "context_evidence": "fixture"},
            unbounded,
        )[0].context_tier
        == "short"
    )
    with pytest.raises(ValueError, match="ambiguous"):
        select_meter({**call, "usage": usage()}, [meters[0], meters[0]])


def test_cache_no_write_surcharge_and_absent_vs_zero_are_not_conflated(frozen):
    _, _, _, card_path, _ = frozen
    raw = RateCard.model_validate_json(card_path.read_bytes()).meters[0].model_dump(mode="json")
    raw.update(
        input_per_million="1.75",
        cached_per_million="0.175",
        cache_write_per_million="1.75",
        output_per_million="14",
        documented_absent_zero=["cache_write_tokens"],
    )
    meter = Meter.model_validate_json(json.dumps(raw))
    recorded = {
        "input_tokens": 100,
        "output_tokens": 10,
        "input_tokens_details": {"cached_tokens": 40},
    }
    assert Decimal(price_usage(recorded, meter)["cost_usd"]) == Decimal("0.000252")
    assert price_usage(None, meter)["cost_usd"] is None
    assert price_usage({"input_tokens": 100, "output_tokens": 10}, meter)["cost_usd"] is None
    assert price_usage(usage(0, 0, 0), meter)["cost_usd"] == "0.00014"
    with pytest.raises(ValueError, match="included input"):
        Meter.model_validate_json(json.dumps({**raw, "documented_absent_zero": ["input_tokens"]}))


@pytest.mark.parametrize("drift", ["rate", "inventory", "evidence"])
def test_hash_chain_drift_is_rejected(frozen, drift):
    protocol, _, inventory, card, root = frozen
    target = {"rate": card, "inventory": inventory, "evidence": root / "fixture-evidence.json"}[
        drift
    ]
    target.write_text(target.read_text() + "\n")
    with pytest.raises(ValueError, match="drifted"):
        load_frozen(protocol, inventory, card, root)


def test_cli_runs_offline_preserves_outputs_and_refuses_duplicate_exports(frozen):
    protocol, protocol_path, inventory, card, root = frozen
    spans, output = root / "spans", root / "cost-report.json"
    model_call(spans, protocol)
    command = [
        sys.executable,
        "-m",
        "evaluation.observed_cost_report",
        "--protocol",
        str(protocol_path),
        "--spans",
        str(spans),
        "--inventory",
        str(inventory),
        "--rate-card",
        str(card),
        "--evidence-root",
        str(root),
        "--output",
        str(output),
    ]
    executed = subprocess.run(command, capture_output=True, text=True, check=False)
    assert executed.returncode == 0, executed.stderr
    assert json.loads(output.read_text())["serving"]["priced_calls"] == 1
    assert output.stat().st_mode & 0o777 == 0o600
    original = output.read_bytes()
    repeated = subprocess.run(command, capture_output=True, text=True, check=False)
    assert repeated.returncode != 0
    assert output.read_bytes() == original
    with pytest.raises(ValueError, match="twice"):
        collect([spans, spans])


def test_cli_rejects_draft_and_unfrozen_context_threshold(frozen):
    protocol, _, inventory, card, root = frozen
    raw = json.loads(card.read_text())
    raw["status"] = "draft"
    dump(card, raw)
    updated = protocol.model_copy(update={"rate_card_sha256": file_hash(card)})
    with pytest.raises(ValueError, match="draft"):
        load_frozen(updated, inventory, card, root)
    raw["status"] = "frozen"
    raw["meters"][0]["context_evidence"] = "unfrozen-evidence"
    dump(card, raw)
    updated = protocol.model_copy(update={"rate_card_sha256": file_hash(card)})
    with pytest.raises(ValueError, match="threshold evidence"):
        load_frozen(updated, inventory, card, root)


def test_private_template_uses_billed_embedding_three_graders_and_verified_baseline(tmp_path):
    root = tmp_path
    directory = root / "private-draft"
    resolution = {
        "embedding": {
            "equivalent_per_million_usd": "0.143",
            "meter_id": "bc3d015f-c87b-54c9-97bf-f872d45c490e",
        },
        "writer_short_context_global_standard": {
            key: {"effective_price_usd": price, "meter_id": f"test-short-{key}"}
            for key, price in zip(
                ("input", "cached_input", "cache_write", "output"),
                ("4", "0.4", "5", "20"),
                strict=True,
            )
        },
        "evaluator_global_standard": {
            "input": {"effective_price_usd": "1.75", "meter_id": "test-evaluator-input"},
            "output": {"effective_price_usd": "14", "meter_id": "test-evaluator-output"},
        },
        "decision": {"effective_price_usd": "0.042", "meter_id": "test-decision-input"},
    }
    fixtures = {
        RESOLUTION: {"resolved": resolution},
        MODERN: {"source": "synthetic offline test evidence"},
        FINDINGS: {"source": "synthetic offline test evidence"},
        RETAIL: {
            "items": [
                {"meterId": meter_id, "retailPrice": price}
                for meter_id, price in (
                    ("09760484-7c47-5f5a-acd7-53d7cfbff733", "8"),
                    ("fab49251-fa53-5aaa-84af-72eff9d3d934", "0.8"),
                    ("3f8397e0-567f-5dd9-ba93-2f0b15336c76", "10"),
                    ("f3f9f3cd-eb4b-56c0-a2f6-222656d84b21", "30"),
                    ("d875aea4-23b4-54b3-9313-a93fe2595a7b", "0.175"),
                )
            ]
        },
        ACTIVE_AGENT: {
            "status": "read_only_active_agent_matches_pinned_source",
            "instructions_match_pinned_source": True,
            "model_deployment": "writer-deployment",
        },
        "manifests/azure-resources.json": {
            "ai_endpoint": "https://fixture.example",
            "writer_deployment": "writer-deployment",
            "embedding_deployment": "embedding-deployment",
            "decision_deployment": "decision-deployment",
        },
    }
    for relative, value in fixtures.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        dump(path, value)
    create_templates(
        root, directory, "fixture-pilot", baseline_endpoint_host="fixture.agent.example"
    )
    inventory = Inventory.model_validate_json((directory / "inventory-template.json").read_bytes())
    card = RateCard.model_validate_json((directory / "rate-card-template.json").read_bytes())
    assert inventory.status == card.status == "draft"
    embedding = next(meter for meter in card.meters if meter.role == "embedding")
    assert embedding.input_per_million == Decimal("0.143")
    assert embedding.meter_ids == ("bc3d015f-c87b-54c9-97bf-f872d45c490e",)
    assert {item.deployment for item in inventory.bindings if item.purpose == "evaluation"} == {
        "decision-rag-evaluator",
        "decision-rag-evaluator-2",
        "decision-rag-evaluator-3",
    }
    writers = [meter for meter in card.meters if meter.role == "writer"]
    assert {meter.context_tier for meter in writers} == {"short", "long"}
    assert all(meter.input_min_tokens is meter.input_max_tokens is None for meter in writers)
    baseline = next(item for item in inventory.bindings if item.binding_id.startswith("baseline"))
    assert baseline.allow_unobserved_deployment
    assert baseline.deployment == "writer-deployment"
    assert card.evidence_sha256[ACTIVE_AGENT] == file_hash(root / ACTIVE_AGENT)
    decision = next(item for item in inventory.bindings if item.role == "decision")
    assert decision.reported_models == ("Microsoft-Decision-1", "microsoft-decision-1")
    assert all(path.stat().st_mode & 0o777 == 0o600 for path in directory.iterdir())
    with pytest.raises(FileExistsError, match="overwrite"):
        create_templates(root, directory, "fixture-pilot")


def test_unknown_context_cost_bounds_are_disclosed_but_not_added_to_subtotal(frozen):
    protocol, protocol_path, inventory, card_path, root = frozen
    raw = json.loads(card_path.read_text())
    for meter in raw["meters"]:
        meter["input_min_tokens"] = meter["input_max_tokens"] = meter["context_evidence"] = None
    dump(card_path, raw)
    updated = protocol.model_copy(update={"rate_card_sha256": file_hash(card_path)})
    dump(protocol_path, updated.model_dump(mode="json"))
    spans = root / "spans"
    model_call(spans, updated)
    result = report(frozen, spans)
    call = result["calls"][0]
    assert call["cost_usd"] is None
    assert call["status"] == "unpriced_unknown_or_conflicting_context_tier"
    bounds = call["context_tier_cost_bounds_usd"]
    assert Decimal(bounds["lower_usd"]) == Decimal("0.000476")
    assert Decimal(bounds["upper_usd"]) == Decimal("0.000852")
    assert not bounds["included_in_priced_subtotal"]
    assert result["priced_subtotal_usd"] == "0"
    assert result["complete_observable_usage_total_usd"] is None


def test_failed_sdk_create_without_transport_child_and_incomplete_send_stay_unknown(frozen):
    protocol, _, _, _, root = frozen
    spans = root / "spans"
    with observation(spans, metadata(protocol, "A")):
        try:
            with span("baseline.sdk_create", call_id="failed-root-no-transport"):
                raise RuntimeError("offline pre-transport SDK failure")
        except RuntimeError:
            pass
    with observation(spans, metadata(protocol, "A")) as recorder:
        with span("baseline.sdk_create", call_id="incomplete-native"):
            with span(
                "baseline.sdk_http_attempt",
                call_id="incomplete-native",
                transport_call_id="incomplete-http",
                endpoint_host="fixture.example",
            ):
                pass
    boundaries = [json.loads(line) for line in recorder.path.read_text().splitlines()]
    # Simulate exporter termination after durable starts, before any provider result.
    recorder.path.write_text(
        "\n".join(json.dumps(event) for event in boundaries if event["event"] == "span_start")
        + "\n"
    )
    result = report(frozen, spans)
    assert {row["call_id"] for row in result["calls"]} == {
        "failed-root-no-transport",
        "incomplete-http",
    }
    assert result["unpriced_calls"] == 2
    assert result["incomplete_traces"] == 1


def test_native_retries_and_extra_successful_transports_are_not_dropped_or_double_counted(frozen):
    protocol, _, _, _, root = frozen
    spans = root / "spans"
    with observation(spans, metadata(protocol, "A")):
        try:
            with span("baseline.sdk_create", call_id="native-1", native_attempt=1):
                with span(
                    "baseline.sdk_http_attempt",
                    call_id="native-1",
                    transport_call_id="native-1-http",
                    http_status=429,
                ):
                    pass
                raise RuntimeError("offline native retry")
        except RuntimeError:
            pass
        with span(
            "baseline.sdk_create",
            call_id="native-2",
            native_attempt=2,
            model="fixture-model",
            usage=usage(),
        ):
            for attempt in (1, 2):
                with span(
                    "baseline.sdk_http_attempt",
                    call_id="native-2",
                    transport_call_id=f"native-2-http-{attempt}",
                    http_status=200,
                ):
                    pass
    result = report(frozen, spans)
    assert {row["call_id"] for row in result["calls"]} == {
        "native-1-http",
        "native-2-http-1",
        "native-2",
    }
    assert result["unpriced_calls"] == 2
    priced = next(row for row in result["calls"] if row["call_id"] == "native-2")
    assert priced["represented_transport_call_id"] == "native-2-http-2"
    assert Decimal(result["priced_subtotal_usd"]) == Decimal("0.000476")
