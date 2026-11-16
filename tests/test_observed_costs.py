from decimal import Decimal

import pytest

from evaluation.experiment import CostScope
from evaluation.observed_costs import Meter, normalize_usage, price_usage, reconcile_calls


def meter(cache=True):
    return Meter(
        model="test-model",
        deployment="test-deployment",
        role="writer",
        region="test-region",
        sku="test-sku",
        source="offline-rate-fixture",
        effective_date="test-date",
        meter_ids=("test-meter",),
        price_kind="official_retail",
        input_per_million=Decimal("4"),
        output_per_million=Decimal("20"),
        cached_per_million=Decimal("0.4") if cache else None,
        cache_write_per_million=Decimal("5") if cache else None,
        cache_semantics="disjoint_input_subsets" if cache else "not_applicable",
        semantics_evidence="offline-test-only",
    )


def test_absent_cache_usage_is_unknown_not_free_or_zero():
    result = price_usage({"prompt_tokens": 100, "completion_tokens": 10}, meter())
    assert result["cost_usd"] is None
    assert result["missing_counters"] == ["cached_tokens", "cache_write_tokens"]
    assert normalize_usage(None) == dict.fromkeys(
        ("input_tokens", "output_tokens", "cached_tokens", "cache_write_tokens")
    )


def test_cache_subsets_and_reasoning_output_not_double_counted():
    result = price_usage(
        {
            "input_tokens": 100,
            "output_tokens": 10,
            "input_tokens_details": {"cached_tokens": 40, "cache_write_tokens": 20},
            "output_tokens_details": {"reasoning_tokens": 7},
        },
        meter(),
    )
    assert Decimal(result["cost_usd"]) == Decimal("0.000476")


def test_root_cache_write_counter_is_preserved_without_overriding_explicit_details():
    usage = {
        "input_tokens": 100,
        "output_tokens": 10,
        "cache_creation_input_tokens": 20,
        "input_tokens_details": {"cached_tokens": 40},
    }
    assert normalize_usage(usage)["cache_write_tokens"] == 20
    usage["input_tokens_details"]["cache_write_tokens"] = 0
    assert normalize_usage(usage)["cache_write_tokens"] == 0


def test_invalid_cache_overlap_and_counter_types_rejected():
    with pytest.raises(ValueError, match="subsets"):
        price_usage(
            {
                "input_tokens": 10,
                "output_tokens": 1,
                "input_tokens_details": {"cached_tokens": 8, "cache_write_tokens": 8},
            },
            meter(),
        )
    with pytest.raises(ValueError, match="nonnegative"):
        normalize_usage({"input_tokens": True})


def test_missing_entire_usage_cannot_be_zeroed_by_documented_absence_rule():
    m = meter().model_copy(
        update={
            "documented_absent_zero": (
                "input_tokens",
                "output_tokens",
                "cached_tokens",
                "cache_write_tokens",
            )
        }
    )
    assert price_usage(None, m)["cost_usd"] is None


def test_exact_rate_match_and_duplicate_call_accounting():
    call = {
        "call_id": "one",
        "model": "test-model",
        "deployment": "test-deployment",
        "role": "writer",
        "region": "test-region",
        "sku": "test-sku",
        "usage": {"input_tokens": 100, "output_tokens": 10},
    }
    result = reconcile_calls([call], [meter(False)])
    assert Decimal(result["priced_subtotal_usd"]) == Decimal("0.0006")
    assert result["unpriced_calls"] == 0
    with pytest.raises(ValueError, match="duplicate"):
        reconcile_calls([call, call], [meter(False)])
    assert reconcile_calls([{**call, "region": "other"}], [meter(False)])["unpriced_calls"] == 1


def test_user_exclusions_are_disclosed_without_erasing_unknown_included_usage():
    scope = CostScope(
        excluded_categories=("search_retrieval", "defender_ai"),
        exclusion_reference="User: ignore these two for both implementations",
    )
    report = reconcile_calls([{"call_id": "unpriced", "usage": None}], [], cost_scope=scope)
    assert report["unpriced_calls"] == 1
    assert report["calls"][0]["cost_usd"] is None
    assert report["cost_scope"]["excluded_categories"] == ["search_retrieval", "defender_ai"]
    assert report["cost_scope"]["applies_to_arms"] == list("ABCDEFG")
    assert not report["cost_scope"]["total_serving_cost_claim"]
