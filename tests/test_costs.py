import pytest

from evaluation.costs import retail_rate, summarize, writer_cost


def prices():
    return {
        "source": "fixture",
        "retrieved_date": "2026-11-04",
        "items": [
            {
                "meterName": name,
                "retailPrice": price,
                "type": "Consumption",
                "currencyCode": "USD",
                "unitOfMeasure": "1M",
            }
            for name, price in (
                ("5.6 sol ShortCo Inp Std Gl 1M Tokens", 4),
                ("5.6 sol ShortCo Cd Inp Std Gl 1M Tokens", 0.4),
                ("5.6 sol ShortCo Cd Wr Std Gl 1M Tokens", 5),
                ("5.6 sol ShortCo Opt Std Gl 1M Tokens", 20),
                ("Microsoft Decision 1 1M Tokens", 0.042),
            )
        ],
    }


def test_cache_write_is_not_double_charged_as_ordinary_input():
    rates = {"input": 4, "cached": 0.4, "cache_write": 5, "output": 20}
    assert writer_cost(1_000_000, 1_000_000, 200_000, 300_000, rates) == 23.58
    with pytest.raises(ValueError):
        writer_cost(10, 0, 5, 6, rates)
    with pytest.raises(ValueError):
        writer_cost(-1, 0, 0, 0, rates)


def test_missing_or_conflicting_retail_rates_cannot_be_zero():
    snapshot = prices()
    with pytest.raises(ValueError, match="missing or ambiguous"):
        retail_rate(snapshot, "missing")
    snapshot["items"].append({**snapshot["items"][0], "retailPrice": 9})
    with pytest.raises(ValueError, match="missing or ambiguous"):
        retail_rate(snapshot, snapshot["items"][0]["meterName"])


def test_partial_cost_keeps_unknown_usage_and_baseline_cache_scenarios_distinct():
    evidence = {
        "baseline_events": [{"usage": {"in": 1_000_000, "out": 1_000}, "kb_calls": 1}],
        "F_calls_including_retries": [
            {
                "model": "gpt-5.6-sol-2026-07-09",
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000,
                    "prompt_tokens_details": {"cached_tokens": 0, "cache_write_tokens": 1_000_000},
                },
            },
            {
                "model": "microsoft-decision-1",
                "usage": {"input_tokens": 1_000_000, "output_tokens": 10},
            },
            {"category": "decision", "state": "failed"},
        ],
    }
    report = summarize(evidence, prices())
    assert report["A"]["ordinary_input_scenario_usd"] == 4.02
    assert report["A"]["all_cache_write_scenario_usd"] == 5.02
    assert report["F"]["reported_writer_plus_decision_subtotal_usd"] == pytest.approx(5.062)
    assert report["F"]["unknown_usage_calls"] == 1
    assert "inconclusive" in report["conclusion"]
