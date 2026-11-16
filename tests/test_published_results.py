"""The published benchmark results must be self-consistent and match the README."""

import json
import re
from decimal import ROUND_HALF_UP, Decimal
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "docs/benchmark-results"
FINAL = RESULTS / "iso7-benchmark-20261114"
ARMS = "ABCDEFG"
CUSTOM_ARMS = "BCDEFG"
IDENTIFIERS = {
    "GUID": re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b", re.I),
    "email address": re.compile(r"[\w.+-]+@[\w-]+\.[\w.-]+"),
    "IPv4 address": re.compile(r"\b(?!127\.|0\.0\.0\.0\b)(?:\d{1,3}\.){3}\d{1,3}\b"),
    "Azure hostname": re.compile(r"\.(?:azure|windows)\.(?:com|net)\b|\.azure-api\.net"),
    "subscription or resource group path": re.compile(r"subscriptions/|resourceGroups", re.I),
    "home directory": re.compile(r"/home/\w+|C:\\\\Users"),
}


@pytest.fixture(scope="module")
def summary() -> dict:
    return json.loads((FINAL / "summary.json").read_text())


@pytest.fixture(scope="module")
def quality() -> dict:
    return json.loads((FINAL / "quality.json").read_text())


@pytest.fixture(scope="module")
def readme() -> str:
    return (ROOT / "README.md").read_text(encoding="utf-8")


def rounded(value: float, places: int) -> Decimal:
    return Decimal(repr(value)).quantize(Decimal(1).scaleb(-places), rounding=ROUND_HALF_UP)


def test_pass_rates_follow_from_pass_counts(summary, quality):
    for arm in ARMS:
        config = summary["configs"][arm]
        counts = config["quality"]["strict_pass"]
        rates = config["quality"]["pass_rate_percent"]
        requests = config["requests"]
        assert rates["judge_1"] == round(100 * counts["judge_1"] / requests, 1)
        assert rates["judge_2"] == round(100 * counts["judge_2"] / requests, 1)
        pooled = 100 * (counts["judge_1"] + counts["judge_2"]) / (2 * requests)
        assert rates["average"] == round(pooled, 1)
        raw = quality["arms"][arm]
        assert (raw["j1_pass"], raw["j2_pass"]) == (counts["judge_1"], counts["judge_2"])


def test_latency_parts_add_up(summary):
    for arm in ARMS:
        latency = summary["configs"][arm]["latency_seconds"]
        parts = (
            latency["mean_excluding_client_rate_limit_wait"]
            + latency["mean_client_rate_limit_wait"]
        )
        assert latency["mean"] == pytest.approx(parts, abs=0.02), arm
    assert summary["configs"]["A"]["latency_seconds"]["mean_client_rate_limit_wait"] == 0


def test_costs_follow_from_usage_and_billed_rates(summary):
    rates = summary["pricing_usd_per_1m_tokens"]
    for arm in CUSTOM_ARMS:
        config = summary["configs"][arm]
        usage, cost = config["large_model"], config["cost_usd"]
        assert (
            usage["plain_input_tokens"] + usage["cache_read_tokens"] + usage["cache_write_tokens"]
            == usage["input_tokens"]
        ), arm
        large = (
            usage["plain_input_tokens"] * rates["large_model_input"]
            + usage["cache_read_tokens"] * rates["large_model_cache_read"]
            + usage["cache_write_tokens"] * rates["large_model_cache_write"]
            + usage["output_tokens"] * rates["large_model_output"]
        ) / 1e6
        decision = config.get("decision_model", {}).get("tokens", 0) * rates["decision_model"] / 1e6
        assert cost["large_model"] == pytest.approx(large, abs=1e-4), arm
        assert cost["decision_model"] == pytest.approx(decision, abs=1e-4), arm
        parts = cost["large_model"] + cost["decision_model"] + cost["embedding"]
        assert cost["total"] == pytest.approx(parts, abs=3e-4), arm
        assert cost["embedding"] < 0.001, arm


def test_config_a_cost_is_the_bill_minus_the_other_configs(summary):
    reconciliation = summary["cost_reconciliation"]
    others = sum(summary["configs"][arm]["cost_usd"]["large_model"] for arm in CUSTOM_ARMS)
    assert reconciliation["recorded_b_to_g_large_model_usd"] == pytest.approx(others, abs=2e-3)
    billed_minus_recorded = (
        reconciliation["billed_large_model_total_usd"]
        - reconciliation["recorded_b_to_g_large_model_usd"]
    )
    cost_a = summary["configs"]["A"]["cost_usd"]["total"]
    assert cost_a == reconciliation["a_large_model_cost_usd"]
    assert cost_a == pytest.approx(billed_minus_recorded, abs=5e-3)
    for key, implied in reconciliation["implied_a_million_tokens"].items():
        billed = reconciliation["billed_large_model"][key]["million_tokens"]
        recorded = reconciliation["recorded_b_to_g_million_tokens"][key]
        assert implied == pytest.approx(billed - recorded, abs=2e-4), key
    implied = reconciliation["implied_a_million_tokens"]
    assert 1e6 * sum(implied[key] for key in ("plain", "read", "write")) == pytest.approx(
        summary["configs"]["A"]["large_model"]["input_tokens"], abs=500
    )


def test_azure_monitor_matches_recorded_usage(summary):
    monitor = summary["azure_monitor"]["configs"]
    for arm in CUSTOM_ARMS:
        config = summary["configs"][arm]
        large = monitor[arm]["deployments"]["gpt-5.6-sol"]
        assert large["requests"] == config["large_model"]["calls"], arm
        assert large["input_tokens"] == config["large_model"]["input_tokens"], arm
        assert large["output_tokens"] == config["large_model"]["output_tokens"], arm
        if "decision_model" in config:
            decision = monitor[arm]["deployments"]["decision1-deployment"]
            assert decision["requests"] == config["decision_model"]["calls"], arm
            tokens = decision["input_tokens"] + decision["output_tokens"]
            assert tokens == config["decision_model"]["tokens"], arm
    baseline = summary["configs"]["A"]["large_model"]
    observed = monitor["A"]["deployments"]["gpt-5.6-sol"]
    assert (observed["requests"], observed["input_tokens"]) == (
        baseline["calls"],
        baseline["input_tokens"],
    )
    assert baseline["reported_by_sdk"]["calls"] < baseline["calls"]


def test_config_f_latency_breakdown_adds_up(summary):
    breakdown = summary["config_f_latency_breakdown"]
    assert breakdown["mean_seconds"] == summary["configs"]["F"]["latency_seconds"]["mean"]
    stages = breakdown["stages"]
    assert sum(stage["mean_seconds"] for stage in stages) == pytest.approx(
        breakdown["mean_seconds"], abs=0.05
    )
    assert sum(stage["share_of_mean_percent"] for stage in stages) == pytest.approx(100, abs=0.5)
    for stage in stages:
        parts = stage["execution_seconds"] + stage["client_rate_limit_wait_seconds"]
        assert stage["mean_seconds"] == pytest.approx(parts, abs=0.015), stage["stage"]


def test_readme_table_matches_results(summary, readme):
    rows = re.findall(
        r"^\| ([A-G]) \| [^|]+ \| ([\d.]+)% \(([^)]*)\) \| \$([\d.]+) \| ([\d.]+) s \|$",
        readme,
        re.M,
    )
    assert [row[0] for row in rows] == list(ARMS)
    for arm, average, spread, cost, latency in rows:
        config = summary["configs"][arm]
        rates = config["quality"]["pass_rate_percent"]
        assert float(average) == rates["average"], arm
        if "both runs" in spread:
            assert rates["judge_1"] == rates["judge_2"] == rates["average"], arm
        else:
            low, high = (float(number) for number in re.findall(r"[\d.]+", spread))
            assert (low, high) == (
                min(rates["judge_1"], rates["judge_2"]),
                max(rates["judge_1"], rates["judge_2"]),
            ), arm
        assert Decimal(cost) == rounded(config["cost_usd"]["total"], 2), arm
        assert Decimal(latency) == rounded(config["latency_seconds"]["mean"], 1), arm


def test_readme_statements_match_results(summary, readme):
    configs, run = summary["configs"], summary["run"]
    cost = {arm: configs[arm]["cost_usd"]["total"] for arm in ARMS}
    latency = {arm: configs[arm]["latency_seconds"] for arm in ARMS}
    failed = {arm: configs[arm]["failed_requests"] for arm in ARMS}
    execution = run["execution"]
    expected = (
        f"{run['requests_per_config']} requests per configuration and "
        f"{run['total_requests']:,} in total",
        f"at {round(100 * (1 - cost['F'] / cost['A']))}% lower",
        f"costs {round(cost['G'] / cost['F'], 1)} times as much",
        f"takes about {round(latency['G']['mean'] / latency['F']['mean'])} times as long",
        f"(6 in A, {failed['B']} in B, {failed['C']} in C, none in D–G)",
        f"({configs['A']['large_model']['calls']} observed in Azure Monitor",
        f"against {configs['A']['large_model']['reported_by_sdk']['calls']} reported)",
        f"({latency['F']['mean_client_rate_limit_wait']:.1f} s on average for F, "
        f"{latency['G']['mean_client_rate_limit_wait']:.0f} s for G)",
        f"F\n  averages {latency['F']['mean_excluding_client_rate_limit_wait']:.1f} s",
        f"{execution['idle_minutes_between_configs']}-minute idle gap",
        f"at least {execution['min_minutes_between_a_questions_two_passes']} minutes",
    )
    flat = readme.replace("\n  ", " ")
    for statement in expected:
        assert statement.replace("\n  ", " ") in flat, statement
    assert failed["A"] == 6 and failed["D"] == failed["E"] == failed["F"] == failed["G"] == 0


def test_readme_comparisons_hold_in_the_paired_analysis(quality):
    paired = quality["paired"]

    def interval(judge: str, pair: str) -> tuple[float, float]:
        low, high = paired[f"{judge}:{pair}"]["ci95_points"]
        return low, high

    for judge in ("j1", "j2"):
        for pair in ("F-A", "F-B"):
            assert interval(judge, pair)[0] > 0, (judge, pair)
        for pair in ("F-G", "C-B", "D-B", "E-B"):
            low, high = interval(judge, pair)
            assert low < 0 < high, (judge, pair)


def test_older_results_are_kept_apart_from_the_final_ones():
    assert FINAL.is_dir() and (RESULTS / "earlier-runs").is_dir()
    assert all(path.is_dir() for path in RESULTS.iterdir())


def test_published_results_contain_no_identifiers():
    findings = []
    for path in sorted(RESULTS.rglob("*")):
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        findings += [
            f"{path.relative_to(ROOT)}: {label}"
            for label, pattern in IDENTIFIERS.items()
            if pattern.search(text)
        ]
    assert not findings, findings
