from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from evaluation.run import ROOT, persist


def writer_cost(
    input_tokens: int, output_tokens: int, cached: int, written: int, rates: dict
) -> float:
    if any(type(x) is not int or x < 0 for x in (input_tokens, output_tokens, cached, written)):
        raise ValueError("token counts must be nonnegative integers")
    if cached + written > input_tokens:
        raise ValueError("cache categories exceed input tokens")
    return float(
        (
            Decimal(input_tokens - cached - written) * Decimal(str(rates["input"]))
            + Decimal(cached) * Decimal(str(rates["cached"]))
            + Decimal(written) * Decimal(str(rates["cache_write"]))
            + Decimal(output_tokens) * Decimal(str(rates["output"]))
        )
        / Decimal(1_000_000)
    )


def retail_rate(snapshot: dict, name: str) -> float:
    matches = [
        item
        for item in snapshot["items"]
        if item["meterName"] == name
        and item["type"] == "Consumption"
        and item["currencyCode"] == "USD"
        and item["unitOfMeasure"] == "1M"
    ]
    prices = {Decimal(str(item["retailPrice"])) for item in matches}
    if len(prices) != 1:
        raise ValueError(f"missing or ambiguous USD retail meter: {name}")
    return float(prices.pop())


def collect(source: Path, retry: Path, audit_path: Path) -> dict:
    evidence_path = retry / "cost-evidence.json"
    if evidence_path.exists():
        return json.loads(evidence_path.read_text())
    paths = sorted((source / "responses").glob("*-A.json"))
    if len(paths) != 72:
        raise ValueError("expected 72 baseline captures")
    start = min(
        path.stat().st_mtime - json.loads(path.read_text())["application_seconds"] for path in paths
    )
    end = max(path.stat().st_mtime for path in paths)
    audit_raw = audit_path.read_bytes()
    events = [
        item
        for line in audit_raw.decode().splitlines()
        if (item := json.loads(line)).get("event") == "chat"
        and start - 5
        <= datetime.fromisoformat(item["ts"].replace("Z", "+00:00")).timestamp()
        <= end + 5
    ]
    if len(events) != 72 or any(
        item.get("error") or item.get("kb") != "dossier-kb" or item.get("usage") is None
        for item in events
    ):
        raise ValueError("baseline audit-window attribution is incomplete or ambiguous")
    calls = []
    trace_hashes = {}
    seen_ledgers = set()
    for directory in (source, retry):
        for path in sorted((directory / "responses").glob("*-F.json")):
            response = json.loads(path.read_text())
            if response.get("reused_from"):
                continue
            payload = json.loads(response["raw_body"])
            trace_id = payload["trace_id"]
            if (
                not isinstance(trace_id, str)
                or len(trace_id) != 32
                or any(ch not in "0123456789abcdef" for ch in trace_id)
            ):
                raise ValueError("invalid private trace identity")
            trace_path = ROOT / f"data/review/traces/{trace_id}.json"
            raw = trace_path.read_bytes()
            trace = json.loads(raw)
            if trace["trace_id"] != trace_id:
                raise ValueError("trace identity mismatch")
            trace_hashes[str(trace_path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
            for call in trace["calls"]:
                ledger = call["ledger_id"]
                if ledger in seen_ledgers:
                    raise ValueError("duplicate runtime ledger reservation")
                seen_ledgers.add(ledger)
                calls.append({**call, "response": f"{directory.name}/{path.name}"})
    evidence = {
        "baseline_audit_attribution": "72 chat events in first-pass response file-mtime window "
        "with 5s tolerance; aggregate observational attribution, not request-ID correlation.",
        "baseline_window_epoch": [start, end],
        "baseline_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "baseline_events": events,
        "F_calls_including_retries": calls,
        "trace_sha256": trace_hashes,
    }
    persist(evidence_path, evidence)
    return evidence


def summarize(evidence: dict, prices: dict) -> dict:
    rates = {
        key: retail_rate(prices, f"5.6 sol ShortCo {meter} Std Gl 1M Tokens")
        for key, meter in (
            ("input", "Inp"),
            ("cached", "Cd Inp"),
            ("cache_write", "Cd Wr"),
            ("output", "Opt"),
        )
    }
    decision_rate = retail_rate(prices, "Microsoft Decision 1 1M Tokens")
    baseline = evidence["baseline_events"]
    a_in = sum(item["usage"]["in"] for item in baseline)
    a_out = sum(item["usage"]["out"] for item in baseline)
    models = defaultdict(Counter)
    unknown = 0
    for call in evidence["F_calls_including_retries"]:
        usage = call.get("usage")
        if usage is None:
            unknown += 1
            continue
        model = call["model"]
        values = models[model]
        values["calls"] += 1
        values["input_tokens"] += usage.get("input_tokens", usage.get("prompt_tokens", 0))
        values["output_tokens"] += usage.get("output_tokens", usage.get("completion_tokens", 0))
        details = usage.get("input_tokens_details", usage.get("prompt_tokens_details", {}))
        values["cached_tokens"] += details.get("cached_tokens", 0)
        values["cache_write_tokens"] += details.get("cache_write_tokens", 0)
    writer = models["gpt-5.6-sol-2026-07-09"]
    f_writer = writer_cost(
        writer["input_tokens"],
        writer["output_tokens"],
        writer["cached_tokens"],
        writer["cache_write_tokens"],
        rates,
    )
    f_decision = float(
        Decimal(models["microsoft-decision-1"]["input_tokens"])
        * Decimal(str(decision_rate))
        / Decimal(1_000_000)
    )
    return {
        "status": "partial_retail_model_cost_estimate_not_invoice_or_total_cost",
        "currency": "USD",
        "questions_per_arm": 72,
        "price_source": prices["source"],
        "price_retrieved_date": prices["retrieved_date"],
        "rates_per_million": {**rates, "decision_input": decision_rate},
        "A": {
            "reported_writer_input_tokens": a_in,
            "reported_writer_output_tokens": a_out,
            "reported_kb_calls": sum(item["kb_calls"] for item in baseline),
            "ordinary_input_scenario_usd": writer_cost(a_in, a_out, 0, 0, rates),
            "all_cache_hit_scenario_usd": writer_cost(a_in, a_out, a_in, 0, rates),
            "all_cache_write_scenario_usd": writer_cost(a_in, a_out, 0, a_in, rates),
            "cache_breakdown": "not reported by reference audit",
        },
        "F": {
            "reported_usage_by_model": dict(models),
            "writer_estimate_usd": f_writer,
            "decision_input_estimate_usd": f_decision,
            "reported_writer_plus_decision_subtotal_usd": f_writer + f_decision,
            "subtotal_per_question_usd": (f_writer + f_decision) / 72,
            "unknown_usage_calls": unknown,
        },
        "assumptions": [
            "GPT-5.6-sol Global Standard short-context rates; not account-negotiated prices.",
            "Cache-hit/write counters are treated as disjoint subsets of prompt tokens.",
            "A's cache sensitivity is not a confidence interval or total-cost bound.",
            "Decision estimate prices reported input tokens; output assumed unbilled. "
            "Confirm this against actual invoice meters.",
            "Retries included for F; baseline native retries remain unobserved.",
        ],
        "unpriced_or_excluded": [
            "F embeddings: tokens measured, exact matching deployment retail meter unverified.",
            "Failed calls with unknown usage, hidden baseline native retries and tool charges.",
            "Baseline Knowledge Base agentic reasoning/planning/answer-synthesis and embeddings.",
            "Search capacity, semantic ranking, storage, hosting and network transfer.",
            "One-time ingestion, Document Intelligence, indexing and cached vision descriptions.",
            "Independent grading and human review: evaluation overhead, not serving cost.",
        ],
        "conclusion": "Total cost ranking is inconclusive. The Decision-1 reported-input "
        "subtotal is small, but writer context/cache charges dominate observed F model cost. "
        "Missing baseline cache/tool billing prevents a fair total-cost winner.",
    }


def render(report: dict) -> str:
    a, f = report["A"], report["F"]
    return "\n".join(
        [
            "# Partial cost comparison (USD retail estimate, not total cost)",
            "",
            "| Component, 72 questions | Original A | Decision-assisted F |",
            "|---|---:|---:|",
            f"| Writer input tokens | {a['reported_writer_input_tokens']:,} | "
            f"{f['reported_usage_by_model']['gpt-5.6-sol-2026-07-09']['input_tokens']:,} |",
            f"| Writer output tokens | {a['reported_writer_output_tokens']:,} | "
            f"{f['reported_usage_by_model']['gpt-5.6-sol-2026-07-09']['output_tokens']:,} |",
            f"| Writer estimate | ${a['ordinary_input_scenario_usd']:.2f} "
            "ordinary-input scenario | "
            f"${f['writer_estimate_usd']:.2f} with reported cache writes |",
            f"| Decision-1 reported-input estimate | Not used | "
            f"${f['decision_input_estimate_usd']:.3f} |",
            f"| Observed writer + Decision subtotal | Cache split unknown | "
            f"${f['reported_writer_plus_decision_subtotal_usd']:.2f} |",
            "",
            f"A's writer-only cache sensitivity is ${a['all_cache_hit_scenario_usd']:.2f} "
            f"(all hits) to ${a['all_cache_write_scenario_usd']:.2f} (all writes). "
            "These extremes are scenarios, not measured charges or bounds on total spend. "
            f"F's priced subtotal averages ${f['subtotal_per_question_usd']:.4f}/question, "
            f"excluding {f['unknown_usage_calls']} unknown-usage calls and the items below.",
            "",
            "**Total-cost ranking remains inconclusive.**",
            report["conclusion"],
            "",
            "## Rate and attribution assumptions",
            "",
            f"Official [Azure retail prices]({report['price_source']}), "
            f"retrieved {report['price_retrieved_date']}; snapshot retained alongside this report.",
            "Per million: writer input $4, cached input $0.40, cache write $5, output $20; "
            "Decision input $0.042. Pricing scope is Global Standard short context, USD.",
            "Baseline usage recovered from 72 audit chat events matched to the capture-time "
            "window, not per-request IDs. F usage comes from unique ledger IDs in private "
            "request traces, including first attempts and retries. Reused responses excluded.",
            *[f"- {item}" for item in report["assumptions"]],
            "",
            "## Unpriced or excluded",
            "",
            *[f"- {item}" for item in report["unpriced_or_excluded"]],
            "",
            "Azure documents separate Search and agentic retrieval charges: "
            "https://learn.microsoft.com/en-us/azure/search/search-sku-manage-costs",
            "",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Recover partial cost evidence without inference.")
    parser.add_argument("--baseline-audit", required=True, type=Path)
    args = parser.parse_args()
    source = ROOT / "data/comparison/phase3-full-comparison"
    retry = source.with_name(source.name + "-retry1")
    evidence = collect(source, retry, args.baseline_audit)
    prices = json.loads((retry / "retail-price-snapshot.json").read_text())
    report = summarize(evidence, prices)
    report["evidence_sha256"] = hashlib.sha256(
        (retry / "cost-evidence.json").read_bytes()
    ).hexdigest()
    report["price_snapshot_sha256"] = hashlib.sha256(
        (retry / "retail-price-snapshot.json").read_bytes()
    ).hexdigest()
    persist(retry / "cost-report.json", report)
    (retry / "cost-report.md").write_text(render(report))
    print(render(report))


if __name__ == "__main__":
    main()
