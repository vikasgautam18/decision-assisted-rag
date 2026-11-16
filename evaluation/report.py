from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path

from evaluation.dataset import load_cases
from evaluation.run import ROOT, persist


def summarize(directory: Path) -> dict:
    manifest = json.loads((directory / "manifest.json").read_text())
    selected = manifest["frozen"]["protocol"].get("selected_case_ids")
    cases = [case for case in load_cases(directory) if selected is None or case.id in selected]
    rows, review = [], []
    for case in cases:
        response_path = directory / "responses" / f"{case.id}.json"
        judge_path = directory / "judgments" / f"{case.id}.json"
        captured = json.loads(response_path.read_text()) if response_path.exists() else None
        grading = json.loads(judge_path.read_text()) if judge_path.exists() else None
        judgment = (grading or {}).get("judgment")
        expected = (
            "conflict" if case.expected_behavior == "flag_conflict" else case.expected_behavior
        )
        reasons = []
        if captured is None:
            reasons.append("application_not_run")
        elif captured["response"].get("status") != "completed":
            reasons.append("application_error")
        elif not captured["deterministic"]["pass"]:
            reasons.append("deterministic_failure")
        if judgment is None:
            reasons.append("independent_grading_missing_or_error")
        else:
            if judgment["behavior"] != expected:
                reasons.append("independently_classified_behavior_mismatch")
            for key in ("correct", "complete", "claims_supported", "citations_supported"):
                if not judgment[key]:
                    reasons.append(key)
            if judgment["gold_issue"]:
                reasons.append("gold_label_needs_adjudication")
            if judgment["uncertain"]:
                reasons.append("judge_uncertain")
        if captured and captured["trace_checks"].get("calculation_error"):
            reasons.append("trace_calculation_failure")
        if captured and not captured["trace_checks"].get("available"):
            reasons.append("trace_checks_unavailable")
        passed = not reasons
        row = {
            "id": case.id,
            "category": case.category,
            "expected_behavior": case.expected_behavior,
            "application_behavior": (captured or {}).get("response", {}).get("behavior"),
            "independent_behavior": judgment["behavior"] if judgment else None,
            "deterministic_pass": captured["deterministic"]["pass"] if captured else False,
            "strict_provisional_pass": passed,
            "application_error": bool(
                captured and captured["response"].get("status") != "completed"
            ),
            "application_seconds": captured["application_seconds"] if captured else None,
            "grading_status": (grading or {}).get("status", "missing"),
            "grading_seconds": (grading or {}).get("grading_seconds"),
            "reason_codes": reasons,
            "judge_reason": judgment["reason"] if judgment else None,
            "trace_id": (captured or {}).get("response", {}).get("trace_id"),
            "trace_checks": (captured or {}).get("trace_checks", {}),
        }
        rows.append(row)
        # Stable sample is reproducible and independent of outcome ordering.
        sample = (
            int(hashlib.sha256(f"phase2-audit:{case.id}".encode()).hexdigest()[:8], 16) % 10 == 0
        )
        if reasons or sample:
            review.append(
                {
                    "id": case.id,
                    "reason_codes": reasons or ["agreement_audit_sample"],
                    "question": case.question,
                    "original_gold_answer": case.gold_answer,
                    "gold_sources": [item.model_dump() for item in case.gold_sources],
                    "answer": (captured or {}).get("response", {}).get("answer"),
                    "independent_judgment": judgment,
                    "human_status": "pending",
                    "human_decision": None,
                    "human_notes": None,
                }
            )
    by_category = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
    latencies = [
        row["application_seconds"] for row in rows if row["application_seconds"] is not None
    ]
    ordered = sorted(latencies)
    usage_reported = sum(row["trace_checks"].get("usage_reported_calls", 0) for row in rows)
    usage_unknown = sum(row["trace_checks"].get("usage_unknown_calls", 0) for row in rows)
    report = {
        "status": "provisional_independent_scoring_human_adjudication_pending",
        "application_sha256": manifest["frozen"]["application_sha256"],
        "protocol": manifest["frozen"]["protocol"],
        "audit": manifest["frozen"]["audit"],
        "cases_expected": len(cases),
        "cases_captured": sum(
            (directory / "responses" / f"{case.id}.json").exists() for case in cases
        ),
        "strict_provisional_passes": sum(row["strict_provisional_pass"] for row in rows),
        "deterministic_passes": sum(row["deterministic_pass"] for row in rows),
        "application_errors": sum(row["application_error"] for row in rows),
        "grading_completed": sum(row["grading_status"] == "completed" for row in rows),
        "grading_errors_or_missing": sum(
            row["grading_status"] in ("missing", "error") for row in rows
        ),
        "latency_seconds": {
            "mean": statistics.mean(latencies) if latencies else None,
            "median": statistics.median(latencies) if latencies else None,
            "p95_nearest_rank": ordered[max(0, (95 * len(ordered) + 99) // 100 - 1)]
            if ordered
            else None,
            "includes_application_failures": True,
            "excludes_grading": True,
        },
        "runtime_usage": {
            "reported_calls": usage_reported,
            "unknown_calls": usage_unknown,
            "monetary_cost": "unknown; not zero",
        },
        "by_category": {
            name: {
                "n": len(items),
                "strict_provisional_passes": sum(row["strict_provisional_pass"] for row in items),
                "deterministic_passes": sum(row["deterministic_pass"] for row in items),
            }
            for name, items in sorted(by_category.items())
        },
        "human_review_count": len(review),
        "limitations": [
            "Golds and runtime development cases were previously exposed; no blind held-out claim.",
            "One run per question; correlated fact families, no repeat-stability or load claim.",
            "Model grading is provisional until human disagreement/label adjudication.",
            "Original labels untouched. Gold issues need original PDF/image review.",
            "Runtime cutoffs are not calibrated by this single run.",
            "No baseline calls, phase 3 comparison, superiority or complete cost claim.",
        ],
        "rows": rows,
    }
    queue_path = directory / "human-review-queue.json"
    if queue_path.exists():
        existing = {item["id"]: item for item in json.loads(queue_path.read_text())["items"]}
        for item in review:
            for key in ("human_status", "human_decision", "human_notes"):
                item[key] = existing.get(item["id"], {}).get(key, item[key])
        selected = {item["id"] for item in review}
        review.extend(item for key, item in existing.items() if key not in selected)
        report["human_review_count"] = len(review)
    persist(queue_path, {"status": "pending", "items": review})
    persist(directory / "report.json", report)
    lines = [
        "# Phase 2 standalone regression: provisional results",
        "",
        f"Cases: {report['cases_captured']}/{len(cases)} captured; deterministic checks "
        f"{report['deterministic_passes']}/{len(cases)}; strict independently scored "
        f"{report['strict_provisional_passes']}/{len(cases)}.",
        "",
        f"Application errors: {report['application_errors']}; independent judgments completed: "
        f"{report['grading_completed']}; missing/failed grading: "
        f"{report['grading_errors_or_missing']}.",
        "",
        "**Human adjudication is pending. These are exposed development cases, not a "
        "confirmatory or comparative result.**",
        "",
        "| Category | Provisional strict passes | Cases |",
        "|---|---:|---:|",
    ]
    for name, item in report["by_category"].items():
        lines.append(f"| {name} | {item['strict_provisional_passes']} | {item['n']} |")
    lines += [
        "",
        f"Application latency (grading excluded): {report['latency_seconds']}",
        "",
        "## Failure and review queue",
        "",
    ]
    for row in rows:
        if not row["strict_provisional_pass"]:
            lines.append(
                f"- **{row['id']}**: {', '.join(row['reason_codes'])}. {row['judge_reason'] or ''}"
            )
    lines += [
        "",
        f"Human-review records: {len(review)}. See `human-review-queue.json`.",
        "",
        *report["limitations"],
    ]
    (directory / "report.md").write_text("\n".join(lines) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize an existing run without paid calls.")
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    directory = (ROOT / "data/evaluation" / args.run_id).resolve()
    if directory.parent != (ROOT / "data/evaluation").resolve():
        raise ValueError("invalid run path")
    report = summarize(directory)
    print(
        json.dumps(
            {
                key: value
                for key, value in report.items()
                if key not in ("rows", "protocol", "audit")
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
