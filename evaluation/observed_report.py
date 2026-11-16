from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from evaluation.experiment import Protocol, digest, schedule
from evaluation.run import persist
from pipelines.arms import ARMS


def distribution(values: list[float]) -> dict:
    if any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0 for value in values
    ):
        raise ValueError("durations must be finite nonnegative numbers")
    if not values:
        return {"n": 0, "mean": None, "median": None, "p90": None, "p95": None, "max": None}
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p90": ordered[math.ceil(0.90 * len(values)) - 1],
        "p95": ordered[math.ceil(0.95 * len(values)) - 1],
        "max": ordered[-1],
    }


def strict_quality(record: dict, grading: dict | None) -> bool | None:
    if record.get("status") != "completed":
        return False
    if grading is None or grading.get("status") != "completed":
        return None
    judgment = grading["judgment"]
    return (
        all(record.get("public_checks", {"missing_checks": False}).values())
        and all(
            judgment[key]
            for key in (
                "correct",
                "complete",
                "claims_supported",
                "citations_supported",
            )
        )
        and not judgment["uncertain"]
        and not judgment["gold_issue"]
        and judgment["behavior"] == grading["expected_behavior"]
    )


def clustered_difference(
    pairs: list[tuple[str, float, float]], *, seed: int = 71, samples: int = 2000
) -> dict:
    if not pairs or samples < 1:
        raise ValueError("nonempty matched observations and positive bootstrap size required")
    clusters = defaultdict(list)
    for case_id, a, b in pairs:
        if not math.isfinite(a) or not math.isfinite(b):
            raise ValueError("paired measurements must be finite")
        clusters[case_id].append(a - b)
    # Each question has equal weight despite missing repetitions.
    means = [statistics.mean(values) for _, values in sorted(clusters.items())]
    rng = random.Random(seed)
    draws = sorted(statistics.mean(rng.choices(means, k=len(means))) for _ in range(samples))
    return {
        "mean_question_difference": statistics.mean(means),
        "interval_95_percentile": [
            draws[int(samples * 0.025)],
            draws[min(samples - 1, math.ceil(samples * 0.975) - 1)],
        ],
        "question_clusters": len(means),
        "matched_observations": len(pairs),
        "method": "seeded question-cluster bootstrap; exploratory, not multiplicity-adjusted",
        "limitations": "Exposed questions; correlated fact families may reduce "
        "effective diversity.",
    }


def summarize(protocol: Protocol, records: list[dict], judgments: dict[str, dict]) -> dict:
    if protocol.phase == "load":
        return summarize_load(protocol, records)
    expected = schedule(protocol)
    ids = {row.observation_id for row in expected}
    by_id = {}
    for record in records:
        key = record["observation_id"]
        if key in by_id:
            raise ValueError("duplicate observation; do not silently select a favorable answer")
        by_id[key] = record
    extras = [record for key, record in by_id.items() if key not in ids]
    if any(record.get("attempt") != 2 for record in extras):
        raise ValueError("unplanned observation in main results")
    logical_ids = {row.logical_request_id for row in expected}
    if any(record.get("logical_request_id") not in logical_ids for record in extras):
        raise ValueError("retry has no matching scheduled logical request")
    by_arm = {}
    for arm in ARMS:
        rows = [row for row in expected if row.arm == arm]
        initial = [by_id[row.observation_id] for row in rows if row.observation_id in by_id]
        completed = [record for record in initial if record.get("status") == "completed"]
        times = [
            record["application_seconds"]
            for record in initial
            if record.get("application_seconds") is not None
        ]
        successful_times = [
            record["application_seconds"]
            for record in completed
            if record.get("application_seconds") is not None
        ]
        quality = [
            strict_quality(record, judgments.get(record["observation_id"])) for record in initial
        ]
        by_arm[arm] = {
            "scheduled_first_attempts": len(rows),
            "captured_first_attempts": len(initial),
            "not_captured": len(rows) - len(initial),
            "completed": len(completed),
            "operational_errors": sum(record.get("status") == "error" for record in initial),
            "interrupted_or_reserved": sum(
                record.get("status") not in ("completed", "error") for record in initial
            ),
            "strict_provisional_passes": quality.count(True),
            "grading_unknown_delivered": sum(
                strict_quality(record, judgments.get(record["observation_id"])) is None
                for record in completed
            ),
            "pass_fraction_all_scheduled": quality.count(True) / len(rows) if rows else None,
            "all_attempt_latency_seconds": distribution(times),
            "completed_latency_seconds": distribution(successful_times),
            "completed_within_180_seconds": sum(
                record.get("application_seconds", math.inf) <= 180 for record in completed
            ),
        }
    retry_elapsed = {}
    for record in extras:
        first = next(
            (
                item
                for item in by_id.values()
                if item.get("logical_request_id") == record["logical_request_id"]
                and item.get("attempt") == 1
            ),
            None,
        )
        if first is None:
            raise ValueError("retry has no captured initial attempt")
        if record.get("status") == "completed":
            elapsed = (
                datetime.fromisoformat(record["completed_utc"])
                - datetime.fromisoformat(first["started_utc"])
            ).total_seconds()
            if elapsed < 0:
                raise ValueError("invalid retry timestamps")
            retry_elapsed[record["logical_request_id"]] = elapsed
    matched = {}
    index = {
        (record["case_id"], record["pass_number"], record["arm"]): record
        for record in by_id.values()
        if record.get("attempt") == 1
    }
    for left, right in (
        ("A", "F"),
        ("C", "B"),
        ("D", "B"),
        ("E", "C"),
        ("E", "D"),
        ("F", "E"),
        ("F", "G"),
    ):
        pairs = []
        for case_id in protocol.case_ids:
            for pass_number in range(1, protocol.passes + 1):
                a = index.get((case_id, pass_number, left))
                b = index.get((case_id, pass_number, right))
                if (
                    a
                    and b
                    and all(record.get("application_seconds") is not None for record in (a, b))
                ):
                    pairs.append((case_id, a["application_seconds"], b["application_seconds"]))
        matched[f"{left}-{right}"] = clustered_difference(pairs) if pairs else None
    return {
        "status": "provisional_quality_billing_and_human_review_not_finalized",
        "protocol_sha256": protocol.sha256,
        "cost_scope": protocol.cost_scope.report(),
        "by_arm": by_arm,
        "application_retries_captured": len(extras),
        "retry_elapsed_to_completed_response_seconds": retry_elapsed,
        "paired_all_attempt_latency_differences": matched,
        "cost": "requires exact-ID call reconciliation and verified rate card; not zero",
        "human_review": "not_incorporated",
        "limitations": [
            "Five repetitions are repeated observations of 72 exposed questions, "
            "not 360 new cases.",
            "Unknown grading is not a confirmed semantic failure.",
            "Native retries/deadlines differ; G's categorical acceptance differs from Decision-1.",
            "Provider cache and deployment placement require separate stratified analysis.",
            "No superiority, equivalence, invoice-total or production-capacity claim.",
        ],
    }


def summarize_load(protocol: Protocol, records: list[dict]) -> dict:
    expected = {row.observation_id: row for row in schedule(protocol)}
    seen = set()
    grouped = defaultdict(list)
    for record in records:
        key = record["observation_id"]
        if key in seen or key not in expected:
            raise ValueError("duplicate or unplanned load observation")
        seen.add(key)
        row = expected[key]
        grouped[(row.arm, row.concurrency)].append(record)
    cells = []
    for level in protocol.concurrency_levels:
        for arm in ARMS:
            rows = grouped[(arm, level)]
            events, durations = [], []
            completed = 0
            for row in rows:
                if row.get("application_seconds") is not None:
                    durations.append(row["application_seconds"])
                if row.get("started_utc") and row.get("completed_utc"):
                    events.append((datetime.fromisoformat(row["started_utc"]).timestamp(), 1))
                    events.append((datetime.fromisoformat(row["completed_utc"]).timestamp(), -1))
                completed += row.get("status") == "completed"
            active = peak = 0
            for _, delta in sorted(events):
                active += delta
                peak = max(peak, active)
            elapsed = max((x[0] for x in events), default=0) - min(
                (x[0] for x in events), default=0
            )
            cells.append(
                {
                    "arm": arm,
                    "target_concurrency": level,
                    "client_observed_peak_in_flight": peak,
                    "scheduled": len(protocol.case_ids),
                    "captured": len(rows),
                    "completed": completed,
                    "latency_seconds": distribution(durations),
                    "client_request_window_seconds": elapsed if events else None,
                    "completed_per_second": completed / elapsed if elapsed > 0 else None,
                }
            )
    return {
        "status": "separate_closed_loop_load_diagnostics_not_quality_or_capacity_proof",
        "protocol_sha256": protocol.sha256,
        "cost_scope": protocol.cost_scope.report(),
        "cells": cells,
        "limitations": [
            "Client-observed outstanding requests are not exact provider-side concurrency.",
            "Shared resource contention and admission must be interpreted using service spans.",
            "Twelve observations per cell do not establish production saturation capacity.",
        ],
    }


def review_queue(protocol: Protocol, records: list[dict], judgments: dict[str, dict]) -> dict:
    groups = {}
    outcomes = defaultdict(set)
    behaviors = defaultdict(set)
    for record in records:
        if record.get("attempt") == 1:
            outcomes[record["case_id"]].add(
                strict_quality(record, judgments.get(record["observation_id"]))
            )
        if record.get("status") == "completed":
            behavior = (
                judgments.get(record["observation_id"], {}).get("judgment", {}).get("behavior")
            )
            behaviors[(record["case_id"], record["arm"])].add(behavior)
    for record in records:
        key = record["observation_id"]
        grading = judgments.get(key)
        quality = strict_quality(record, grading)
        reasons = []
        if quality is not True:
            reasons.append("failure_or_unknown_or_uncertain_grading")
        if len(outcomes[record["case_id"]]) > 1:
            reasons.append("quality_disagreement_within_question")
        if len(behaviors[(record["case_id"], record["arm"])]) > 1:
            reasons.append("within_arm_behavior_instability")
        if int(digest({"seed": protocol.seed, "case": record["case_id"]})[:8], 16) % 10 == 0:
            reasons.append("seeded_agreement_audit")
        if not reasons:
            continue
        group_id = digest(
            {
                "case_id": record["case_id"],
                "public_answer": record.get("public_answer"),
                "dataset_sha256": protocol.dataset_sha256,
                "review_overlay_sha256": protocol.review_overlay_sha256,
            }
        )
        group = groups.setdefault(
            group_id,
            {
                "group_id": group_id,
                "case_id": record["case_id"],
                "public_answer": record.get("public_answer"),
                "observation_ids": [],
                "review_reasons": [],
                "human_status": "pending",
                "human_decision": None,
                "human_notes": None,
            },
        )
        group["observation_ids"].append(key)
        group["review_reasons"] = sorted(set(group["review_reasons"] + reasons))
    return {
        "status": "human_review_pending",
        "protocol_sha256": protocol.sha256,
        "deduplication": "Exact public answer/citations, case, frozen dataset "
        "and review overlay only.",
        "transfer_policy": "A group decision explicitly covers listed observations only.",
        "groups": list(groups.values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reproduce provisional observed-run results offline."
    )
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    protocol = Protocol.model_validate_json((args.directory / "protocol.json").read_bytes())
    records = [
        json.loads(path.read_text())
        for path in sorted((args.directory / "responses").glob("*.json"))
    ]
    judgments = {}
    for path in sorted((args.directory / "judgments").glob("*.json")):
        record = json.loads(path.read_text())
        key = record.get("observation_id", path.stem)
        prior = judgments.get(key)
        if prior is None or (prior.get("status") == "error" and record.get("grading_attempt") == 2):
            judgments[key] = record
    report = summarize(protocol, records, judgments)
    persist(args.directory / "provisional-report.json", report)
    queue = args.directory / "reviews" / "human-review-queue.json"
    if protocol.phase != "load" and not queue.exists():
        persist(queue, review_queue(protocol, records, judgments))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
