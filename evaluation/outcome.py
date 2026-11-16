from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path

from evaluation.run import ROOT, persist


def latency_summary(values: list[float]) -> dict:
    if not values or any(not math.isfinite(value) or value < 0 for value in values):
        raise ValueError("latency samples must be nonempty, finite and nonnegative")
    ordered = sorted(values)
    return {
        "n": len(values),
        "mean_seconds": statistics.mean(values),
        "median_seconds": statistics.median(values),
        "p95_seconds": ordered[math.ceil(0.95 * len(values)) - 1],
    }


def consolidate(source: Path, retry: Path) -> dict:
    first = json.loads((source / "report.json").read_text())
    final = json.loads((retry / "report.json").read_text())
    human = json.loads((retry / "human-adjudicated-summary.json").read_text())
    queue_bytes = (retry / "human-review-queue.json").read_bytes()
    if hashlib.sha256(queue_bytes).hexdigest() != human["source_review_sha256"]:
        raise ValueError("human review changed; refresh adjudication before consolidating")
    queue = json.loads(queue_bytes)
    if any(item["human_status"] != "reviewed" for item in queue["items"]):
        raise ValueError("human review is incomplete")
    source_manifest = (source / "manifest.json").read_bytes()
    policy = json.loads((retry / "retry-provenance.json").read_text())
    if hashlib.sha256(source_manifest).hexdigest() != policy["source_manifest_sha256"]:
        raise ValueError("retry does not match the first-attempt manifest")
    if len(first["rows"]) != 144 or len(final["rows"]) != 144:
        raise ValueError("full comparison requires all 72 pairs")
    by_arm = {}
    for arm in ("A", "F"):
        first_rows = {row["id"]: row for row in first["rows"] if row["arm"] == arm}
        final_rows = {row["id"]: row for row in final["rows"] if row["arm"] == arm}
        if len(first_rows) != 72 or first_rows.keys() != final_rows.keys():
            raise ValueError("incomplete or duplicate comparison pairs")
        initial_latencies, attempt_sums = [], []
        retries = 0
        for case_id in first_rows:
            name = f"{case_id}-{arm}.json"
            original = json.loads((source / "responses" / name).read_text())
            selected = json.loads((retry / "responses" / name).read_text())
            initial = original["application_seconds"]
            if not isinstance(initial, (int, float)):
                raise ValueError("first-attempt response latency is missing")
            initial_latencies.append(initial)
            if selected.get("reused_from"):
                if selected["application_seconds"] != initial:
                    raise ValueError("reused response latency changed")
                attempt_sums.append(initial)
            else:
                if not isinstance(selected.get("application_seconds"), (int, float)):
                    raise ValueError("retry response latency is missing")
                retries += 1
                attempt_sums.append(initial + selected["application_seconds"])
        by_arm[arm] = {
            "first_attempt": first["by_arm"][arm],
            "retry_adjusted": final["by_arm"][arm],
            "human_adjusted": human["by_arm"][arm],
            "application_retries": retries,
            "remaining_unknown_grading": [
                case_id
                for case_id, row in final_rows.items()
                if row["status"] == "completed" and row["grading_status"] != "completed"
            ],
            "first_attempt_latency": latency_summary(initial_latencies),
            "application_attempt_sum_latency": latency_summary(attempt_sums),
        }
    grading = {
        "attempts": 0,
        "reported_usage_attempts": 0,
        "unknown_usage_attempts": 0,
        "reported_prompt_tokens": 0,
        "reported_completion_tokens": 0,
    }
    for directory in (source, retry):
        for path in (directory / "judgments").glob("*.json"):
            record = json.loads(path.read_text())
            if record.get("reused_from"):
                continue
            grading["attempts"] += 1
            usage = record.get("usage")
            if usage is None:
                grading["unknown_usage_attempts"] += 1
                continue
            grading["reported_usage_attempts"] += 1
            grading["reported_prompt_tokens"] += usage["prompt_tokens"]
            grading["reported_completion_tokens"] += usage["completion_tokens"]
    provenance = {
        f"{directory.name}/{name}": hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for directory, names in (
            (source, ("manifest.json", "report.json")),
            (retry, ("report.json", "human-review-queue.json", "human-adjudicated-summary.json")),
        )
        for name in names
    }
    return {
        "status": "primary_AF_comparison_complete_controls_pending",
        "source_runs": [source.name, retry.name],
        "provenance_sha256": provenance,
        "by_arm": by_arm,
        "human_paired_counts": human["paired_counts"],
        "manual_review_pairs": human["reviewed_question_pairs"],
        "unreviewed_automated_pairs": human["unreviewed_question_pairs"],
        "grading_usage": grading,
        "runtime_usage_and_monetary_cost": "Unknown in public captures; not zero.",
        "latency_policy": "First-attempt chat HTTP only. Attempt sums include failed first "
        "requests plus retry HTTP duration, not intervening grading, queue waits between "
        "attempts, image retrieval or elapsed time until eventual success.",
        "conclusion": "No demonstrated quality or first-attempt reliability improvement for F. "
        "Human-adjusted A passes 72/72, F 71/72; A completed every first attempt, F needed "
        "six application retries. This is not a statistical superiority claim.",
        "remaining_correction": human["remaining_corrections"],
        "limitations": human["limitations"]
        + [
            "A's native retries remain hidden; A has 420s external allowance, "
            "F 180s internal deadline and 195s HTTP allowance.",
            "PNG delivery/metadata checks do not establish original-image semantic identity.",
            "Single exposed development run; no calibration, stability or production-load claim.",
            "No new per-arm usage instrumentation was added to the unchanged reference.",
            "No component controls B-E or ordinary-LLM control G have run.",
        ],
    }


def render(report: dict) -> str:
    a, f = report["by_arm"]["A"], report["by_arm"]["F"]
    lines = [
        "# Primary A/F comparison outcome",
        "",
        "**No demonstrated quality or first-attempt reliability improvement for the new app.**",
        "This primary comparison is complete; component-control experiments remain pending.",
        "",
        "| Metric | Original A | Decision-assisted F |",
        "|---|---:|---:|",
    ]
    for label, av, fv in (
        (
            "First-attempt completed responses",
            a["first_attempt"]["application_completed"],
            f["first_attempt"]["application_completed"],
        ),
        (
            "First-attempt strict provisional passes",
            a["first_attempt"]["strict_provisional_passes"],
            f["first_attempt"]["strict_provisional_passes"],
        ),
        (
            "Retry-adjusted strict provisional passes",
            a["retry_adjusted"]["strict_provisional_passes"],
            f["retry_adjusted"]["strict_provisional_passes"],
        ),
        (
            "Human-adjusted accepted answers",
            a["human_adjusted"]["human_adjusted_passes"],
            f["human_adjusted"]["human_adjusted_passes"],
        ),
    ):
        lines.append(f"| {label} | {av}/72 | {fv}/72 |")
    for label, key in (
        ("First-attempt mean HTTP seconds", "first_attempt_latency"),
        ("Retry-inclusive mean application-attempt seconds", "application_attempt_sum_latency"),
    ):
        lines.append(f"| {label} | {a[key]['mean_seconds']:.2f} | {f[key]['mean_seconds']:.2f} |")
    lines.extend(
        [
            "",
            f"{report['manual_review_pairs']} question pairs were manually reviewed; "
            f"{report['unreviewed_automated_pairs']} retain automated passes. "
            "There are 71 both-pass pairs and one A-only accepted pair after adjudication. "
            "One discordant case does not establish statistical superiority.",
            "",
            "## Interpretation",
            "",
            "The automated retry-adjusted gap (A 51/72 versus F 68/72) did not survive "
            "human adjudication. Grading outages, disputed applicability/extra-detail judgments "
            "and gold issues materially affected that automated comparison. Do not use the "
            "unadjudicated score gap as evidence of improvement.",
            "",
            "F's remaining issue is q71: its torque answer must condition rear subframe "
            "isolation-bushing fitment on the Plug-in Hybrid variant. A already did so. "
            "No post-comparison correction has been substituted into these scores.",
            "",
            "A was also faster on mean observed HTTP time in this run. Native retry, timeout "
            "and infrastructure differences prevent a controlled model-speed conclusion.",
            report["latency_policy"],
            "",
            "## Usage and cost",
            "",
            report["runtime_usage_and_monetary_cost"],
            f"Independent grading had {report['grading_usage']['attempts']} attempts, "
            f"{report['grading_usage']['reported_usage_attempts']} with reported usage and "
            f"{report['grading_usage']['unknown_usage_attempts']} without reported usage. "
            "Reused judgments are counted only once. Reported-only grader tokens: "
            f"{report['grading_usage']['reported_prompt_tokens']} input and "
            f"{report['grading_usage']['reported_completion_tokens']} output. "
            "These are not total application tokens or monetary cost.",
            "",
            "## Recommendation and outstanding work",
            "",
            "Retain the original application as the demonstrated reference. Treat the new "
            "application as an experimental architecture, not a proven replacement. Fix q71 "
            "separately, investigate transport reliability, and use matched controls to learn "
            "whether any Decision-1 stage justifies its complexity. B-E/G remain unexecuted; "
            "their protocol and call allocation must be agreed before live experiments.",
            "",
            "## Limitations",
            "",
            *[f"- {item}" for item in report["limitations"]],
            "",
            "## Evidence",
            "",
            *[
                f"- `{name}` SHA-256 `{digest}`"
                for name, digest in report["provenance_sha256"].items()
            ],
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Consolidate accepted A/F results without inference."
    )
    parser.add_argument("--run-id", default="phase3-full-comparison")
    args = parser.parse_args()
    if not args.run_id or Path(args.run_id).name != args.run_id or args.run_id in (".", ".."):
        raise ValueError("invalid run ID")
    source = ROOT / "data/comparison" / args.run_id
    retry = source.with_name(source.name + "-retry1")
    report = consolidate(source, retry)
    cost_path = retry / "cost-report.json"
    if cost_path.exists():
        report["partial_cost_comparison"] = json.loads(cost_path.read_text())
        report["runtime_usage_and_monetary_cost"] = (
            "Partial retail model-cost estimates recovered from baseline audit and F traces; "
            "full runtime cost remains unknown. See the appended cost comparison."
        )
    persist(retry / "outcome.json", report)
    target = retry / "outcome.md"
    text = render(report)
    if "partial_cost_comparison" in report:
        from evaluation.costs import render as render_cost

        text += "\n" + render_cost(report["partial_cost_comparison"])
    target.write_text(text)
    print(target)


if __name__ == "__main__":
    main()
