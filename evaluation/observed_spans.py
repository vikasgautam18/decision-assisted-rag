from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from evaluation.observed_report import distribution
from evaluation.run import persist


def union_seconds(ranges: list[tuple[float, float]]) -> float:
    merged = []
    for start, end in sorted(ranges):
        if end < start:
            raise ValueError("span ends before it starts")
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
        else:
            merged.append((start, end))
    return sum(end - start for start, end in merged)


def analyze_trace(events: list[dict]) -> dict:
    if not events or len({event["trace_id"] for event in events}) != 1:
        raise ValueError("exactly one nonempty trace required")
    if len({event.get("observation_id") for event in events}) != 1:
        raise ValueError("trace crosses observation identities")
    if events[0].get("observation_id") is None:
        raise ValueError("trace has no exact observation identity")
    starts, ends = {}, {}
    for event in events:
        target = starts if event["event"] == "span_start" else ends
        if event["event"] not in ("span_start", "span_end"):
            raise ValueError("unsupported event schema")
        if event["span_id"] in target:
            raise ValueError("duplicate span boundary")
        target[event["span_id"]] = event
    if not ends.keys() <= starts.keys():
        raise ValueError("span end without a durable start")
    children = defaultdict(list)
    for key, event in starts.items():
        parent = event["parent_span_id"]
        if parent is not None and parent not in starts:
            raise ValueError("span parent does not exist in this trace")
        children[parent].append(key)
    rows = []
    for key, event in ends.items():
        start = starts[key]
        if event["name"] != start["name"] or event["parent_span_id"] != start["parent_span_id"]:
            raise ValueError("span identity changed")
        a, b = start["start_monotonic"], event["end_monotonic"]
        if b < a or abs(event["duration_seconds"] - (b - a)) > 1e-6:
            raise ValueError("span duration inconsistent with monotonic boundaries")
        intervals = []
        complete = True
        for child in children[key]:
            if child not in ends:
                complete = False
                continue
            ca = starts[child]["start_monotonic"]
            cb = ends[child]["end_monotonic"]
            if ca < a or cb > b:
                raise ValueError("child span lies outside its parent")
            intervals.append((ca, cb))
        rows.append(
            {
                "span_id": key,
                "name": event["name"],
                "status": event["status"],
                "inclusive_seconds": b - a,
                "exclusive_seconds": b - a - union_seconds(intervals) if complete else None,
            }
        )
    calls = []
    baseline_http_parents = {
        event["parent_span_id"]
        for event in starts.values()
        if event["name"] == "baseline.sdk_http_attempt"
    }
    # Only the last successful transport can be represented by the SDK result.
    # Earlier transports (including failed/incomplete sends) remain separate unknowns.
    represented_http = {}
    for event in sorted(ends.values(), key=lambda event: event["end_monotonic"]):
        if (
            event["name"] == "baseline.sdk_http_attempt"
            and event["status"] == "ok"
            and 200 <= (event.get("attributes", {}).get("http_status") or 0) < 300
        ):
            represented_http[event["parent_span_id"]] = event["span_id"]
    for key, start in starts.items():
        event = ends.get(key, start)
        attributes = {**start.get("attributes", {}), **event.get("attributes", {})}
        name = event["name"]
        if name in ("model.attempt", "evaluation.provider"):
            include = True
        elif name == "baseline.sdk_create":
            include = (
                event.get("status") == "ok"
                or key not in baseline_http_parents
                or key in represented_http
            )
        elif name == "baseline.sdk_http_attempt":
            include = represented_http.get(start["parent_span_id"]) != key
        else:
            include = False
        if include:
            call_id = (
                attributes.get("transport_call_id")
                if name == "baseline.sdk_http_attempt"
                else attributes.get("call_id")
            )
            if not call_id:
                raise ValueError("observable provider attempt has no unique call ID")
            calls.append(
                {
                    **attributes,
                    "call_id": call_id,
                    "observation_id": event["observation_id"],
                    "trace_id": event["trace_id"],
                    "span_id": key,
                    "span_name": name,
                    "parent_call_id": (
                        attributes.get("call_id") if name == "baseline.sdk_http_attempt" else None
                    ),
                    "represented_transport_call_id": (
                        ends[represented_http[key]]["attributes"].get("transport_call_id")
                        if key in represented_http
                        else None
                    ),
                    "experiment_id": event.get("experiment_id"),
                    "configuration_hash": event.get("configuration_hash"),
                    "usage": attributes.get("usage"),
                    "span_status": event["status"],
                    "arm": event.get("arm"),
                    "phase": event.get("phase"),
                    "attempt": event.get("attempt", event.get("attempt_id")),
                    "grading_attempt": event.get("grading_attempt"),
                    "usage_scope": "observable_provider_attempt_not_hidden_hosted_children",
                }
            )
    return {
        "trace_id": events[0]["trace_id"],
        "observation_id": events[0]["observation_id"],
        "experiment_id": events[0].get("experiment_id"),
        "phase": events[0].get("phase"),
        "arm": events[0].get("arm"),
        "attempt": events[0].get("attempt", events[0].get("attempt_id")),
        "status": "complete" if starts.keys() == ends.keys() else "incomplete",
        "incomplete_span_ids": sorted(starts.keys() - ends.keys()),
        "spans": rows,
        "calls": calls,
    }


def collect(directories: list[Path]) -> dict:
    traces, seen = [], set()
    exports = []
    for directory in directories:
        if not directory.exists():
            raise ValueError(f"span export does not exist: {directory}")
        paths = [directory] if directory.is_file() else sorted(directory.glob("*.jsonl"))
        for path in paths:
            data = path.read_bytes()
            events = [json.loads(line) for line in data.splitlines() if line.strip()]
            trace = analyze_trace(events)
            if path.read_bytes() != data:
                raise ValueError("span export changed during offline collection")
            if trace["trace_id"] in seen:
                raise ValueError("trace exported twice; refuse duplicate accounting")
            seen.add(trace["trace_id"])
            traces.append(trace)
            exports.append({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})
    grouped = defaultdict(list)
    for trace in traces:
        for span in trace["spans"]:
            grouped[span["name"]].append(span["inclusive_seconds"])
    return {
        "status": "observed_spans_not_total_hosted_visibility",
        "trace_count": len(traces),
        "source_exports": exports,
        "incomplete_traces": sum(trace["status"] != "complete" for trace in traces),
        "stage_inclusive_distributions": {
            name: distribution(values) for name, values in sorted(grouped.items())
        },
        "traces": traces,
        "calls": [call for trace in traces for call in trace["calls"]],
        "limitations": [
            "Inclusive stage durations overlap; never sum them as request latency.",
            "Exclusive times subtract the union of direct child intervals.",
            "Unobserved hosted model/tool work and failed-call usage remain unknown.",
            "Different process monotonic clocks are not subtracted.",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze private spans offline without paid calls."
    )
    parser.add_argument("--spans", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    persist(args.output, collect(args.spans))


if __name__ == "__main__":
    main()
