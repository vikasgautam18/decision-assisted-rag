from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from clients.decision import parse_choices
from evaluation.run import ROOT, persist
from pipelines.live import (
    APPLICABILITY_SUPPORT,
    CLAIM_SUPPORT,
    SUPPORT,
    DraftResponse,
    validated_answer,
)


def build_fixture(trace: dict, *, trace_sha256: str, origin: str) -> dict:
    sources = trace.get("evidence")
    drafts = [
        stage
        for stage in trace.get("stages", [])
        if stage["stage"] == "draft-v1" and stage.get("attempt") == 0
    ]
    checks = [
        stage
        for stage in trace.get("stages", [])
        if stage["stage"] == "verification-v1" and stage.get("attempt") == 0
    ]
    if not sources or len(drafts) != 1 or len(checks) != 1:
        raise ValueError("initial evidence/draft/verification boundary is unavailable")
    ids = [item["id"] for item in sources]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate replay source IDs")
    routes = [stage for stage in trace["stages"] if stage["stage"].startswith("routing-")]
    if len(routes) != 1:
        raise ValueError("routing provenance unavailable")
    route = routes[0]
    decision = route["decisions"]["figure_requirement"]["choice"]
    intent = route["decisions"]["intent"]["choice"]
    from pipelines.live import requests_visual_evidence

    required_figure = (
        intent == "figure"
        or requests_visual_evidence(trace["question"])
        or intent == "procedure"
        and decision == "required"
        and route["plan"]["temporal_scope"] != "history"
    )
    return {
        "kind": "fixed_final_evidence_initial_draft_verification_replay",
        "source_trace": origin,
        "source_trace_sha256": trace_sha256,
        "question": trace["question"],
        "sources": sources,
        "initial_draft": drafts[0]["artifact"],
        "initial_verification": checks[0]["decisions"],
        "require_figure": required_figure,
        "require_calculation": intent == "calculation",
        "routing_plan": route["plan"],
        "limitations": [
            "Final evidence was already selected by the full pipeline.",
            "No original pre-assessment candidate pool is available.",
            "Routing and expansion are fixed, not evaluated here.",
            "Recorded decisions are replayed; no fresh model or writer calls.",
            "This is a gate-mechanism check, not an independent quality score or live B/D arm.",
        ],
    }


def replay_verification(fixture: dict, *, enabled: bool) -> dict:
    draft = DraftResponse.model_validate_json(json.dumps(fixture["initial_draft"]))
    if draft.behavior not in ("answer", "conflict"):
        raise ValueError("verification replay requires a factual draft")
    try:
        validated_answer(draft, fixture["sources"], require_figure=fixture["require_figure"])
        if fixture["require_calculation"] and not draft.calculations:
            raise ValueError("arithmetic answer lacks a validated calculation record")
        platforms = set(fixture["routing_plan"]["platforms"])
        source_platforms = {
            item["document"]
            for item in fixture["sources"]
            if item["id"] in {sid for claim in draft.claims for sid in claim.source_ids}
        }
        if not platforms <= source_platforms:
            raise ValueError("answer omits requested platform evidence")
    except ValueError as exc:
        return {
            "action": "deterministic_rejection",
            "reason": str(exc),
            "verification_enabled": enabled,
            "failed_checks": [],
        }
    if not enabled:
        return {
            "action": "eligible_without_semantic_verification",
            "verification_enabled": False,
            "failed_checks": [],
        }
    recorded = fixture["initial_verification"]
    if {"whole_draft", "selected_figures", "applicability"} & {
        claim.claim_id for claim in draft.claims
    }:
        raise ValueError("claim ID collides with reserved verification check")
    options = {
        "whole_draft": SUPPORT,
        **{claim.claim_id: CLAIM_SUPPORT for claim in draft.claims},
    }
    if draft.figure_ids:
        options["selected_figures"] = CLAIM_SUPPORT
    if "applicability" in recorded:
        options["applicability"] = APPLICABILITY_SUPPORT
    decisions = parse_choices(
        {"answers": {key: {"type": "choice", **value} for key, value in recorded.items()}},
        options,
    )
    failed = [
        key
        for key, value in decisions.items()
        if value.choice != "supported"
        or value.probabilities is not None
        and value.probabilities["supported"] < 0.8
    ]
    return {
        "action": "requires_repair" if failed else "eligible_after_semantic_verification",
        "verification_enabled": True,
        "failed_checks": failed,
    }


def prepare(source: Path, target: Path) -> dict:
    if target.exists():
        raise ValueError("replay output already exists; choose a fresh run ID")
    manifest_path = source / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    target.mkdir(parents=True, mode=0o700)
    (target / "fixtures").mkdir(mode=0o700)
    rows = []
    for path in sorted((source / "responses").glob("*-F.json")):
        record = json.loads(path.read_text())
        payload = json.loads(record.get("raw_body", "{}"))
        trace_id = payload.get("trace_id")
        if not isinstance(trace_id, str) or not re.fullmatch(r"[a-f0-9]{32}", trace_id):
            rows.append({"id": path.stem, "status": "unavailable", "reason": "no trace identity"})
            continue
        trace_path = ROOT / f"data/review/traces/{trace_id}.json"
        if not trace_path.exists():
            rows.append({"id": path.stem, "status": "unavailable", "reason": "trace not retained"})
            continue
        raw = trace_path.read_bytes()
        trace = json.loads(raw)
        if trace.get("trace_id") != trace_id or trace.get("question") is None:
            raise ValueError("trace identity mismatch")
        try:
            fixture = build_fixture(
                trace,
                trace_sha256=hashlib.sha256(raw).hexdigest(),
                origin=str(trace_path.relative_to(ROOT)),
            )
        except ValueError as exc:
            rows.append({"id": path.stem, "status": "unavailable", "reason": str(exc)})
            continue
        persist(target / "fixtures" / path.name, fixture)
        rows.append(
            {
                "id": path.stem,
                "status": "prepared",
                "without_verification": replay_verification(fixture, enabled=False),
                "with_verification": replay_verification(fixture, enabled=True),
            }
        )
    if not rows:
        raise ValueError("source run has no F response records")
    result = {
        "status": "offline_verification_replay_prepared_not_quality_evaluation",
        "source_run": source.name,
        "source_manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "source_application_sha256": manifest["application_sha256"],
        "response_records": len(rows),
        "prepared": sum(row["status"] == "prepared" for row in rows),
        "unavailable": sum(row["status"] == "unavailable" for row in rows),
        "assessment_replay": "blocked: complete pre-assessment candidate pools not recorded",
        "paid_calls": 0,
        "rows": rows,
    }
    persist(target / "report.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare fixed-evidence offline verification replay."
    )
    parser.add_argument("--source-run", default="phase3-full-comparison")
    parser.add_argument("--run-id", default="phase3-verification-replay-offline")
    args = parser.parse_args()
    for value in (args.source_run, args.run_id):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", value):
            raise ValueError("invalid run ID")
    result = prepare(
        ROOT / "data/comparison" / args.source_run,
        ROOT / "data/comparison" / args.run_id,
    )
    print(json.dumps({key: value for key, value in result.items() if key != "rows"}, indent=2))


if __name__ == "__main__":
    main()
