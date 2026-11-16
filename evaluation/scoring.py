from __future__ import annotations

import json
import re

from pydantic import TypeAdapter

from evaluation.dataset import Case
from pipelines.calculations import CalculationRecord, verify_calculations
from pipelines.contracts import MaterialClaim


def deterministic(
    case: Case, response: dict, catalogue: dict[str, dict], figure_pages: dict[str, tuple[str, int]]
) -> dict:
    expected = "conflict" if case.expected_behavior == "flag_conflict" else case.expected_behavior
    completed = response.get("status") == "completed"
    citations = response.get("citations", [])
    figures = response.get("figures", [])
    valid_ids, covered, errors = set(), set(), []
    for cite in citations:
        source = catalogue.get(cite.get("source_id"))
        if source is None or any(
            cite.get(key) != source[key]
            for key in ("document", "revision", "page_start", "page_end")
        ):
            errors.append("citation identity not in prepared source catalogue")
            continue
        valid_ids.add(source["id"])
        covered.update(
            (source["document"], page)
            for page in range(source["page_start"], source["page_end"] + 1)
        )
    markers = set(re.findall(r"\[([^\[\]\n]+)\]", response.get("answer") or ""))
    if not markers <= valid_ids:
        errors.append("citation markers are not mapped to validated citations")
    gold = {(item.doc, item.page) for item in case.gold_sources}
    correct_figures, wrong_figures = [], []
    for figure in figures:
        key = figure.get("key")
        location = figure_pages.get(key)
        if location is None or figure.get("url") != f"/api/figures/{key}":
            wrong_figures.append(key)
        elif location in gold:
            correct_figures.append(key)
        elif gold:
            wrong_figures.append(key)
    factual = expected in ("answer", "conflict")
    matched = gold & covered
    checks = {
        "completed": completed,
        "behavior": completed and response.get("behavior") == expected,
        "verified": not factual or response.get("semantic_verification") == "passed",
        "citation_identity": not errors,
        "citation_present": not factual or bool(valid_ids) and bool(markers),
        "gold_page_match": not factual or bool(matched),
        "original_figure": not case.requires_figure or bool(correct_figures),
        "no_wrong_figures": not wrong_figures,
    }
    return {
        "checks": checks,
        "pass": all(checks.values()),
        "gold_page_recall": len(matched) / len(gold) if gold else None,
        "matched_gold_pages": sorted(matched),
        "missing_gold_pages": sorted(gold - covered),
        "correct_figures": correct_figures,
        "wrong_figures": wrong_figures,
        "identity_errors": errors,
    }


def trace_checks(response: dict, trace: dict | None) -> dict:
    if trace is None:
        return {"available": False, "reason": "No private trace available; checks unknown."}
    sources = trace.get("evidence", [])
    drafts = [stage for stage in trace.get("stages", []) if stage["stage"] == "draft-v1"]
    delivered = response.get("behavior") in ("answer", "conflict")
    calculation_error = None
    if delivered:
        if not drafts:
            calculation_error = "delivered answer has no recorded draft"
        else:
            artifact = drafts[-1]["artifact"]
            try:
                claims = tuple(
                    MaterialClaim.model_validate_json(json.dumps(item))
                    for item in artifact["claims"]
                )
                records = tuple(
                    TypeAdapter(CalculationRecord).validate_json(json.dumps(item))
                    for item in artifact.get("calculations", [])
                )
                verify_calculations(records, sources, artifact["draft"], claims)
                if artifact["draft"] != response["answer"]:
                    calculation_error = "trace draft does not match delivered answer"
            except ValueError as exc:
                calculation_error = str(exc)
    calls = trace.get("calls", [])
    return {
        "available": True,
        "calculation_error": calculation_error,
        "decision_calls": sum(call["category"] == "decision" for call in calls),
        "model_calls": sum(call["category"] == "model" for call in calls),
        "expansion_rounds": sum(
            stage["stage"] == "expansion-v1" for stage in trace.get("stages", [])
        ),
        "usage_reported_calls": sum(call.get("usage") is not None for call in calls),
        "usage_unknown_calls": sum(call.get("usage") is None for call in calls),
        "source_ids": [source["id"] for source in sources],
    }


def page_evidence(case: Case, response: dict, catalogue: dict[str, dict]) -> list[dict]:
    cited = {item["source_id"] for item in response.get("citations", [])}
    gold_pages = {(item.doc, item.page) for item in case.gold_sources}
    return [
        item
        for item in catalogue.values()
        if item["id"] in cited
        or any(
            item["document"] == doc and item["page_start"] <= page <= item["page_end"]
            for doc, page in gold_pages
        )
    ]
