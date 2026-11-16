from __future__ import annotations

import re

from ingestion.corpus import DOSSIERS
from pipelines.contracts import QueryContext, RetrievalPlan

EXPANSION_SECTIONS = {
    "sufficient": "The requested facts for this platform are already fully covered.",
    "overview": "Section 1 declared performance tables: maximum power (engine/system), torque, "
    "range, capacity, mass, payload, towing and dimensions. Baseline power/range are HERE.",
    "powertrain": "Section 2 engine construction, compression/injection and chassis; "
    "NOT baseline maximum power/range (use overview/Section 1).",
    "homologation": "Section 3 test log IDs, outcomes and cross-page context; NOT power ratings.",
    "mounting": "Missing mounting, callouts, assembly or torque details.",
    "suppliers": "Missing supplier quality or commercial terms.",
    "warranty": "Missing warranty procedure, outcome codes or authority.",
    "thermal": "Missing temperature corrections, limits or derating.",
    "maintenance": "Missing replacement intervals, wear items or referenced maintenance tables.",
    "history": "Missing explicitly requested superseded appendix values.",
}


def section_candidates(
    catalogue: list[dict],
    platform: str,
    section: str,
    historical: bool,
    sections: dict[str, dict[str, tuple[int, ...]]],
) -> list[dict]:
    if section == "sufficient":
        return []
    if platform not in sections or section not in sections[platform]:
        raise ValueError("reviewed section mapping is unavailable")
    pages = sections[platform][section]
    return [
        item
        for item in catalogue
        if item["document"] == platform
        and any(item["page_start"] <= page <= item["page_end"] for page in pages)
        and (historical or not item["historical"])
    ]


def referenced_section_candidates(
    sources: list[dict],
    catalogue: list[dict],
    sections: dict[str, dict[str, tuple[int, ...]]],
    historical: bool,
) -> list[dict]:
    names = (
        "overview",
        "powertrain",
        "homologation",
        "mounting",
        "suppliers",
        "warranty",
        "thermal",
        "maintenance",
    )
    dependencies = []
    for source in sources:
        identifiers = set(re.findall(r"\b\d{3,6}-[A-Z0-9]{1,6}\b", source["text"]))
        dependencies.extend(
            item
            for item in catalogue
            if item["document"] == source["document"]
            and (historical or not item["historical"])
            and identifiers.intersection(re.findall(r"\b\d{3,6}-[A-Z0-9]{1,6}\b", item["text"]))
        )
        if source["document"] not in sections:
            continue
        references = re.findall(
            r"\bSections?\s+([1-8](?:(?:\s*,\s*|\s+and\s+)[1-8])*)",
            source["text"],
            re.IGNORECASE,
        )
        for reference in references:
            for number in re.findall(r"[1-8]", reference):
                dependencies.extend(
                    section_candidates(
                        catalogue, source["document"], names[int(number) - 1], historical, sections
                    )
                )
    return bounded_sources(dependencies, max_chars=45_000)


def explicit_platforms(question: str) -> tuple[str, ...]:
    return tuple(
        document
        for document in DOSSIERS
        if re.search(rf"\b{re.escape(document)}\b", question, re.IGNORECASE)
    )


def request_plan(
    question: str, history: list[str], routing: dict
) -> tuple[QueryContext, RetrievalPlan]:
    platforms = tuple(
        document for document in DOSSIERS if routing[f"scope_{document}"].choice == "include"
    )
    intent = routing["intent"].choice
    strategy = {
        "figure": "procedure",
        "lookup": "lookup",
        "comparison": "comparison",
        "calculation": "calculation",
        "history": "history",
        "procedure": "procedure",
    }.get(intent, "lookup")
    current_explicit = explicit_platforms(question)
    # Older turns cannot replace newly requested documentary scope.
    prior = [] if current_explicit else history[-2:]
    retrieval = question + (
        "\nPrevious user intent only, not evidence: " + "\n".join(prior) if prior else ""
    )
    context = QueryContext(
        original_question=question, retrieval_question=retrieval, platforms=platforms
    )
    plan = RetrievalPlan(
        strategy=strategy,
        question=retrieval,
        platforms=platforms,
        temporal_scope="historical" if routing["temporal_scope"].choice == "history" else "current",
        required_slots=platforms,
        primary_section=(
            "overview"
            if routing.get("comparison_basis")
            and routing["comparison_basis"].choice == "declared_specifications"
            else None
        ),
    )
    return context, plan


def bounded_sources(sources: list[dict], *, max_chars: int = 65_000) -> list[dict]:
    grouped: dict[str, list[dict]] = {}
    for source in sources:
        grouped.setdefault(source.get("document", ""), []).append(source)
    ranked = [
        group[index]
        for index in range(max((len(group) for group in grouped.values()), default=0))
        for group in grouped.values()
        if index < len(group)
    ]
    selected, seen, size = [], set(), 0
    for source in ranked:
        if source["id"] in seen:
            continue
        length = len(source["text"])
        if size + length <= max_chars and len(selected) < 24:
            selected.append(source)
            seen.add(source["id"])
            size += length
    return selected


def merge_expansion(
    sources: list[dict], assessed: list[dict], required: list[dict], require_figures: bool
) -> list[dict]:
    figures = (
        [item for item in sources + assessed + required if item["kind"] == "figure"]
        if require_figures
        else []
    )
    return bounded_sources(figures + required + assessed + sources)


def expand_context(
    sources: list[dict], catalogue: list[dict], platforms: tuple[str, ...], historical: bool
) -> list[dict]:
    candidates = []
    for source in sources:
        refs = set(re.findall(r"\b(?:Table|Figure)\s+(\d+\.\d+[a-z]?)", source["text"], re.I))
        for record in catalogue:
            if record["document"] != source["document"] or record["document"] not in platforms:
                continue
            if record["historical"] and not historical:
                continue
            same_page = (
                record["page_start"] <= source["page_end"]
                and record["page_end"] >= source["page_start"]
            )
            referenced = any(
                re.search(rf"\b(?:Table|Figure)\s+{re.escape(ref)}\b", record["text"], re.I)
                for ref in refs
            )
            continuation = (
                f"Header context from {source['id']}:" in record["text"]
                or f"Header context from {record['id']}:" in source["text"]
            )
            if same_page or referenced or continuation:
                candidates.append(record)
    return bounded_sources(candidates, max_chars=45_000)
