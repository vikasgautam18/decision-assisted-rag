from clients.decision import ChoiceResult
from ingestion.full import DOSSIERS
from pipelines.evidence import (
    bounded_sources,
    expand_context,
    merge_expansion,
    referenced_section_candidates,
    request_plan,
)


def routing(platforms, intent="lookup", temporal="current"):
    return {
        "intent": ChoiceResult(intent, None),
        "temporal_scope": ChoiceResult(temporal, None),
        **{
            f"scope_{key}": ChoiceResult("include" if key in platforms else "exclude", None)
            for key in DOSSIERS
        },
    }


def test_scope_switch_does_not_retrieve_old_user_scope():
    context, plan = request_plan(
        "Now VEGA-7 petrol range?", ["NOVA-3 Dedicated CNG range?"], routing(("VEGA-7",))
    )
    assert plan.platforms == ("VEGA-7",)
    assert "NOVA-3" not in context.retrieval_question


def test_followup_retains_user_intent_not_assistant_evidence():
    context, plan = request_plan(
        "What about petrol?", ["VEGA-7 diesel range?"], routing(("VEGA-7",))
    )
    assert "VEGA-7" in context.retrieval_question
    assert plan.platforms == ("VEGA-7",)


def test_declared_comparison_basis_selects_reviewed_performance_section():
    choices = routing(("NOVA-3", "VEGA-7"), intent="comparison")
    choices["comparison_basis"] = ChoiceResult("declared_specifications", None)
    _, plan = request_plan("Compare declared power", [], choices)
    assert plan.primary_section == "overview"


def test_expansion_respects_platform_and_temporal_bounds():
    source = {
        "id": "source",
        "document": "NOVA-3",
        "text": "Refer to Table 8.2.",
        "page_start": 10,
        "page_end": 10,
        "historical": False,
    }
    matching = {
        **source,
        "id": "linked-table",
        "text": "Table 8.2 - wear items",
        "page_start": 12,
        "page_end": 12,
    }
    history = {**matching, "id": "old-table", "historical": True}
    foreign = {**matching, "id": "other", "document": "VEGA-7"}
    extra = expand_context([source], [matching, history, foreign], ("NOVA-3",), False)
    assert [item["id"] for item in extra] == ["linked-table"]


def test_context_budget_deduplicates_whole_records_without_truncating():
    sources = [
        {"id": "one", "text": "1234"},
        {"id": "one", "text": "1234"},
        {"id": "two", "text": "5678"},
    ]
    assert bounded_sources(sources, max_chars=5) == sources[:1]


def test_context_budget_reserves_cross_platform_coverage_before_extra_hits():
    sources = [
        {"id": "a1", "document": "ATLAS-6", "text": "1234"},
        {"id": "a2", "document": "ATLAS-6", "text": "5678"},
        {"id": "n1", "document": "NOVA-3", "text": "1234"},
    ]
    assert [source["id"] for source in bounded_sources(sources, max_chars=8)] == ["a1", "n1"]


def test_historical_appendix_resolves_governing_section_references_to_current_records():
    source = {
        "id": "appendix",
        "document": "NOVA-3",
        "text": "Every figure is superseded. Governing values are in Sections 1, 4 and 8.",
    }
    records = [
        {
            "id": f"page-{page}",
            "document": "NOVA-3",
            "text": f"Current page {page}",
            "page_start": page,
            "page_end": page,
            "historical": False,
        }
        for page in (4, 8, 12)
    ]
    sections = {"NOVA-3": {"overview": (4,), "mounting": (8,), "maintenance": (12,)}}
    actual = referenced_section_candidates([source], records, sections, True)
    assert {item["id"] for item in actual} == {"page-4", "page-8", "page-12"}


def test_shared_part_identifiers_expand_across_component_names_without_scope_leakage():
    summary = {
        "id": "change",
        "document": "TEST",
        "text": "Bush part 1234-Z changed.",
        "historical": False,
    }
    bracket = {**summary, "id": "specification", "text": "Bracket 1234-Z, torque 12 N-m."}
    old = {**bracket, "id": "old", "historical": True}
    other = {**bracket, "id": "foreign", "document": "OTHER"}
    sources = referenced_section_candidates([summary], [summary, bracket, old, other], {}, False)
    assert {item["id"] for item in sources} == {"change", "specification"}


def test_full_candidate_cap_does_not_crowd_out_explicit_expansion():
    sources = [
        {
            "id": f"old-{index}",
            "kind": "table",
            "document": f"doc-{index % 6}",
            "text": "old context",
        }
        for index in range(24)
    ]
    required = [
        {
            "id": f"new-{index}",
            "kind": "table",
            "document": f"doc-{index}",
            "text": "required missing value",
        }
        for index in range(6)
    ]
    selected = merge_expansion(sources, [], required, False)
    assert len(selected) == 24
    assert {item["id"] for item in required} <= {item["id"] for item in selected}


def test_original_requested_figure_survives_expansion_priority():
    figure = {"id": "image", "kind": "figure", "document": "doc", "text": "Original"}
    required = [
        {"id": f"row-{index}", "kind": "table", "document": "doc", "text": "fact"}
        for index in range(30)
    ]
    assert merge_expansion([figure], [], required, True)[0] == figure
    assert merge_expansion([], required + [figure], [], True)[0] == figure
    assert merge_expansion([], [], required + [figure], True)[0] == figure
