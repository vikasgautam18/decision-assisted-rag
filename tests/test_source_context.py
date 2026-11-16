import json
from pathlib import Path

import pytest

from ingestion.context import SECTION_NAMES, SourceContext
from ingestion.corpus import DOSSIERS
from pipelines.evidence import section_candidates


@pytest.mark.parametrize("document", DOSSIERS)
def test_reviewed_section_maps_select_every_document_section(document):
    root = Path(__file__).parents[1]
    context = SourceContext.model_validate_json(
        (root / f"manifests/{document}-context.json").read_bytes()
    )
    assert set(context.sections) == SECTION_NAMES
    expected = {
        "overview": (4,),
        "powertrain": (5,),
        "homologation": (6, 7),
        "mounting": (8,),
        "suppliers": (9,),
        "warranty": (10,),
        "thermal": (11,),
        "maintenance": (12,),
        "history": (13,),
    }
    if document == "VEGA-7":
        expected = {
            name: (
                (6, 7, 8)
                if name == "homologation"
                else tuple(page + 1 if page >= 8 else page for page in pages)
            )
            for name, pages in expected.items()
        }
    assert context.sections == expected
    records = [
        {
            "id": f"page-{page}",
            "document": document,
            "page_start": page,
            "page_end": page,
            "historical": page in context.historical_pages,
        }
        for page in range(1, DOSSIERS[document][1] + 1)
    ]
    for section, pages in expected.items():
        actual = section_candidates(records, document, section, True, {document: context.sections})
        assert [item["page_start"] for item in actual] == list(pages)
    assert (
        section_candidates(records, document, "history", False, {document: context.sections}) == []
    )


def test_unknown_section_mapping_is_not_silently_empty():
    with pytest.raises(ValueError, match="unavailable"):
        section_candidates([], "NOVA-3", "overview", False, {})


def test_context_source_identity_mismatch_fails():
    root = Path(__file__).parents[1]
    raw = json.loads((root / "manifests/NOVA-3-context.json").read_text())
    context = SourceContext.model_validate_json(json.dumps(raw))

    class OtherLayout:
        document = "VEGA-7"
        revision = "C"
        pdf_sha256 = raw["pdf_sha256"]

    with pytest.raises(ValueError, match="identity"):
        context.validate_layout(OtherLayout())
