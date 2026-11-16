import base64
import copy
import hashlib
import json

import pytest
from pydantic import ValidationError

from ingestion import corpus
from ingestion.normalize import normalize_layout


@pytest.fixture
def analysis():
    region = {"pageNumber": 1, "polygon": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]}
    second_page = {**region, "pageNumber": 2}
    return {
        "apiVersion": "2024-11-30",
        "modelId": "prebuilt-layout",
        "stringIndexType": "unicodeCodePoint",
        "content": "Platform Electric Diesel Range 120 90 a. Qualified value. Figure 1",
        "pages": [{"pageNumber": n, "width": 8.5, "height": 11.0, "unit": "inch"} for n in (1, 2)],
        "tables": [
            {
                "rowCount": 2,
                "columnCount": 3,
                "boundingRegions": [region, second_page],
                "spans": [{"offset": 0, "length": 35}],
                "cells": [
                    {
                        "rowIndex": 0,
                        "columnIndex": i,
                        "kind": "columnHeader",
                        "content": value,
                        "boundingRegions": [region],
                    }
                    for i, value in enumerate(("Platform", "Electric", "Diesel"))
                ]
                + [
                    {
                        "rowIndex": 1,
                        "columnIndex": i,
                        "content": value,
                        "boundingRegions": [second_page],
                    }
                    for i, value in enumerate(("Range", "120", "90"))
                ],
                "footnotes": [{"content": "a. Qualified value."}],
            }
        ],
        "paragraphs": [
            {
                "content": "a. Qualified value.",
                "boundingRegions": [second_page],
                "spans": [{"offset": 38, "length": 19}],
            }
        ],
        "figures": [
            {
                "id": "2.1",
                "boundingRegions": [second_page],
                "caption": {"content": "Figure 1"},
                "elements": ["/paragraphs/0"],
            }
        ],
    }


def normalize(analysis):
    return normalize_layout(
        json.dumps(analysis).encode(),
        document="DEMO-1",
        revision="A",
        pdf_sha256="a" * 64,
    )


def test_preserve_cells_headers_footnotes_and_cross_page_provenance(analysis):
    result = normalize(analysis)
    table = result.tables[0]
    assert table.cells[1].text == "Electric"
    assert table.cells[1].kind == "columnHeader"
    assert table.cells[4].text == "120" and table.cells[4].column == 1
    assert table.cells[4].regions[0].page_number == 2
    assert table.footnotes == ("a. Qualified value.",)
    assert result.paragraphs[0].text == "a. Qualified value."
    assert result.pdf_sha256 == "a" * 64
    assert result.index_ready is False


def test_preserve_figure_location_and_explicit_missing_image_status(analysis):
    result = normalize(analysis)
    figure = result.figures[0]
    assert figure.caption == "Figure 1"
    assert figure.provider_figure_id == "2.1"
    assert figure.regions[0].page_number == 2
    assert figure.element_refs == ("/paragraphs/0",)
    assert figure.image_status == "not_rendered"
    assert figure.description_status == "not_generated"


def test_preserve_merged_cells_without_flattening(analysis):
    table = analysis["tables"][0]
    table["cells"] = [
        {"rowIndex": 0, "columnIndex": 0, "columnSpan": 3, "content": "Merged"},
        *table["cells"][3:],
    ]
    result = normalize(analysis)
    assert result.tables[0].cells[0].column_span == 3
    assert any("lack page regions" in issue for issue in result.issues)


def test_missing_header_flagged_not_invented_from_previous_table(analysis):
    continuation = copy.deepcopy(analysis["tables"][0])
    for cell in continuation["cells"]:
        cell.pop("kind", None)
    analysis["tables"].append(continuation)
    result = normalize(analysis)
    assert any("table-2: missing explicit column headers" in issue for issue in result.issues)
    assert result.tables[1].cells[0].kind == "content"


@pytest.mark.parametrize("problem", ["overlap", "outside", "missing_page", "span", "unknown_model"])
def test_invalid_layout_rejected(analysis, problem):
    if problem == "overlap":
        analysis["tables"][0]["cells"][1]["columnIndex"] = 0
    elif problem == "outside":
        analysis["tables"][0]["cells"][0]["rowSpan"] = 3
    elif problem == "missing_page":
        analysis["figures"][0]["boundingRegions"][0]["pageNumber"] = 99
    elif problem == "span":
        analysis["paragraphs"][0]["spans"][0]["offset"] = 10000
    else:
        analysis["modelId"] = "unexpected-model"
    with pytest.raises((ValueError, ValidationError)):
        normalize(analysis)


def test_unknown_offset_encoding_is_not_silently_misread(analysis):
    analysis["stringIndexType"] = "utf16CodeUnit"
    with pytest.raises(ValidationError):
        normalize(analysis)


def test_preserve_exact_content_for_unicode_codepoint_spans(analysis):
    analysis["content"] = "  \U0001f600 Original content  "
    analysis["tables"] = []
    analysis["figures"] = []
    analysis["paragraphs"] = [
        {
            "content": "Original content",
            "boundingRegions": [
                {"pageNumber": 1, "polygon": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]}
            ],
            "spans": [{"offset": 4, "length": 16}],
        }
    ]
    result = normalize(analysis)
    span = result.paragraphs[0].spans[0]
    assert result.content == analysis["content"]
    assert result.content[span.offset : span.offset + span.length] == "Original content"


def test_blank_paragraphs_preserve_reference_indices_and_surface_issue(analysis):
    analysis["paragraphs"].insert(
        0,
        {
            "content": " ",
            "boundingRegions": [
                {"pageNumber": 1, "polygon": [0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0]}
            ],
        },
    )
    result = normalize(analysis)
    assert result.paragraphs[0].paragraph_id == "/paragraphs/0"
    assert result.paragraphs[0].text == " "
    assert result.paragraphs[1].paragraph_id == "/paragraphs/1"
    assert any("blank text preserved" in issue for issue in result.issues)


def test_unmodeled_provider_fields_are_disclosed(analysis):
    analysis["styles"] = [{"isHandwritten": False}]
    result = normalize(analysis)
    assert any("styles" in issue and "raw artifact" in issue for issue in result.issues)


def test_real_di_empty_cell_zero_length_span_is_preserved(analysis):
    cell = analysis["tables"][0]["cells"][1]
    cell["content"] = ""
    cell["spans"] = [{"offset": 4, "length": 0}]
    result = normalize(analysis)
    assert result.tables[0].cells[1].text == ""
    assert result.tables[0].cells[1].spans[0].length == 0


def fake_github(monkeypatch, *, corrupt=False):
    paths = [*(f"docs/{name}" for name in corpus.PDF_NAMES), "LICENSE"]
    blobs = {}
    entries = []
    for path in paths:
        data = b"MIT license fixture" if path == "LICENSE" else b"%PDF-1.7 " + path.encode()
        sha = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
        entries.append({"path": path, "type": "blob", "sha": sha, "size": len(data)})
        blobs[sha] = {
            "encoding": "base64",
            "content": base64.b64encode(b"corrupt" if corrupt else data).decode(),
        }

    def response(path):
        if "/git/trees/" in path:
            return {"tree": entries, "truncated": False}
        return blobs[path.rsplit("/", 1)[1]]

    monkeypatch.setattr(corpus, "_github_json", response)


def test_corpus_hash_inventory_and_license_preserved(monkeypatch, tmp_path):
    fake_github(monkeypatch)
    output = corpus.fetch_pinned_corpus(tmp_path / "corpus")
    manifest = json.loads(output.read_text())
    assert manifest["commit"] == corpus.REFERENCE_COMMIT
    assert manifest["analysis_performed"] is False
    assert manifest["index_created"] is False
    assert len(manifest["artifacts"]) == 7
    assert (output.parent / "LICENSE").read_text() == "MIT license fixture"
    for artifact in manifest["artifacts"]:
        data = (output.parent / artifact["path"]).read_bytes()
        assert hashlib.sha256(data).hexdigest() == artifact["sha256"]


def test_corrupt_blob_leaves_no_partial_corpus(monkeypatch, tmp_path):
    fake_github(monkeypatch, corrupt=True)
    with pytest.raises(corpus.CorpusError, match="identity mismatch"):
        corpus.fetch_pinned_corpus(tmp_path / "corpus")
    assert not (tmp_path / "corpus").exists()


def test_existing_destination_never_overwritten(tmp_path):
    (tmp_path / "keep.pdf").write_bytes(b"existing")
    with pytest.raises(corpus.CorpusError, match="will not be overwritten"):
        corpus.fetch_pinned_corpus(tmp_path)
    assert (tmp_path / "keep.pdf").read_bytes() == b"existing"
