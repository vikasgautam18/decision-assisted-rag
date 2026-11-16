from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from pipelines.contracts import Artifact, Identifier

Sha256 = Annotated[str, StringConstraints(pattern=r"^[a-f0-9]{64}$")]


class Region(Artifact):
    page_number: int = Field(ge=1)
    polygon: tuple[float, ...] = Field(min_length=8)

    @model_validator(mode="after")
    def coordinate_pairs(self) -> Self:
        if len(self.polygon) % 2:
            raise ValueError("polygon must contain coordinate pairs")
        return self


class Span(Artifact):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)


class TableCell(Artifact):
    row: int = Field(ge=0)
    column: int = Field(ge=0)
    row_span: int = Field(default=1, ge=1)
    column_span: int = Field(default=1, ge=1)
    kind: Literal["content", "rowHeader", "columnHeader", "stubHead", "description"]
    text: str
    regions: tuple[Region, ...]
    spans: tuple[Span, ...]


class TableArtifact(Artifact):
    table_id: Identifier
    row_count: int = Field(ge=1)
    column_count: int = Field(ge=1)
    cells: tuple[TableCell, ...] = Field(min_length=1)
    regions: tuple[Region, ...] = Field(min_length=1)
    caption: str | None = None
    footnotes: tuple[str, ...] = ()
    element_refs: tuple[str, ...] = ()
    spans: tuple[Span, ...]
    context_policy: Literal["preserve_full_table_and_document_context"] = (
        "preserve_full_table_and_document_context"
    )

    @model_validator(mode="after")
    def valid_grid(self) -> Self:
        occupied = set()
        for cell in self.cells:
            if (
                cell.row + cell.row_span > self.row_count
                or cell.column + cell.column_span > self.column_count
            ):
                raise ValueError("cell extends outside declared table dimensions")
            for row in range(cell.row, cell.row + cell.row_span):
                for column in range(cell.column, cell.column + cell.column_span):
                    if (row, column) in occupied:
                        raise ValueError("table cells overlap")
                    occupied.add((row, column))
        return self


class ParagraphArtifact(Artifact):
    paragraph_id: Identifier
    text: str
    role: str | None = None
    regions: tuple[Region, ...] = Field(min_length=1)
    spans: tuple[Span, ...]


class FigureArtifact(Artifact):
    figure_id: Identifier
    provider_figure_id: Identifier
    regions: tuple[Region, ...] = Field(min_length=1)
    caption: str | None = None
    footnotes: tuple[str, ...] = ()
    element_refs: tuple[str, ...] = ()
    spans: tuple[Span, ...]
    image_status: Literal["not_rendered"] = "not_rendered"
    description_status: Literal["not_generated"] = "not_generated"


class PageArtifact(Artifact):
    page_number: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: Literal["inch", "pixel"]


class LayoutArtifact(Artifact):
    document: Identifier
    revision: Identifier
    pdf_sha256: Sha256
    analysis_sha256: Sha256
    model_id: Literal["prebuilt-layout"] = "prebuilt-layout"
    api_version: Identifier
    content: str = Field(min_length=1)
    pages: tuple[PageArtifact, ...] = Field(min_length=1)
    tables: tuple[TableArtifact, ...]
    figures: tuple[FigureArtifact, ...]
    paragraphs: tuple[ParagraphArtifact, ...]
    issues: tuple[str, ...]
    index_ready: Literal[False] = False


class ProviderModel(BaseModel):
    model_config = ConfigDict(extra="ignore", strict=True, allow_inf_nan=False)


class DIRegion(ProviderModel):
    pageNumber: int = Field(ge=1)
    polygon: list[float] = Field(min_length=8)


class DISpan(ProviderModel):
    offset: int = Field(ge=0)
    length: int = Field(ge=0)


class DICaption(ProviderModel):
    content: str


class DICell(ProviderModel):
    rowIndex: int = Field(ge=0)
    columnIndex: int = Field(ge=0)
    rowSpan: int = Field(default=1, ge=1)
    columnSpan: int = Field(default=1, ge=1)
    kind: Literal["content", "rowHeader", "columnHeader", "stubHead", "description"] = "content"
    content: str
    boundingRegions: list[DIRegion] = Field(default_factory=list)
    spans: list[DISpan] = Field(default_factory=list)


class DITable(ProviderModel):
    rowCount: int = Field(ge=1)
    columnCount: int = Field(ge=1)
    cells: list[DICell] = Field(min_length=1)
    boundingRegions: list[DIRegion] = Field(min_length=1)
    spans: list[DISpan] = Field(default_factory=list)
    caption: DICaption | None = None
    footnotes: list[DICaption] = Field(default_factory=list)
    elements: list[str] = Field(default_factory=list)


class DIFigure(ProviderModel):
    id: str
    boundingRegions: list[DIRegion] = Field(min_length=1)
    spans: list[DISpan] = Field(default_factory=list)
    caption: DICaption | None = None
    footnotes: list[DICaption] = Field(default_factory=list)
    elements: list[str] = Field(default_factory=list)


class DIParagraph(ProviderModel):
    content: str
    role: str | None = None
    boundingRegions: list[DIRegion] = Field(min_length=1)
    spans: list[DISpan] = Field(default_factory=list)


class DIPage(ProviderModel):
    pageNumber: int = Field(ge=1)
    width: float = Field(gt=0)
    height: float = Field(gt=0)
    unit: Literal["inch", "pixel"]


class DIAnalysis(ProviderModel):
    apiVersion: str
    modelId: Literal["prebuilt-layout"]
    stringIndexType: Literal["unicodeCodePoint"]
    content: str = Field(min_length=1)
    pages: list[DIPage] = Field(min_length=1)
    tables: list[DITable] = Field(default_factory=list)
    figures: list[DIFigure] = Field(default_factory=list)
    paragraphs: list[DIParagraph] = Field(default_factory=list)
