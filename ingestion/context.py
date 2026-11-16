from __future__ import annotations

from pydantic import Field

from ingestion.layout import LayoutArtifact, Sha256
from pipelines.contracts import Artifact, Identifier, Text

SECTION_NAMES = {
    "overview",
    "powertrain",
    "homologation",
    "mounting",
    "suppliers",
    "warranty",
    "thermal",
    "maintenance",
    "history",
}


class SourceContext(Artifact):
    document: Identifier
    revision: Identifier
    pdf_sha256: Sha256
    review_status: Text
    historical_pages: tuple[int, ...]
    table_continuations: dict[str, str]
    sections: dict[str, tuple[int, ...]] = Field(min_length=9, max_length=9)
    provenance: Text

    def validate_layout(self, layout: LayoutArtifact) -> None:
        if (self.document, self.revision, self.pdf_sha256) != (
            layout.document,
            layout.revision,
            layout.pdf_sha256,
        ):
            raise ValueError("reviewed context does not match the original PDF identity")
        pages = {page.page_number for page in layout.pages}
        if set(self.sections) != SECTION_NAMES or any(
            not values or not set(values) <= pages or any(type(page) is not int for page in values)
            for values in self.sections.values()
        ):
            raise ValueError("reviewed sections must name valid physical pages")
        if not set(self.historical_pages) <= pages:
            raise ValueError("historical page is outside the document")
        tables = {table.table_id for table in layout.tables}
        if (
            not set(self.table_continuations) <= tables
            or not set(self.table_continuations.values()) <= tables
        ):
            raise ValueError("reviewed continuation references an unknown table")
