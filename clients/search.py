from __future__ import annotations

from clients.azure import AzureSession
from clients.telemetry import payload_hash, span
from ingestion.corpus import PDF_NAMES
from pipelines.budget import ServiceError

SEARCH_API_VERSION = "2025-09-01"
PROJECT_MARKER = "decision-assisted-rag pilot; separate from semantic_rag baseline"


def index_definition(name: str, dimensions: int) -> dict:
    fields = [
        {"name": "id", "type": "Edm.String", "key": True},
        {"name": "document", "type": "Edm.String", "filterable": True},
        {"name": "revision", "type": "Edm.String", "filterable": True},
        {"name": "page_start", "type": "Edm.Int32", "filterable": True},
        {"name": "page_end", "type": "Edm.Int32", "filterable": True},
        {"name": "kind", "type": "Edm.String", "filterable": True},
        {"name": "historical", "type": "Edm.Boolean", "filterable": True},
        {"name": "text", "type": "Edm.String", "searchable": True},
        {"name": "figure_id", "type": "Edm.String", "filterable": True},
        {
            "name": "vector",
            "type": "Collection(Edm.Single)",
            "searchable": True,
            "retrievable": False,
            "dimensions": dimensions,
            "vectorSearchProfile": "experiment-vector",
        },
    ]
    return {
        "name": name,
        "description": PROJECT_MARKER,
        "fields": fields,
        "vectorSearch": {
            "algorithms": [{"name": "experiment-hnsw", "kind": "hnsw"}],
            "profiles": [{"name": "experiment-vector", "algorithm": "experiment-hnsw"}],
        },
        "semantic": {
            "defaultConfiguration": "experiment-semantic",
            "configurations": [
                {
                    "name": "experiment-semantic",
                    "prioritizedFields": {"prioritizedContentFields": [{"fieldName": "text"}]},
                }
            ],
        },
    }


class Search:
    def __init__(self, session: AzureSession) -> None:
        self.session = session
        self.name = session.resources.index_name

    async def prepare(self) -> None:
        with span("search.prepare", service="search", index_hash=payload_hash(self.name)):
            await self._prepare()

    async def _prepare(self) -> None:
        self.session.resources.require_write_target("index", self.name)
        definition = index_definition(self.name, self.session.resources.embedding_dimensions)
        response = await self.session.request(
            "search",
            "GET",
            f"/indexes/{self.name}",
            params={"api-version": SEARCH_API_VERSION},
            allowed_statuses=(404,),
        )
        if response.status_code != 404:
            existing = response.json()
            if existing.get("description") != PROJECT_MARKER:
                raise ServiceError(
                    "experiment index already exists without this project's ownership"
                )
            return
        self.session.resources.require_write_target("index", self.name)
        await self.session.request(
            "search",
            "PUT",
            f"/indexes/{self.name}",
            params={"api-version": SEARCH_API_VERSION},
            headers={"If-None-Match": "*"},
            json=definition,
        )

    async def upload(self, chunks: list[dict]) -> None:
        with span(
            "search.upload", service="search", index_hash=payload_hash(self.name),
            payload_hash=payload_hash(chunks),
        ):
            await self._upload(chunks)

    async def _upload(self, chunks: list[dict]) -> None:
        if not chunks or len(chunks) > 100:
            raise ValueError("pilot upload requires 1-100 chunks")
        self.session.resources.require_write_target("index", self.name)
        response = await self.session.request(
            "search",
            "POST",
            f"/indexes/{self.name}/docs/index",
            params={"api-version": SEARCH_API_VERSION},
            json={"value": [{"@search.action": "upload", **chunk} for chunk in chunks]},
        )
        result = response.json().get("value")
        if (
            not isinstance(result, list)
            or len(result) != len(chunks)
            or any(item.get("status") is not True for item in result)
        ):
            raise ServiceError("experiment index returned a partial or invalid upload result")

    async def query(
        self,
        question: str,
        vector: list[float],
        *,
        historical: bool = False,
        include_figures: bool = False,
        platforms: tuple[str, ...] = (),
    ) -> list[dict]:
        if len(vector) != self.session.resources.embedding_dimensions:
            raise ValueError("query vector dimension does not match the experiment index")
        if platforms:
            allowed = {name.split("_")[0] for name in PDF_NAMES}
            if len(set(platforms)) != len(platforms) or not set(platforms) <= allowed:
                raise ValueError("query scope is outside the configured corpus")
            results = []
            for platform in platforms:
                results.extend(
                    await self._query_platform(
                        question, vector, historical, include_figures, platform
                    )
                )
            return results
        return await self._query_platform(question, vector, historical, include_figures, None)

    async def _query_platform(
        self,
        question: str,
        vector: list[float],
        historical: bool,
        include_figures: bool,
        platform: str | None,
    ) -> list[dict]:
        if historical and include_figures:
            results = []
            for previous, figures, top in (
                (True, True, 1),
                (True, False, 2),
                (False, True, 1),
                (False, False, 1),
            ):
                results.extend(
                    await self._query_scope(
                        question,
                        vector,
                        historical=previous,
                        top=top,
                        figures=figures,
                        platform=platform,
                    )
                )
            return results
        if historical:
            previous = await self._query_scope(
                question, vector, historical=True, top=3, platform=platform
            )
            current = await self._query_scope(
                question, vector, historical=False, top=2, platform=platform
            )
            return previous + current
        if include_figures:
            figures = await self._query_scope(
                question, vector, historical=False, top=2, figures=True, platform=platform
            )
            context = await self._query_scope(
                question, vector, historical=False, top=3, figures=False, platform=platform
            )
            return figures + context
        return await self._query_scope(question, vector, historical=False, top=5, platform=platform)

    async def _query_scope(
        self,
        question: str,
        vector: list[float],
        *,
        historical: bool,
        top: int,
        figures: bool | None = None,
        platform: str | None = None,
    ) -> list[dict]:
        with span(
            "search.query", service="search", index_hash=payload_hash(self.name),
            api_version=SEARCH_API_VERSION, question_hash=payload_hash(question),
            vector_hash=payload_hash(vector), historical=historical, top=top,
            figures=figures, platform=platform,
        ) as attributes:
            results = await self._query_scope_impl(
                question, vector, historical=historical, top=top, figures=figures,
                platform=platform,
            )
            attributes["result_count"] = len(results)
            return results

    async def _query_scope_impl(
        self,
        question: str,
        vector: list[float],
        *,
        historical: bool,
        top: int,
        figures: bool | None = None,
        platform: str | None = None,
    ) -> list[dict]:
        scope = "historical eq true" if historical else "historical eq false"
        if figures is not None:
            scope += " and kind " + ("eq" if figures else "ne") + " 'figure'"
        if platform is not None:
            scope += f" and document eq '{platform}'"
        response = await self.session.request(
            "search",
            "POST",
            f"/indexes/{self.name}/docs/search",
            params={"api-version": SEARCH_API_VERSION},
            json={
                "search": question,
                "queryType": "semantic",
                "semanticConfiguration": "experiment-semantic",
                "vectorQueries": [
                    {"kind": "vector", "vector": vector, "fields": "vector", "k": 50}
                ],
                "filter": scope,
                "top": top,
                "select": "id,document,revision,page_start,page_end,kind,text,figure_id,historical",
            },
        )
        with span("search.parse", service="search"):
            results = response.json().get("value")
            if not isinstance(results, list):
                raise ServiceError("experiment search response is invalid")
            if any(
                not isinstance(item, dict) or not isinstance(item.get("id"), str)
                for item in results
            ):
                raise ServiceError("experiment search returned invalid evidence records")
            return results
