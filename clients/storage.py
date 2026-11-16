from __future__ import annotations

from urllib.parse import quote

from clients.azure import AzureSession
from pipelines.budget import ServiceError

STORAGE_HEADERS = {"x-ms-version": "2023-11-03"}


class Storage:
    def __init__(self, session: AzureSession) -> None:
        self.session = session

    async def prepare(self, container: str) -> None:
        self.session.resources.require_write_target("container", container)
        response = await self.session.request(
            "storage",
            "GET",
            f"/{container}",
            params={"restype": "container"},
            headers=STORAGE_HEADERS,
            allowed_statuses=(404,),
        )
        if response.status_code != 404:
            if response.headers.get("x-ms-meta-project") != "decision-assisted-rag":
                raise ServiceError("existing experiment container lacks project ownership metadata")
            if response.headers.get("x-ms-blob-public-access"):
                raise ServiceError("experiment container must not allow public access")
            return
        self.session.resources.require_write_target("container", container)
        await self.session.request(
            "storage",
            "PUT",
            f"/{container}",
            params={"restype": "container"},
            headers={
                **STORAGE_HEADERS,
                "x-ms-meta-project": "decision-assisted-rag",
                "If-None-Match": "*",
            },
        )

    async def upload(self, container: str, key: str, data: bytes, content_type: str) -> None:
        self.session.resources.require_write_target("container", container)
        if any(part in ("", ".", "..") for part in key.split("/")) or key.startswith("/"):
            raise ValueError("invalid experiment artifact key")
        path = f"/{container}/{quote(key, safe='/')}"
        response = await self.session.request(
            "storage", "GET", path, headers=STORAGE_HEADERS, allowed_statuses=(404,)
        )
        if response.status_code != 404:
            if response.content != data:
                raise ServiceError(
                    "immutable experiment artifact already exists with different bytes"
                )
            return
        self.session.resources.require_write_target("container", container)
        await self.session.request(
            "storage",
            "PUT",
            path,
            content=data,
            headers={
                **STORAGE_HEADERS,
                "x-ms-blob-type": "BlockBlob",
                "Content-Type": content_type,
                "If-None-Match": "*",
            },
        )
