from __future__ import annotations

import asyncio
import base64
import json

from clients.azure import AzureSession
from pipelines.budget import ServiceError


async def analyze_pdf(session: AzureSession, pdf: bytes, *, pages: int) -> bytes:
    if not pdf.startswith(b"%PDF-") or not 1 <= pages <= 20:
        raise ValueError("approved analysis requires a PDF and at most 20 pages")
    async with asyncio.timeout(300):
        response = await session.request(
            "di",
            "POST",
            "/documentintelligence/documentModels/prebuilt-layout:analyze",
            params={
                "api-version": "2024-11-30",
                "outputContentFormat": "markdown",
                "stringIndexType": "unicodeCodePoint",
                "pages": f"1-{pages}",
            },
            json={"base64Source": base64.b64encode(pdf).decode()},
        )
        operation_url = response.headers.get("operation-location")
        if response.status_code != 202 or not operation_url:
            raise ServiceError("DI did not return an accepted analysis operation")
        path = session.operation_path("di", operation_url)
        while True:
            await asyncio.sleep(2)
            response = await session.request("di", "GET", path)
            try:
                body = response.json()
            except ValueError as exc:
                raise ServiceError("DI poll response is not JSON") from exc
            if not isinstance(body, dict):
                raise ServiceError("DI poll response is not an object")
            status = body.get("status")
            if status == "succeeded":
                result = body.get("analyzeResult")
                if not isinstance(result, dict):
                    raise ServiceError("DI succeeded without an analyzeResult")
                return json.dumps(result, ensure_ascii=False).encode()
            if status not in ("running", "notStarted"):
                raise ServiceError("DI analysis failed or returned an unknown status")
