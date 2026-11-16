from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from datetime import UTC, datetime
from email.utils import parsedate_to_datetime
from typing import Literal, Protocol
from urllib.parse import urlsplit

import httpx
from azure.core.exceptions import ClientAuthenticationError

from app.resources import ExperimentResources
from clients.rate import ModelAdmission
from clients.telemetry import payload_hash, response_ids, span
from pipelines.budget import ServiceError

logger = logging.getLogger(__name__)
Service = Literal["ai", "di", "search", "storage"]


class Token(Protocol):
    token: str


class Credential(Protocol):
    async def get_token(self, *scopes: str) -> Token: ...


class AzureSession:
    def __init__(
        self,
        resources: ExperimentResources,
        credential: Credential,
        client: httpx.AsyncClient,
        *,
        deployment_capacities: Mapping[str, tuple[int, int]] | None = None,
    ) -> None:
        self.resources = resources
        self.credential = credential
        self.client = client
        self.admission = ModelAdmission(deployment_capacities=deployment_capacities)

    def endpoint(self, service: Service) -> str:
        return {
            "ai": self.resources.ai_endpoint,
            "di": self.resources.di_endpoint,
            "search": self.resources.search_endpoint,
            "storage": f"https://{self.resources.storage_account}.blob.core.windows.net",
        }[service]

    async def request(
        self,
        service: Service,
        method: str,
        path: str,
        *,
        allowed_statuses: tuple[int, ...] = (),
        headers: dict[str, str] | None = None,
        **kwargs,
    ) -> httpx.Response:
        with span(
            "azure.request", service=service, method=method, path_hash=payload_hash(path),
            endpoint_hash=payload_hash(self.endpoint(service)),
            transport_timing_status="unavailable",
        ):
            return await self._request(
                service, method, path, allowed_statuses=allowed_statuses, headers=headers,
                **kwargs,
            )

    async def _request(
        self, service: Service, method: str, path: str, *,
        allowed_statuses: tuple[int, ...], headers: dict[str, str] | None, **kwargs,
    ) -> httpx.Response:
        if not path.startswith("/") or path.startswith("//") or "://" in path:
            raise ValueError("request must use a service-relative path")
        scope = {
            "ai": "https://cognitiveservices.azure.com/.default",
            "di": "https://cognitiveservices.azure.com/.default",
            "search": "https://search.azure.com/.default",
            "storage": "https://storage.azure.com/.default",
        }[service]
        with span("azure.admission", service=service) as admission:
            window = (
                await self.admission.acquire(path, kwargs.get("json", {}))
                if service == "ai" else None
            )
            admission["admitted"] = True
        try:
            with span("azure.authentication", service=service):
                token = await self.credential.get_token(scope)
            with span(
                "azure.transport", service=service,
                payload_hash=payload_hash(kwargs.get("json")),
            ) as transport:
                request_headers = dict(headers or {})
                if "span_id" in transport and not any(
                    key.lower() == "x-ms-client-request-id" for key in request_headers
                ):
                    request_headers["x-ms-client-request-id"] = transport["span_id"]
                transport["client_request_id"] = next(
                    (value for key, value in request_headers.items()
                     if key.lower() == "x-ms-client-request-id"), None
                )
                response = await self.client.request(
                    method,
                    self.endpoint(service) + path,
                    headers={**request_headers, "Authorization": f"Bearer {token.token}"},
                    follow_redirects=False,
                    **kwargs,
                )
                try:
                    request_content = response.request.content
                except httpx.RequestNotRead:
                    request_content = None
                transport.update(
                    http_status=response.status_code,
                    request_ids=response_ids(response.headers),
                    response_bytes=len(response.content),
                    request_bytes=len(request_content) if request_content is not None else None,
                    payload_hash=(
                        hashlib.sha256(request_content).hexdigest()
                        if request_content is not None else None
                    ),
                )
                if response.status_code not in allowed_statuses:
                    response.raise_for_status()
                return response
        except ClientAuthenticationError:
            logger.warning("azure_request_failed service=%s reason=authentication", service)
            raise ServiceError(f"{service} server authentication failed") from None
        except httpx.HTTPStatusError as exc:
            retry_after = None
            if exc.response.status_code == 429:
                value = exc.response.headers.get("retry-after", "65")
                try:
                    retry_after = float(value)
                except ValueError:
                    try:
                        retry_after = (
                            parsedate_to_datetime(value) - datetime.now(UTC)
                        ).total_seconds()
                    except (ValueError, TypeError, OverflowError):
                        retry_after = 65.0
                if not math.isfinite(retry_after):
                    retry_after = 65.0
                retry_after = max(65.0, retry_after)
                if window is not None:
                    window.throttle(retry_after)
            logger.warning(
                "azure_request_failed service=%s status=%s request_id=%s",
                service,
                exc.response.status_code,
                exc.response.headers.get("x-ms-request-id", "unknown"),
            )
            error = ServiceError(
                f"{service} returned HTTP {exc.response.status_code}",
                status_code=exc.response.status_code,
                retry_after=retry_after,
            )
            error.request_ids = response_ids(exc.response.headers)
            raise error from None
        except httpx.RequestError as exc:
            logger.warning(
                "azure_request_failed service=%s reason=transport error_type=%s",
                service,
                type(exc).__name__,
            )
            raise ServiceError(f"{service} transport failed") from None

    def operation_path(self, service: Service, operation_url: str) -> str:
        parsed = urlsplit(operation_url)
        expected = urlsplit(self.endpoint(service))
        if (
            parsed.scheme != expected.scheme
            or parsed.netloc != expected.netloc
            or parsed.fragment
            or parsed.username is not None
            or not parsed.path.startswith(
                "/documentintelligence/documentModels/prebuilt-layout/analyzeResults/"
            )
        ):
            raise ServiceError("DI operation URL is outside the configured analysis endpoint")
        return parsed.path + ("?" + parsed.query if parsed.query else "")
