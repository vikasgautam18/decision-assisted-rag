import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def install_api_boundary(application: FastAPI) -> None:
    @application.middleware("http")
    async def api_boundary(request: Request, call_next):
        if request.url.path.startswith("/api/"):
            origin = request.headers.get("origin")
            fetch_site = request.headers.get("sec-fetch-site")
            if (
                origin is not None
                and origin != str(request.base_url).rstrip("/")
                or fetch_site == "cross-site"
                or (
                    request.method not in ("GET", "HEAD", "OPTIONS")
                    and origin is None
                    and fetch_site != "same-origin"
                )
            ):
                logger.warning("api_request_rejected reason=cross_origin")
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Cross-origin API requests are not allowed."},
                    headers={"Cache-Control": "no-store"},
                )
        response = await call_next(request)
        response.headers["Cache-Control"] = "no-store"
        return response
