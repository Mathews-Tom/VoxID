from __future__ import annotations

import os

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_EXEMPT_PATHS = frozenset({"/health", "/docs", "/openapi.json"})


class APIKeyMiddleware(BaseHTTPMiddleware):
    """Validate API key from X-API-Key header or api_key query param.

    If VOXID_API_KEY env var is not set, auth is disabled (open access).
    Health, docs, and openapi endpoints are always exempt.
    """

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        api_key = os.environ.get("VOXID_API_KEY")
        if api_key is None:
            return await call_next(request)

        if request.url.path in _EXEMPT_PATHS:
            return await call_next(request)

        provided = request.headers.get("X-API-Key") or request.query_params.get(
            "api_key"
        )
        if provided != api_key:
            raise HTTPException(status_code=401, detail="Invalid or missing API key")

        return await call_next(request)
