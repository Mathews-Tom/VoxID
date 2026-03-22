from __future__ import annotations

import os
import time
from collections import defaultdict
from typing import Any

from fastapi import HTTPException
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response


class RateLimitMiddleware(BaseHTTPMiddleware):
    """Sliding window rate limiter on generation endpoints.

    Config via env:
      VOXID_RATE_LIMIT: max requests per window (default: 60)
      VOXID_RATE_WINDOW: window in seconds (default: 60)

    Only applies to POST /generate* endpoints.
    Returns 429 with Retry-After header when exceeded.
    """

    def __init__(self, app: Any, **kwargs: Any) -> None:
        super().__init__(app, **kwargs)
        self._limit = int(os.environ.get("VOXID_RATE_LIMIT", "60"))
        self._window = int(os.environ.get("VOXID_RATE_WINDOW", "60"))
        self._requests: dict[str, list[float]] = defaultdict(list)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        if not (
            request.method == "POST"
            and request.url.path.startswith("/generate")
        ):
            return await call_next(request)

        key: str
        api_key_header = request.headers.get("X-API-Key")
        if api_key_header:
            key = api_key_header
        elif request.client is not None:
            key = request.client.host
        else:
            key = "unknown"

        now = time.monotonic()
        window_start = now - self._window

        self._requests[key] = [
            t for t in self._requests[key] if t > window_start
        ]

        if len(self._requests[key]) >= self._limit:
            retry_after = int(self._window - (now - self._requests[key][0]))
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded",
                headers={"Retry-After": str(max(1, retry_after))},
            )

        self._requests[key].append(now)
        return await call_next(request)
