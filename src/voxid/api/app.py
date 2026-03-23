from __future__ import annotations

from fastapi import FastAPI

from .auth import APIKeyMiddleware
from .rate_limit import RateLimitMiddleware
from .routes import all_routers


def create_app() -> FastAPI:
    app = FastAPI(
        title="VoxID",
        description="Voice Identity Management Platform API",
        version="0.2.0",
    )

    # Middleware — outermost first (rate limit wraps auth wraps app)
    app.add_middleware(RateLimitMiddleware)  # type: ignore[arg-type]
    app.add_middleware(APIKeyMiddleware)

    for router in all_routers:
        app.include_router(router)

    return app
