from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from .auth import APIKeyMiddleware
from .rate_limit import RateLimitMiddleware
from .routes import all_routers
from .routes.serving import router as serving_router

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).parent / "static" / "enrollment"
_INDEX_HTML = _STATIC_DIR / "index.html"


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start/stop the GPUDispatcher if a serving config is set."""
    from .deps import get_dispatcher

    dispatcher = get_dispatcher()
    if dispatcher is not None:
        await dispatcher.start()
        logger.info("GPUDispatcher started with %d workers", len(dispatcher.workers))

    yield

    if dispatcher is not None:
        await dispatcher.stop()
        logger.info("GPUDispatcher stopped")


def create_app() -> FastAPI:
    app = FastAPI(
        title="VoxID",
        description="Voice Identity Management Platform API",
        version="0.2.0",
        lifespan=_lifespan,
    )

    # Middleware — outermost first (rate limit wraps auth wraps app)
    app.add_middleware(RateLimitMiddleware)  # type: ignore[arg-type]
    app.add_middleware(APIKeyMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # All API routes under /api prefix
    for router in all_routers:
        app.include_router(router, prefix="/api")

    # Serving health endpoint (multi-GPU)
    app.include_router(serving_router, prefix="/api/v1")

    # UI routes — serve index.html for client-side routing
    if _INDEX_HTML.exists():

        @app.get("/", include_in_schema=False)
        async def ui_root() -> FileResponse:
            return FileResponse(_INDEX_HTML)

        @app.get("/dashboard", include_in_schema=False)
        async def ui_dashboard() -> FileResponse:
            return FileResponse(_INDEX_HTML)

        @app.get("/enrollment", include_in_schema=False)
        async def ui_enrollment() -> FileResponse:
            return FileResponse(_INDEX_HTML)

        @app.get("/generate", include_in_schema=False)
        async def ui_generate() -> FileResponse:
            return FileResponse(_INDEX_HTML)

        @app.get("/identity/{identity_id}", include_in_schema=False)
        async def ui_identity_detail(identity_id: str) -> FileResponse:
            return FileResponse(_INDEX_HTML)

    # Static assets (CSS, JS, worklets) — mounted last
    if _STATIC_DIR.is_dir():
        app.mount("/assets", StaticFiles(directory=_STATIC_DIR), name="assets")

    return app
