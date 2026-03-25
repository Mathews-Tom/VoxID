from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.cors import CORSMiddleware

from .auth import APIKeyMiddleware
from .rate_limit import RateLimitMiddleware
from .routes import all_routers

_STATIC_DIR = Path(__file__).parent / "static" / "enrollment"
_INDEX_HTML = _STATIC_DIR / "index.html"


def create_app() -> FastAPI:
    app = FastAPI(
        title="VoxID",
        description="Voice Identity Management Platform API",
        version="0.2.0",
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
