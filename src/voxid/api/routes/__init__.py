from __future__ import annotations

from fastapi import APIRouter

from .generate import router as generate_router
from .health import router as health_router
from .identities import router as identities_router
from .route import router as route_router

all_routers: list[APIRouter] = [
    identities_router,
    generate_router,
    route_router,
    health_router,
]

__all__ = ["all_routers"]
