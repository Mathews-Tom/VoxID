from __future__ import annotations

from fastapi import APIRouter

import voxid
from voxid.api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(version=voxid.__version__)
