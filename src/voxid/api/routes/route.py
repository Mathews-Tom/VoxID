from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException

from voxid.api.deps import get_voxid
from voxid.api.models import RouteRequest, RouteResponse
from voxid.core import VoxID

router = APIRouter(tags=["route"])


@router.post("/route", response_model=RouteResponse)
async def route_text(
    req: RouteRequest,
    vox: VoxID = Depends(get_voxid),
) -> RouteResponse:
    try:
        result = vox.route(req.text, req.identity_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return RouteResponse(**result)
