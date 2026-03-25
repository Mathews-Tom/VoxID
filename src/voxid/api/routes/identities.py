from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from voxid.api.deps import get_voxid
from voxid.api.models import (
    AddStyleRequest,
    CreateIdentityRequest,
    IdentityListResponse,
    IdentityResponse,
    StyleListResponse,
    StyleResponse,
)
from voxid.core import VoxID

router = APIRouter(prefix="/identities", tags=["identities"])


def _identity_not_found(identity_id: str) -> HTTPException:
    return HTTPException(
        status_code=404, detail=f"Identity {identity_id!r} not found"
    )


@router.post("", response_model=IdentityResponse, status_code=201)
async def create_identity(
    req: CreateIdentityRequest,
    vox: VoxID = Depends(get_voxid),
) -> IdentityResponse:
    try:
        identity = vox.create_identity(
            id=req.id,
            name=req.name,
            description=req.description,
            default_style=req.default_style,
            metadata=req.metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return IdentityResponse(
        id=identity.id,
        name=identity.name,
        description=identity.description,
        default_style=identity.default_style,
        created_at=identity.created_at,
        metadata=identity.metadata,
    )


@router.get("", response_model=IdentityListResponse)
async def list_identities(
    vox: VoxID = Depends(get_voxid),
) -> IdentityListResponse:
    return IdentityListResponse(identities=vox.list_identities())


@router.get("/{identity_id}", response_model=IdentityResponse)
async def get_identity(
    identity_id: str,
    vox: VoxID = Depends(get_voxid),
) -> IdentityResponse:
    try:
        identity = vox._store.get_identity(identity_id)
    except FileNotFoundError as exc:
        raise _identity_not_found(identity_id) from exc
    return IdentityResponse(
        id=identity.id,
        name=identity.name,
        description=identity.description,
        default_style=identity.default_style,
        created_at=identity.created_at,
        metadata=identity.metadata,
    )


@router.delete("/{identity_id}", status_code=204)
async def delete_identity(
    identity_id: str,
    vox: VoxID = Depends(get_voxid),
) -> None:
    try:
        vox._store.delete_identity(identity_id)
    except FileNotFoundError as exc:
        raise _identity_not_found(identity_id) from exc


@router.post(
    "/{identity_id}/styles",
    response_model=StyleResponse,
    status_code=201,
)
async def add_style(
    identity_id: str,
    req: AddStyleRequest,
    vox: VoxID = Depends(get_voxid),
) -> StyleResponse:
    try:
        style = vox.add_style(
            identity_id=identity_id,
            id=req.id,
            label=req.label,
            description=req.description,
            ref_audio=req.ref_audio_path,
            ref_text=req.ref_text,
            engine=req.engine,
            language=req.language,
            metadata=req.metadata,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return StyleResponse(
        id=style.id,
        identity_id=style.identity_id,
        label=style.label,
        description=style.description,
        default_engine=style.default_engine,
        language=style.language,
        metadata=style.metadata,
    )


@router.get("/{identity_id}/styles", response_model=StyleListResponse)
async def list_styles(
    identity_id: str,
    vox: VoxID = Depends(get_voxid),
) -> StyleListResponse:
    try:
        vox._store.get_identity(identity_id)
    except FileNotFoundError as exc:
        raise _identity_not_found(identity_id) from exc
    return StyleListResponse(styles=vox.list_styles(identity_id))


@router.get("/{identity_id}/styles/{style_id}", response_model=StyleResponse)
async def get_style(
    identity_id: str,
    style_id: str,
    vox: VoxID = Depends(get_voxid),
) -> StyleResponse:
    try:
        style = vox._store.get_style(identity_id, style_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return StyleResponse(
        id=style.id,
        identity_id=style.identity_id,
        label=style.label,
        description=style.description,
        default_engine=style.default_engine,
        language=style.language,
        metadata=style.metadata,
    )


@router.get("/{identity_id}/styles/{style_id}/audio")
async def get_style_audio(
    identity_id: str,
    style_id: str,
    vox: VoxID = Depends(get_voxid),
) -> FileResponse:
    try:
        style = vox._store.get_style(identity_id, style_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    audio_path = Path(style.ref_audio_path)
    if not audio_path.exists():
        raise HTTPException(status_code=404, detail="Reference audio file not found")
    return FileResponse(str(audio_path), media_type="audio/wav")
