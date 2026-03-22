from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from sse_starlette.sse import EventSourceResponse

from voxid.api.deps import get_voxid
from voxid.api.models import (
    GenerateRequest,
    GenerateResponse,
    GenerateSegmentsRequest,
    GenerateSegmentsResponse,
    SegmentItemResponse,
)
from voxid.core import VoxID
from voxid.schemas import SceneManifest

router = APIRouter(prefix="/generate", tags=["generate"])


@router.post("", response_model=GenerateResponse)
async def generate(
    req: GenerateRequest,
    vox: VoxID = Depends(get_voxid),
) -> GenerateResponse:
    try:
        audio_path, sr = vox.generate(
            text=req.text,
            identity_id=req.identity_id,
            style=req.style,
            engine=req.engine,
        )
        decision = vox.route(req.text, req.identity_id)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return GenerateResponse(
        audio_path=str(audio_path),
        sample_rate=sr,
        style_used=req.style or decision["style"],
        identity_id=req.identity_id,
    )


@router.post("/segments", response_model=GenerateSegmentsResponse)
async def generate_segments(
    req: GenerateSegmentsRequest,
    vox: VoxID = Depends(get_voxid),
) -> GenerateSegmentsResponse:
    try:
        result = vox.generate_segments(
            text=req.text,
            identity_id=req.identity_id,
            engine=req.engine,
            stitch=req.stitch,
        )
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    segments = [
        SegmentItemResponse(
            index=seg.index,
            text=seg.text,
            style=seg.style,
            audio_path=str(seg.audio_path),
            duration_ms=seg.duration_ms,
            boundary_type=seg.boundary_type,
        )
        for seg in result.segments
    ]
    return GenerateSegmentsResponse(
        segments=segments,
        stitched_path=str(result.stitched_path) if result.stitched_path else None,
        total_duration_ms=result.total_duration_ms,
    )


@router.post("/manifest")
async def generate_manifest(
    manifest: SceneManifest,
    vox: VoxID = Depends(get_voxid),
) -> dict[str, Any]:
    try:
        result = vox.generate_from_manifest(manifest, stitch=True)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return result.model_dump()


@router.post("/stream")
async def generate_stream(
    req: GenerateSegmentsRequest,
    vox: VoxID = Depends(get_voxid),
) -> EventSourceResponse:
    """SSE endpoint for segment-level generation progress."""

    async def event_generator() -> AsyncGenerator[dict[str, str], None]:
        try:
            result = vox.generate_segments(
                text=req.text,
                identity_id=req.identity_id,
                engine=req.engine,
                stitch=req.stitch,
            )
        except (FileNotFoundError, ValueError) as exc:
            yield {
                "event": "error",
                "data": json.dumps({"detail": str(exc)}),
            }
            return

        for seg in result.segments:
            yield {
                "event": "segment",
                "data": json.dumps(
                    {
                        "index": seg.index,
                        "style": seg.style,
                        "audio_path": str(seg.audio_path),
                        "duration_ms": seg.duration_ms,
                    }
                ),
            }

        yield {
            "event": "complete",
            "data": json.dumps(
                {
                    "total_duration_ms": result.total_duration_ms,
                    "stitched_path": (
                        str(result.stitched_path)
                        if result.stitched_path
                        else None
                    ),
                    "segment_count": len(result.segments),
                }
            ),
        }

    return EventSourceResponse(event_generator())
