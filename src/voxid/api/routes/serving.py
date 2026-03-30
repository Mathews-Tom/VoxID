from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from voxid.api.deps import get_dispatcher
from voxid.serving.dispatcher import GPUDispatcher

router = APIRouter(tags=["serving"])


@router.get("/serving/health")
async def serving_health(
    dispatcher: GPUDispatcher | None = Depends(get_dispatcher),
) -> dict[str, Any]:
    """Return per-worker health status for multi-GPU serving.

    Returns 200 with ``{"enabled": false}`` when no serving config is active.
    """
    if dispatcher is None:
        return {"enabled": False}

    health = dispatcher.health()
    workers: list[dict[str, Any]] = []
    for key, wh in health.workers.items():
        workers.append({
            "id": key,
            "status": wh.status.value,
            "queue_depth": wh.queue_depth,
            "total_processed": wh.total_processed,
            "total_errors": wh.total_errors,
            "gpu_memory_bytes": wh.gpu_memory_bytes,
        })

    return {
        "enabled": True,
        "total_workers": health.total_workers,
        "healthy_workers": health.healthy_workers,
        "workers": workers,
    }
