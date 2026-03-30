from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass

from .config import DispatchStrategy, GenerationRequest, GenerationResult, ServingConfig
from .worker import (
    TTSWorker,
    WorkerHealth,
    WorkerStatus,
)

logger = logging.getLogger(__name__)


@dataclass
class ServingHealth:
    """Aggregate health snapshot for all workers."""

    workers: dict[str, WorkerHealth]
    total_workers: int
    healthy_workers: int


class GPUDispatcher:
    """Routes generation requests to a pool of TTSWorkers.

    Supports two strategies:
    - **round_robin**: cycle through healthy workers for the target engine.
    - **least_loaded**: pick the healthy worker with the smallest queue depth.

    Engine affinity is enforced: a request for engine ``X`` only routes to
    workers running engine ``X``.
    """

    def __init__(self, config: ServingConfig) -> None:
        self._config = config
        self._workers: list[TTSWorker] = [
            TTSWorker(wc) for wc in config.workers
        ]
        # Round-robin cursor per engine.
        self._rr_index: dict[str, int] = {}
        self._health_task: asyncio.Task[None] | None = None

    @property
    def workers(self) -> list[TTSWorker]:
        return list(self._workers)

    async def start(self) -> None:
        """Start all workers and the periodic health-check loop."""
        await asyncio.gather(*(w.start() for w in self._workers))
        self._health_task = asyncio.create_task(self._health_loop())
        logger.info("dispatcher started: %d workers", len(self._workers))

    async def stop(self) -> None:
        """Stop all workers and the health-check loop."""
        if self._health_task is not None:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass
            self._health_task = None
        await asyncio.gather(*(w.stop() for w in self._workers))
        logger.info("dispatcher stopped")

    async def dispatch(self, request: GenerationRequest) -> GenerationResult:
        """Route a request to a worker and return the result.

        Raises ``NoWorkerAvailableError`` if no healthy worker serves the
        requested engine, or ``QueueFullError`` if all matching workers
        have full queues.
        """
        worker = self._select_worker(request.engine)
        return await worker.submit(request)

    def health(self) -> ServingHealth:
        """Return aggregate health for all workers."""
        worker_health: dict[str, WorkerHealth] = {}
        healthy = 0
        for w in self._workers:
            h = w.health()
            key = f"{w.engine}@{w.device}"
            worker_health[key] = h
            if h.status == WorkerStatus.READY:
                healthy += 1
        return ServingHealth(
            workers=worker_health,
            total_workers=len(self._workers),
            healthy_workers=healthy,
        )

    def engines(self) -> set[str]:
        """Return the set of engine names served by this dispatcher."""
        return {w.engine for w in self._workers}

    # ── internals ──────────────────────────────

    def _select_worker(self, engine: str) -> TTSWorker:
        """Pick a worker for the given engine using the configured strategy."""
        candidates = [
            w
            for w in self._workers
            if w.engine == engine and w.status == WorkerStatus.READY
        ]
        if not candidates:
            raise NoWorkerAvailableError(engine)

        if self._config.dispatch_strategy == DispatchStrategy.LEAST_LOADED:
            return self._least_loaded(candidates)
        return self._round_robin(candidates, engine)

    def _round_robin(self, candidates: list[TTSWorker], engine: str) -> TTSWorker:
        idx = self._rr_index.get(engine, 0) % len(candidates)
        self._rr_index[engine] = idx + 1
        return candidates[idx]

    def _least_loaded(self, candidates: list[TTSWorker]) -> TTSWorker:
        return min(candidates, key=lambda w: w.queue_depth)

    async def _health_loop(self) -> None:
        """Periodically check worker health."""
        interval = self._config.health_check_interval_s
        while True:
            await asyncio.sleep(interval)
            for w in self._workers:
                w.check_health()


# ── Errors ──────────────────────────────────


class NoWorkerAvailableError(Exception):
    """No healthy worker serves the requested engine."""

    def __init__(self, engine: str) -> None:
        super().__init__(f"no healthy worker available for engine '{engine}'")
        self.engine = engine
