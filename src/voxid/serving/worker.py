from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import dataclass, field
from enum import Enum

from voxid.adapters import get_adapter
from voxid.adapters.protocol import TTSEngineAdapter

from .config import GenerationRequest, GenerationResult, WorkerConfig

logger = logging.getLogger(__name__)


class WorkerStatus(Enum):
    STARTING = "starting"
    READY = "ready"
    UNHEALTHY = "unhealthy"
    STOPPED = "stopped"


@dataclass
class WorkerHealth:
    """Snapshot of worker health metrics."""

    status: WorkerStatus
    queue_depth: int
    total_processed: int
    total_errors: int
    gpu_memory_bytes: int
    last_health_check: float  # epoch seconds


@dataclass
class _WorkerState:
    """Mutable internal state for a TTSWorker."""

    total_processed: int = 0
    total_errors: int = 0
    status: WorkerStatus = WorkerStatus.STARTING
    last_health_check: float = field(default_factory=time.monotonic)


class TTSWorker:
    """Single-engine GPU worker with async request queue and backpressure.

    Each worker owns one TTS engine adapter pinned to a specific GPU device.
    Requests are queued via ``submit()`` and processed sequentially by the
    internal consumer loop. Backpressure is enforced: ``submit()`` raises
    ``QueueFullError`` when the queue reaches ``max_queue_depth``.
    """

    def __init__(self, config: WorkerConfig) -> None:
        self._config = config
        self._queue: asyncio.Queue[
            tuple[GenerationRequest, asyncio.Future[GenerationResult]]
        ] = asyncio.Queue(maxsize=config.max_queue_depth)
        self._state = _WorkerState()
        self._consumer_task: asyncio.Task[None] | None = None
        self._adapter: TTSEngineAdapter | None = None

    @property
    def config(self) -> WorkerConfig:
        return self._config

    @property
    def engine(self) -> str:
        return self._config.engine

    @property
    def device(self) -> str:
        return self._config.device

    @property
    def queue_depth(self) -> int:
        return self._queue.qsize()

    @property
    def status(self) -> WorkerStatus:
        return self._state.status

    async def start(self) -> None:
        """Initialize the adapter and start the consumer loop."""
        self._set_device_env()
        adapter_cls = get_adapter(self._config.engine)
        self._adapter = adapter_cls()
        self._state.status = WorkerStatus.READY
        self._state.last_health_check = time.monotonic()
        self._consumer_task = asyncio.create_task(self._consume())
        logger.info(
            "worker started: engine=%s device=%s",
            self._config.engine,
            self._config.device,
        )

    async def stop(self) -> None:
        """Cancel the consumer loop and drain remaining requests."""
        self._state.status = WorkerStatus.STOPPED
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            try:
                await self._consumer_task
            except asyncio.CancelledError:
                pass
            self._consumer_task = None

        # Reject anything left in the queue.
        while not self._queue.empty():
            try:
                _req, fut = self._queue.get_nowait()
                if not fut.done():
                    fut.set_exception(WorkerStoppedError(self._config.device))
            except asyncio.QueueEmpty:
                break

        logger.info("worker stopped: device=%s", self._config.device)

    async def submit(self, request: GenerationRequest) -> GenerationResult:
        """Submit a generation request. Raises ``QueueFullError`` on backpressure."""
        if self._state.status != WorkerStatus.READY:
            msg = f"worker on {self._config.device} is {self._state.status.value}"
            raise WorkerUnavailableError(msg)

        loop = asyncio.get_running_loop()
        fut: asyncio.Future[GenerationResult] = loop.create_future()

        try:
            self._queue.put_nowait((request, fut))
        except asyncio.QueueFull:
            raise QueueFullError(self._config.device, self._config.max_queue_depth)

        return await fut

    def health(self) -> WorkerHealth:
        """Return a snapshot of worker health."""
        return WorkerHealth(
            status=self._state.status,
            queue_depth=self._queue.qsize(),
            total_processed=self._state.total_processed,
            total_errors=self._state.total_errors,
            gpu_memory_bytes=self._read_gpu_memory(),
            last_health_check=self._state.last_health_check,
        )

    def check_health(self) -> None:
        """Run a health check and update status."""
        self._state.last_health_check = time.monotonic()
        if self._consumer_task is not None and self._consumer_task.done():
            self._state.status = WorkerStatus.UNHEALTHY
            return
        if self._state.status == WorkerStatus.READY:
            return
        # If stopped or starting, leave as-is.

    # ── internals ──────────────────────────────

    async def _consume(self) -> None:
        """Process requests from the queue sequentially."""
        while True:
            request, fut = await self._queue.get()
            try:
                result = await self._generate(request)
                if not fut.done():
                    fut.set_result(result)
                self._state.total_processed += 1
            except Exception as exc:
                self._state.total_errors += 1
                if not fut.done():
                    fut.set_exception(exc)

    async def _generate(self, request: GenerationRequest) -> GenerationResult:
        """Run adapter.generate() in a thread to avoid blocking the event loop."""
        assert self._adapter is not None  # noqa: S101
        loop = asyncio.get_running_loop()
        waveform, sr = await loop.run_in_executor(
            None,
            self._adapter.generate,
            request.text,
            request.prompt_path,
            request.language,
            request.context_params if request.context_params else None,
        )
        return GenerationResult(
            request_id=request.request_id,
            waveform=waveform,
            sample_rate=sr,
        )

    def _set_device_env(self) -> None:
        """Set CUDA_VISIBLE_DEVICES so the adapter sees only one GPU."""
        if self._config.device.startswith("cuda:"):
            device_index = self._config.device.split(":")[1]
            os.environ["CUDA_VISIBLE_DEVICES"] = device_index

    def _read_gpu_memory(self) -> int:
        """Best-effort GPU memory reading. Returns 0 if torch unavailable."""
        try:
            import torch

            if torch.cuda.is_available():
                idx = 0
                if self._config.device.startswith("cuda:"):
                    idx = int(self._config.device.split(":")[1])
                return int(torch.cuda.memory_allocated(idx))
        except Exception:
            pass
        return 0


# ── Errors ──────────────────────────────────


class QueueFullError(Exception):
    """Raised when a worker's request queue is at capacity."""

    def __init__(self, device: str, max_depth: int) -> None:
        super().__init__(
            f"worker on {device} queue full (max_depth={max_depth})"
        )
        self.device = device
        self.max_depth = max_depth


class WorkerUnavailableError(Exception):
    """Raised when submitting to a non-ready worker."""


class WorkerStoppedError(Exception):
    """Raised for requests still queued when a worker stops."""

    def __init__(self, device: str) -> None:
        super().__init__(f"worker on {device} stopped while request was queued")
        self.device = device
