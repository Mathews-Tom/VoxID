from __future__ import annotations

import asyncio
import threading
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voxid.serving.config import GenerationRequest, WorkerConfig
from voxid.serving.worker import (
    QueueFullError,
    TTSWorker,
    WorkerStatus,
    WorkerStoppedError,
    WorkerUnavailableError,
)


def _make_config(
    engine: str = "stub",
    device: str = "cuda:0",
    max_batch_size: int = 4,
    max_queue_depth: int = 4,
) -> WorkerConfig:
    return WorkerConfig(
        engine=engine,
        device=device,
        max_batch_size=max_batch_size,
        max_queue_depth=max_queue_depth,
    )


def _make_request(request_id: str = "req-1") -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        text="Hello world",
        prompt_path=Path("/tmp/prompt.safetensors"),
        engine="stub",
        language="en",
    )


def _mock_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.engine_name = "stub"
    adapter.generate.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    return adapter


def _blocking_adapter() -> tuple[MagicMock, threading.Event]:
    """Return an adapter whose generate() blocks on a threading.Event."""
    adapter = MagicMock()
    adapter.engine_name = "stub"
    block = threading.Event()

    def _generate(*_a: object, **_kw: object) -> tuple[np.ndarray, int]:  # type: ignore[type-arg]
        block.wait()
        return (np.zeros(100, dtype=np.float32), 16000)

    adapter.generate.side_effect = _generate
    return adapter, block


class TestWorkerConfig:
    def test_defaults(self) -> None:
        cfg = WorkerConfig(engine="stub", device="cuda:0")
        assert cfg.max_batch_size == 4
        assert cfg.max_queue_depth == 16

    def test_frozen(self) -> None:
        cfg = _make_config()
        with pytest.raises(AttributeError):
            cfg.engine = "other"  # type: ignore[misc]


class TestTTSWorkerLifecycle:
    @pytest.mark.asyncio
    async def test_start_sets_ready(self) -> None:
        worker = TTSWorker(_make_config())
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _mock_adapter()
            await worker.start()

        assert worker.status == WorkerStatus.READY
        await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_sets_stopped(self) -> None:
        worker = TTSWorker(_make_config())
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _mock_adapter()
            await worker.start()
            await worker.stop()

        assert worker.status == WorkerStatus.STOPPED

    @pytest.mark.asyncio
    async def test_submit_before_start_raises(self) -> None:
        worker = TTSWorker(_make_config())
        with pytest.raises(WorkerUnavailableError):
            await worker.submit(_make_request())


class TestTTSWorkerGeneration:
    @pytest.mark.asyncio
    async def test_submit_returns_result(self) -> None:
        worker = TTSWorker(_make_config())
        adapter = _mock_adapter()
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: adapter
            await worker.start()

        result = await worker.submit(_make_request("gen-1"))
        assert result.request_id == "gen-1"
        assert result.sample_rate == 16000
        assert result.waveform.shape == (16000,)
        await worker.stop()

    @pytest.mark.asyncio
    async def test_health_reports_queue_depth(self) -> None:
        worker = TTSWorker(_make_config())
        adapter = _mock_adapter()
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: adapter
            await worker.start()

        health = worker.health()
        assert health.status == WorkerStatus.READY
        assert health.queue_depth == 0
        await worker.stop()


class TestTTSWorkerBackpressure:
    @pytest.mark.asyncio
    async def test_queue_full_raises(self) -> None:
        cfg = _make_config(max_queue_depth=1)
        worker = TTSWorker(cfg)

        adapter, block = _blocking_adapter()
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: adapter
            await worker.start()

        # First request goes to consumer, blocks in generate (thread).
        _task1 = asyncio.create_task(worker.submit(_make_request("r1")))
        await asyncio.sleep(0.1)

        # Second request sits in queue (depth=1).
        _task2 = asyncio.create_task(worker.submit(_make_request("r2")))
        await asyncio.sleep(0.05)

        # Third request should overflow.
        with pytest.raises(QueueFullError):
            await worker.submit(_make_request("r3"))

        block.set()
        await worker.stop()

    @pytest.mark.asyncio
    async def test_stop_rejects_queued_requests(self) -> None:
        cfg = _make_config(max_queue_depth=4)
        worker = TTSWorker(cfg)

        adapter, block = _blocking_adapter()
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: adapter
            await worker.start()

        _task1 = asyncio.create_task(worker.submit(_make_request("r1")))
        await asyncio.sleep(0.1)
        task2 = asyncio.create_task(worker.submit(_make_request("r2")))
        await asyncio.sleep(0.05)

        block.set()
        await worker.stop()

        with pytest.raises(
            (WorkerStoppedError, asyncio.CancelledError)
        ):
            await task2
