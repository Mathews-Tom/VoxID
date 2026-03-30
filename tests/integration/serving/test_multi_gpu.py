"""Integration tests for multi-GPU serving pipeline.

These tests exercise the full dispatcher → worker → adapter chain using the
stub adapter (no real GPU required). They verify end-to-end request flow,
concurrent dispatch, and error isolation.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from voxid.serving.config import (
    DispatchStrategy,
    GenerationRequest,
    ServingConfig,
    WorkerConfig,
)
from voxid.serving.dispatcher import GPUDispatcher


def _stub_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.engine_name = "stub"
    adapter.generate.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    return adapter


def _make_request(engine: str = "stub", request_id: str = "req-1") -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        text="Integration test text",
        prompt_path=Path("/tmp/prompt.safetensors"),
        engine=engine,
        language="en",
    )


@pytest.mark.asyncio
class TestMultiGPUEndToEnd:
    async def test_concurrent_dispatch_to_multiple_workers(self) -> None:
        """Dispatch 10 requests concurrently across 2 workers."""
        cfg = ServingConfig(
            workers=[
                WorkerConfig(engine="stub", device="cuda:0", max_queue_depth=16),
                WorkerConfig(engine="stub", device="cuda:1", max_queue_depth=16),
            ],
            dispatch_strategy=DispatchStrategy.ROUND_ROBIN,
            health_check_interval_s=9999,
        )
        dispatcher = GPUDispatcher(cfg)

        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        requests = [_make_request(request_id=f"r-{i}") for i in range(10)]
        results = await asyncio.gather(
            *(dispatcher.dispatch(r) for r in requests)
        )

        assert len(results) == 10
        assert all(r.sample_rate == 16000 for r in results)

        total_processed = sum(
            w.health().total_processed for w in dispatcher.workers
        )
        assert total_processed == 10
        await dispatcher.stop()

    async def test_worker_failure_isolates_errors(self) -> None:
        """A failing worker does not crash the dispatcher or other workers."""
        good_adapter = _stub_adapter()
        bad_adapter = MagicMock()
        bad_adapter.engine_name = "bad-engine"
        bad_adapter.generate.side_effect = RuntimeError("GPU OOM")

        cfg = ServingConfig(
            workers=[
                WorkerConfig(engine="stub", device="cuda:0", max_queue_depth=8),
                WorkerConfig(engine="bad-engine", device="cuda:1", max_queue_depth=8),
            ],
            dispatch_strategy=DispatchStrategy.ROUND_ROBIN,
            health_check_interval_s=9999,
        )
        dispatcher = GPUDispatcher(cfg)

        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.side_effect = [
                lambda: good_adapter,
                lambda: bad_adapter,
            ]
            await dispatcher.start()

        # Good engine works.
        result = await dispatcher.dispatch(_make_request(engine="stub"))
        assert result.sample_rate == 16000

        # Bad engine raises but doesn't crash.
        with pytest.raises(RuntimeError, match="GPU OOM"):
            await dispatcher.dispatch(_make_request(engine="bad-engine"))

        # Good engine still works after the failure.
        result2 = await dispatcher.dispatch(
            _make_request(engine="stub", request_id="r2")
        )
        assert result2.request_id == "r2"

        health = dispatcher.health()
        assert health.healthy_workers >= 1
        await dispatcher.stop()

    async def test_health_endpoint_reflects_state(self) -> None:
        """Health snapshot updates after processing requests."""
        cfg = ServingConfig(
            workers=[
                WorkerConfig(engine="stub", device="cuda:0", max_queue_depth=8),
            ],
            dispatch_strategy=DispatchStrategy.ROUND_ROBIN,
            health_check_interval_s=9999,
        )
        dispatcher = GPUDispatcher(cfg)

        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        h1 = dispatcher.health()
        assert h1.healthy_workers == 1
        assert h1.workers["stub@cuda:0"].total_processed == 0

        await dispatcher.dispatch(_make_request())

        h2 = dispatcher.health()
        assert h2.workers["stub@cuda:0"].total_processed == 1

        await dispatcher.stop()

    async def test_full_toml_config_roundtrip(self, tmp_path: Path) -> None:
        """Load config from TOML, build dispatcher, dispatch a request."""
        from voxid.serving.config import load_serving_config

        config_file = tmp_path / "serving.toml"
        config_file.write_text(
            'dispatch_strategy = "least_loaded"\n'
            "health_check_interval_s = 9999\n"
            "\n"
            "[[workers]]\n"
            'engine = "stub"\n'
            'device = "cuda:0"\n'
            "max_queue_depth = 8\n"
        )

        cfg = load_serving_config(config_file)
        dispatcher = GPUDispatcher(cfg)

        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        result = await dispatcher.dispatch(_make_request())
        assert result.sample_rate == 16000
        await dispatcher.stop()
