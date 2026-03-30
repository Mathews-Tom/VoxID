from __future__ import annotations

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
from voxid.serving.dispatcher import GPUDispatcher, NoWorkerAvailableError
from voxid.serving.worker import WorkerStatus


def _stub_adapter() -> MagicMock:
    adapter = MagicMock()
    adapter.engine_name = "stub"
    adapter.generate.return_value = (np.zeros(16000, dtype=np.float32), 16000)
    return adapter


def _make_config(
    strategy: DispatchStrategy = DispatchStrategy.ROUND_ROBIN,
    workers: list[WorkerConfig] | None = None,
) -> ServingConfig:
    if workers is None:
        workers = [
            WorkerConfig(engine="stub", device="cuda:0", max_queue_depth=8),
            WorkerConfig(engine="stub", device="cuda:1", max_queue_depth=8),
        ]
    return ServingConfig(
        workers=workers,
        dispatch_strategy=strategy,
        health_check_interval_s=9999,  # disable periodic checks in tests
    )


def _make_request(engine: str = "stub", request_id: str = "req-1") -> GenerationRequest:
    return GenerationRequest(
        request_id=request_id,
        text="Test text",
        prompt_path=Path("/tmp/prompt.safetensors"),
        engine=engine,
        language="en",
    )


class TestDispatcherLifecycle:
    @pytest.mark.asyncio
    async def test_start_stop(self) -> None:
        cfg = _make_config()
        dispatcher = GPUDispatcher(cfg)
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        assert all(w.status == WorkerStatus.READY for w in dispatcher.workers)
        await dispatcher.stop()
        assert all(w.status == WorkerStatus.STOPPED for w in dispatcher.workers)

    @pytest.mark.asyncio
    async def test_engines(self) -> None:
        cfg = _make_config(workers=[
            WorkerConfig(engine="stub", device="cuda:0"),
            WorkerConfig(engine="fish-speech", device="cuda:1"),
        ])
        dispatcher = GPUDispatcher(cfg)
        assert dispatcher.engines() == {"stub", "fish-speech"}


class TestDispatcherRoundRobin:
    @pytest.mark.asyncio
    async def test_round_robin_distributes(self) -> None:
        cfg = _make_config(strategy=DispatchStrategy.ROUND_ROBIN)
        dispatcher = GPUDispatcher(cfg)
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        r1 = await dispatcher.dispatch(_make_request(request_id="a"))
        r2 = await dispatcher.dispatch(_make_request(request_id="b"))

        assert r1.request_id == "a"
        assert r2.request_id == "b"

        # Both workers should have processed one request each.
        processed = [w.health().total_processed for w in dispatcher.workers]
        assert sum(processed) == 2
        await dispatcher.stop()


class TestDispatcherLeastLoaded:
    @pytest.mark.asyncio
    async def test_least_loaded_prefers_empty_queue(self) -> None:
        cfg = _make_config(strategy=DispatchStrategy.LEAST_LOADED)
        dispatcher = GPUDispatcher(cfg)
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        result = await dispatcher.dispatch(_make_request())
        assert result.request_id == "req-1"
        await dispatcher.stop()


class TestDispatcherEngineAffinity:
    @pytest.mark.asyncio
    async def test_no_worker_for_engine_raises(self) -> None:
        cfg = _make_config()
        dispatcher = GPUDispatcher(cfg)
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        with pytest.raises(NoWorkerAvailableError):
            await dispatcher.dispatch(_make_request(engine="nonexistent"))
        await dispatcher.stop()

    @pytest.mark.asyncio
    async def test_affinity_routes_to_correct_engine(self) -> None:
        fish_adapter = MagicMock()
        fish_adapter.engine_name = "fish-speech"
        fish_adapter.generate.return_value = (np.zeros(22050, dtype=np.float32), 22050)

        cfg = _make_config(workers=[
            WorkerConfig(engine="stub", device="cuda:0"),
            WorkerConfig(engine="fish-speech", device="cuda:1"),
        ])
        dispatcher = GPUDispatcher(cfg)

        def adapter_factory(engine_name: str):  # noqa: ANN202
            def factory() -> MagicMock:
                if engine_name == "fish-speech":
                    return fish_adapter
                return _stub_adapter()
            return factory

        with patch("voxid.serving.worker.get_adapter") as mock_get:
            # get_adapter is called once per worker at start time.
            mock_get.side_effect = [
                lambda: _stub_adapter(),
                lambda: fish_adapter,
            ]
            await dispatcher.start()

        result = await dispatcher.dispatch(_make_request(engine="fish-speech"))
        assert result.sample_rate == 22050
        await dispatcher.stop()


class TestDispatcherHealth:
    @pytest.mark.asyncio
    async def test_health_report(self) -> None:
        cfg = _make_config()
        dispatcher = GPUDispatcher(cfg)
        with patch("voxid.serving.worker.get_adapter") as mock_get:
            mock_get.return_value = lambda: _stub_adapter()
            await dispatcher.start()

        health = dispatcher.health()
        assert health.total_workers == 2
        assert health.healthy_workers == 2
        assert len(health.workers) == 2
        await dispatcher.stop()


class TestServingConfigParsing:
    def test_load_serving_config_from_toml(self, tmp_path: Path) -> None:
        from voxid.serving.config import load_serving_config

        config_file = tmp_path / "serving.toml"
        config_file.write_text(
            'dispatch_strategy = "least_loaded"\n'
            "health_check_interval_s = 15.0\n"
            "\n"
            "[[workers]]\n"
            'engine = "qwen3-tts"\n'
            'device = "cuda:0"\n'
            "max_batch_size = 8\n"
            "max_queue_depth = 32\n"
            "\n"
            "[[workers]]\n"
            'engine = "fish-speech"\n'
            'device = "cuda:1"\n'
        )

        cfg = load_serving_config(config_file)
        assert cfg.dispatch_strategy == DispatchStrategy.LEAST_LOADED
        assert cfg.health_check_interval_s == 15.0
        assert len(cfg.workers) == 2
        assert cfg.workers[0].engine == "qwen3-tts"
        assert cfg.workers[0].device == "cuda:0"
        assert cfg.workers[0].max_batch_size == 8
        assert cfg.workers[0].max_queue_depth == 32
        assert cfg.workers[1].engine == "fish-speech"
        assert cfg.workers[1].max_batch_size == 4  # default

    def test_load_empty_workers_raises(self, tmp_path: Path) -> None:
        from voxid.serving.config import load_serving_config

        config_file = tmp_path / "empty.toml"
        config_file.write_text('dispatch_strategy = "round_robin"\n')

        with pytest.raises(ValueError, match="no workers"):
            load_serving_config(config_file)
