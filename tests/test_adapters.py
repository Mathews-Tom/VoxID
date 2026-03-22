from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest

from voxid.adapters import (
    EngineCapabilities,
    _registry,  # type: ignore[attr-defined]
    get_adapter,
    list_adapters,
    register_adapter,
)


@pytest.fixture(autouse=True)
def _clean_registry() -> Iterator[None]:
    """Restore the adapter registry to its original state after each test."""
    original = dict(_registry)
    yield
    _registry.clear()
    _registry.update(original)


class _DummyAdapter:
    engine_name = "test-engine"  # plain string, not property
    capabilities = EngineCapabilities()

    def build_prompt(
        self, ref_audio: Path, ref_text: str, output_path: Path
    ) -> Path:
        return output_path

    def generate(
        self, text: str, prompt_path: Path, language: str = "en"
    ) -> tuple[np.ndarray, int]:
        return np.zeros(100, dtype=np.float32), 24000

    def generate_streaming(
        self, text: str, prompt_path: Path, language: str = "en"
    ) -> Iterator[np.ndarray]:
        yield np.zeros(100, dtype=np.float32)


def test_register_adapter_adds_to_registry() -> None:
    # Act
    register_adapter(_DummyAdapter)

    # Assert
    assert "test-engine" in list_adapters()


def test_get_adapter_returns_registered_class() -> None:
    # Arrange
    register_adapter(_DummyAdapter)

    # Act
    cls = get_adapter("test-engine")

    # Assert
    assert cls is _DummyAdapter


def test_get_adapter_unknown_raises() -> None:
    # Act / Assert
    with pytest.raises(KeyError):
        get_adapter("nonexistent-engine")


def test_register_adapter_missing_method_raises() -> None:
    # Arrange
    class _IncompleteAdapter:
        engine_name = "incomplete"
        capabilities = EngineCapabilities()
        # build_prompt, generate, generate_streaming all missing

    # Act / Assert
    with pytest.raises(TypeError):
        register_adapter(_IncompleteAdapter)


def test_engine_capabilities_defaults() -> None:
    # Act
    caps = EngineCapabilities()

    # Assert
    assert caps.supports_streaming is False
    assert caps.supports_emotion_control is False
    assert caps.supports_paralinguistic_tags is False
    assert caps.max_ref_audio_seconds == 30.0
    assert caps.supported_languages == ("en",)
    assert caps.streaming_latency_ms == 0
    assert caps.supports_word_timing is False
