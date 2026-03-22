from __future__ import annotations

import voxid.adapters.stub  # noqa: F401 — registers StubAdapter
from voxid.adapters.protocol import TTSEngineAdapter
from voxid.adapters.qwen3_tts import Qwen3TTSAdapter
from voxid.adapters.stub import StubAdapter

# ── Qwen3TTSAdapter tests ─────────────────────────────────────────────────────


def test_qwen3_tts_adapter_engine_name() -> None:
    assert Qwen3TTSAdapter.engine_name == "qwen3-tts"


def test_qwen3_tts_adapter_capabilities_languages() -> None:
    adapter = Qwen3TTSAdapter()
    langs = adapter.capabilities.supported_languages

    assert "en" in langs
    assert "zh" in langs
    assert "ja" in langs


def test_qwen3_tts_adapter_capabilities_streaming() -> None:
    adapter = Qwen3TTSAdapter()

    assert adapter.capabilities.supports_streaming is False


# ── StubAdapter tests ─────────────────────────────────────────────────────────


def test_stub_adapter_engine_name() -> None:
    assert StubAdapter.engine_name == "stub"


def test_stub_adapter_capabilities() -> None:
    adapter = StubAdapter()
    caps = adapter.capabilities

    assert caps.supports_streaming is True
    assert caps.supports_emotion_control is False
    assert caps.supports_paralinguistic_tags is False
    assert caps.max_ref_audio_seconds == 60.0
    assert "en" in caps.supported_languages
    assert "zh" in caps.supported_languages
    assert "ja" in caps.supported_languages
    assert caps.streaming_latency_ms == 0
    assert caps.supports_word_timing is False


def test_stub_adapter_satisfies_protocol() -> None:
    adapter = StubAdapter()

    assert isinstance(adapter, TTSEngineAdapter)
    assert hasattr(adapter, "engine_name")
    assert hasattr(adapter, "capabilities")
    assert hasattr(adapter, "build_prompt")
    assert hasattr(adapter, "generate")
    assert hasattr(adapter, "generate_streaming")
