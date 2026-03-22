from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import voxid.adapters.chatterbox  # noqa: F401
import voxid.adapters.cosyvoice2  # noqa: F401
import voxid.adapters.fish_speech  # noqa: F401
import voxid.adapters.indextts2  # noqa: F401
import voxid.adapters.stub  # noqa: F401
from voxid.config import VoxIDConfig
from voxid.core import VoxID


@pytest.fixture
def vox(tmp_path: Path) -> VoxID:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    return VoxID(config=config)


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


def test_select_engine_returns_default(vox: VoxID) -> None:
    # "en" is supported by the stub adapter (the default)
    engine = vox._select_engine(language="en")

    assert engine == "stub"


def test_select_engine_fallback_for_unsupported_language(vox: VoxID) -> None:
    # Arabic is not in stub's supported_languages — should fall back to
    # an engine that supports "ar" (fish-speech has it)
    engine = vox._select_engine(language="ar")

    assert engine != "stub"
    # fish-speech is the only registered engine with Arabic
    assert engine == "fish-speech"


def test_select_engine_streaming_preference(vox: VoxID) -> None:
    # stub supports streaming, so it should be returned for a supported language
    engine = vox._select_engine(language="en", need_streaming=True)

    from voxid.adapters import _registry

    cls = _registry[engine]
    adapter = cls()
    assert adapter.capabilities.supports_streaming is True


def test_select_engine_emotion_preference(vox: VoxID) -> None:
    # Only indextts2 has supports_emotion_control=True
    # indextts2 supports "en", so it must be selected
    engine = vox._select_engine(language="en", need_emotion=True)

    assert engine == "indextts2"


def test_cross_engine_prompt_cache(vox: VoxID, ref_audio: Path) -> None:
    # Create identity and style with stub engine
    vox.create_identity(id="alice", name="Alice")
    vox.add_style(
        identity_id="alice",
        id="conversational",
        label="Conversational",
        description="Warm tone",
        ref_audio=ref_audio,
        ref_text="Hello from Alice",
        engine="stub",
    )

    # Verify stub prompt cache was created
    stub_prompt = vox._store.get_prompt_path("alice", "conversational", "stub")
    assert stub_prompt is not None
    assert stub_prompt.exists()

    # The stub prompt file is engine-namespaced
    assert stub_prompt.name == "stub.safetensors"

    # Verify no cross-contamination: indextts2 cache must not exist yet
    indextts2_prompt = vox._store.get_prompt_path(
        "alice", "conversational", "indextts2"
    )
    assert indextts2_prompt is None
