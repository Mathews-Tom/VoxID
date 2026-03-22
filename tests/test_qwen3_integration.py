"""Integration tests for Qwen3-TTS adapter with real model.

Requires mlx-audio or qwen-tts installed. Skipped if neither is available.
Model is loaded once per session via module-scoped fixture.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from voxid.adapters.qwen3_tts import Qwen3TTSAdapter, _detect_backend
from voxid.config import VoxIDConfig
from voxid.core import VoxID

# Skip entire module if no backend is available
try:
    _backend = _detect_backend()
except ImportError:
    pytest.skip(
        "No Qwen3-TTS backend (qwen-tts or mlx-audio)",
        allow_module_level=True,
    )

# Use 0.6B for faster CI; override with env var if needed
_MODEL = (
    "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
    if _backend == "mlx-audio"
    else "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
)


@pytest.fixture(scope="module")
def adapter() -> Qwen3TTSAdapter:
    return Qwen3TTSAdapter(model_name=_MODEL, backend=_backend)


@pytest.fixture(scope="module")
def ref_audio(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """3-second sine wave reference audio."""
    sr = 24000
    t = np.linspace(0, 3.0, sr * 3, dtype=np.float32)
    audio = (0.3 * np.sin(2 * np.pi * 220 * t)).astype(
        np.float32,
    )
    path = tmp_path_factory.mktemp("audio") / "ref.wav"
    sf.write(str(path), audio, sr)
    return path


@pytest.fixture(scope="module")
def prompt_path(
    adapter: Qwen3TTSAdapter,
    ref_audio: Path,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    out = tmp_path_factory.mktemp("prompts") / "prompt.safetensors"
    adapter.build_prompt(
        ref_audio,
        "This is a test reference audio clip.",
        out,
    )
    return out


# ── Adapter-level tests ───────────────────────────


class TestQwen3TTSAdapter:
    def test_build_prompt_creates_file(
        self, prompt_path: Path,
    ) -> None:
        assert prompt_path.exists()
        assert prompt_path.stat().st_size > 0

    def test_generate_returns_waveform_and_sr(
        self,
        adapter: Qwen3TTSAdapter,
        prompt_path: Path,
    ) -> None:
        wav, sr = adapter.generate(
            "Hello world.", prompt_path, language="en",
        )
        assert isinstance(wav, np.ndarray)
        assert wav.dtype == np.float32
        assert sr > 0
        assert len(wav) > 0

    def test_generate_output_is_valid_wav(
        self,
        adapter: Qwen3TTSAdapter,
        prompt_path: Path,
        tmp_path_factory: pytest.TempPathFactory,
    ) -> None:
        wav, sr = adapter.generate(
            "Testing audio output.", prompt_path,
        )
        out = tmp_path_factory.mktemp("out") / "test.wav"
        sf.write(str(out), wav, sr)
        data, read_sr = sf.read(str(out))
        assert read_sr == sr
        assert len(data) > 0

    def test_generate_different_texts_different_lengths(
        self,
        adapter: Qwen3TTSAdapter,
        prompt_path: Path,
    ) -> None:
        short_wav, _ = adapter.generate(
            "Hi.", prompt_path,
        )
        long_wav, _ = adapter.generate(
            "The retrieval augmented generation pipeline "
            "processes queries through embedding lookup, "
            "context reranking, and language model completion.",
            prompt_path,
        )
        # Longer text should generally produce longer audio
        assert len(long_wav) > len(short_wav)

    def test_generate_streaming_raises(
        self,
        adapter: Qwen3TTSAdapter,
        prompt_path: Path,
    ) -> None:
        with pytest.raises(NotImplementedError):
            list(adapter.generate_streaming(
                "Test.", prompt_path,
            ))

    def test_backend_detected_correctly(self) -> None:
        assert _backend in ("mlx-audio", "qwen-tts")


# ── VoxID end-to-end tests ────────────────────────


class TestVoxIDEndToEnd:
    @pytest.fixture()
    def vox(
        self,
        tmp_path: Path,
        ref_audio: Path,
    ) -> VoxID:
        config = VoxIDConfig(
            store_path=tmp_path / "voxid",
            default_engine="qwen3-tts",
        )
        v = VoxID(config=config)

        # Patch adapter to use test model
        from voxid.adapters.qwen3_tts import Qwen3TTSAdapter
        patched = Qwen3TTSAdapter(
            model_name=_MODEL, backend=_backend,
        )
        # Replace the adapter class in registry so core uses it
        from voxid.adapters import _registry
        _registry["qwen3-tts"] = type(patched)
        # Store instance for reuse
        self._adapter_instance = patched

        v.create_identity(
            id="test", name="Test",
            default_style="conversational",
        )
        v.add_style(
            identity_id="test",
            id="conversational",
            label="Conversational",
            description="Casual, warm",
            ref_audio=ref_audio,
            ref_text="Test reference audio clip.",
            engine="qwen3-tts",
        )
        return v

    def test_generate_produces_valid_wav(
        self, vox: VoxID, tmp_path: Path,
    ) -> None:
        out_path, sr = vox.generate(
            text="Hello from VoxID.",
            identity_id="test",
            style="conversational",
        )
        assert out_path.exists()
        data, read_sr = sf.read(str(out_path))
        assert read_sr == sr
        assert len(data) > 0

    def test_route_then_generate(
        self, vox: VoxID,
    ) -> None:
        decision = vox.route(
            "Testing the route then generate flow.",
            identity_id="test",
        )
        assert "style" in decision
        assert "confidence" in decision

        out_path, sr = vox.generate(
            text="Testing the route then generate flow.",
            identity_id="test",
        )
        assert out_path.exists()
        assert sr > 0
