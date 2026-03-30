from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxid.tokenizer.config import AcousticConfig


class TestAcousticTokenizerLoadAudio:
    """Test audio loading and resampling without requiring WavTokenizer."""

    def test_load_mono(self, tmp_path: Path) -> None:
        from voxid.tokenizer.acoustic import AcousticTokenizer

        config = AcousticConfig(device="cpu", sample_rate=24000)
        tokenizer = AcousticTokenizer(config)

        audio = np.random.default_rng(42).standard_normal(48000).astype(np.float32)
        path = tmp_path / "mono.wav"
        sf.write(str(path), audio, 24000)

        loaded = tokenizer._load_audio(path)
        assert loaded.dtype == np.float32
        assert len(loaded) == 48000

    def test_load_stereo_to_mono(self, tmp_path: Path) -> None:
        from voxid.tokenizer.acoustic import AcousticTokenizer

        config = AcousticConfig(device="cpu", sample_rate=24000)
        tokenizer = AcousticTokenizer(config)

        rng = np.random.default_rng(42)
        stereo = rng.standard_normal((48000, 2)).astype(np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, 24000)

        loaded = tokenizer._load_audio(path)
        assert loaded.ndim == 1
        assert loaded.dtype == np.float32

    def test_load_resample(self, tmp_path: Path) -> None:
        from voxid.tokenizer.acoustic import AcousticTokenizer

        config = AcousticConfig(device="cpu", sample_rate=24000)
        tokenizer = AcousticTokenizer(config)

        # 16kHz audio → should be resampled to 24kHz
        audio = np.random.default_rng(42).standard_normal(16000).astype(np.float32)
        path = tmp_path / "16k.wav"
        sf.write(str(path), audio, 16000)

        loaded = tokenizer._load_audio(path)
        expected_len = int(1.0 * 24000)  # 1 second at 24kHz
        assert len(loaded) == expected_len

    def test_ensure_loaded_missing_wavtokenizer(self) -> None:
        from voxid.tokenizer.acoustic import AcousticTokenizer

        config = AcousticConfig(device="cpu")
        tokenizer = AcousticTokenizer(config)

        # WavTokenizer is not installed in test env
        with pytest.raises(ImportError, match="wavtokenizer"):
            tokenizer._ensure_loaded()


class TestAcousticConfig:
    def test_defaults(self) -> None:
        config = AcousticConfig()
        assert config.device == "cpu"
        assert config.frame_rate == 40.0
        assert config.n_codebooks == 1
        assert config.sample_rate == 24000

    def test_frozen(self) -> None:
        config = AcousticConfig()
        with pytest.raises(AttributeError):
            config.device = "cuda"  # type: ignore[misc]
