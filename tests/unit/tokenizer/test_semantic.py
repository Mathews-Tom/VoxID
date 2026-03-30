from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxid.tokenizer.config import SemanticConfig


class TestSemanticTokenizerLoadAudio:
    """Test audio loading without requiring HuBERT."""

    def test_load_mono_16k(self, tmp_path: Path) -> None:
        from voxid.tokenizer.semantic import SemanticTokenizer

        config = SemanticConfig(device="cpu", sample_rate=16000)
        tokenizer = SemanticTokenizer(config)

        audio = np.random.default_rng(42).standard_normal(16000).astype(np.float32)
        path = tmp_path / "mono_16k.wav"
        sf.write(str(path), audio, 16000)

        loaded = tokenizer._load_audio(path)
        assert loaded.dtype == np.float32
        assert len(loaded) == 16000

    def test_load_resample_24k_to_16k(self, tmp_path: Path) -> None:
        from voxid.tokenizer.semantic import SemanticTokenizer

        config = SemanticConfig(device="cpu", sample_rate=16000)
        tokenizer = SemanticTokenizer(config)

        audio = np.random.default_rng(42).standard_normal(24000).astype(np.float32)
        path = tmp_path / "24k.wav"
        sf.write(str(path), audio, 24000)

        loaded = tokenizer._load_audio(path)
        expected_len = int(1.0 * 16000)  # 1 second at 16kHz
        assert len(loaded) == expected_len

    def test_load_stereo_downmix(self, tmp_path: Path) -> None:
        from voxid.tokenizer.semantic import SemanticTokenizer

        config = SemanticConfig(device="cpu", sample_rate=16000)
        tokenizer = SemanticTokenizer(config)

        rng = np.random.default_rng(42)
        stereo = rng.standard_normal((32000, 2)).astype(np.float32)
        path = tmp_path / "stereo.wav"
        sf.write(str(path), stereo, 16000)

        loaded = tokenizer._load_audio(path)
        assert loaded.ndim == 1


class TestSemanticConfig:
    def test_defaults(self) -> None:
        config = SemanticConfig()
        assert config.device == "cpu"
        assert config.frame_rate == 50.0
        assert config.n_clusters == 500
        assert config.layer == 6
        assert config.sample_rate == 16000

    def test_frozen(self) -> None:
        config = SemanticConfig()
        with pytest.raises(AttributeError):
            config.device = "cuda"  # type: ignore[misc]

    def test_custom_cluster_path(self, tmp_path: Path) -> None:
        path = tmp_path / "clusters.npz"
        config = SemanticConfig(cluster_path=path)
        assert config.cluster_path == path
