from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import soundfile as sf

from voxid.tokenizer.config import TokenizerConfig
from voxid.tokenizer.types import (
    AcousticTokens,
    SemanticTokens,
    TokenizedSpeaker,
)
from voxid.tokenizer.unified import UnifiedTokenizer


def _make_audio(tmp_path: Path, duration: float = 1.0, sr: int = 24000) -> Path:
    """Create a test WAV file."""
    rng = np.random.default_rng(42)
    samples = int(duration * sr)
    audio = rng.standard_normal(samples).astype(np.float32) * 0.1
    path = tmp_path / "test_audio.wav"
    sf.write(str(path), audio, sr)
    return path


def _stub_acoustic_tokens() -> AcousticTokens:
    return AcousticTokens(
        codes=np.array([[1, 2, 3, 4, 5]], dtype=np.int64),
        frame_rate=40.0,
        embedding=np.random.default_rng(42).standard_normal(64).astype(np.float32),
        sample_rate=24000,
    )


def _stub_semantic_tokens() -> SemanticTokens:
    return SemanticTokens(
        codes=np.array([10, 20, 30, 40, 50], dtype=np.int64),
        frame_rate=50.0,
        features=np.random.default_rng(42).standard_normal((5, 128)).astype(np.float32),
    )


class TestUnifiedTokenizerBuildEmbedding:
    def test_concatenation_shape(self) -> None:
        config = TokenizerConfig()
        tokenizer = UnifiedTokenizer(config)

        acoustic = _stub_acoustic_tokens()
        semantic = _stub_semantic_tokens()

        unified = tokenizer._build_unified_embedding(acoustic, semantic)

        # acoustic.embedding (64,) + semantic mean-pooled (128,) = (192,)
        assert unified.shape == (64 + 128,)
        assert unified.dtype == np.float32

    def test_embedding_consistency(self) -> None:
        config = TokenizerConfig()
        tokenizer = UnifiedTokenizer(config)

        acoustic = _stub_acoustic_tokens()
        semantic = _stub_semantic_tokens()

        e1 = tokenizer._build_unified_embedding(acoustic, semantic)
        e2 = tokenizer._build_unified_embedding(acoustic, semantic)

        np.testing.assert_array_equal(e1, e2)


class TestSaveLoadTokenized:
    def test_roundtrip(self, tmp_path: Path) -> None:
        config = TokenizerConfig()
        tokenizer = UnifiedTokenizer(config)

        rng = np.random.default_rng(42)
        speaker = TokenizedSpeaker(
            identity_id="alice",
            style_id="conversational",
            acoustic=_stub_acoustic_tokens(),
            semantic=_stub_semantic_tokens(),
            unified_embedding=rng.standard_normal(192).astype(np.float32),
            duration_seconds=1.5,
            metadata={"model_version": "v1"},
        )

        path = tmp_path / "test_unified.safetensors"
        tokenizer.save_tokenized(speaker, path)
        assert path.exists()

        loaded = tokenizer.load_tokenized(path)

        assert loaded.identity_id == "alice"
        assert loaded.style_id == "conversational"
        assert abs(loaded.duration_seconds - 1.5) < 1e-6
        np.testing.assert_allclose(
            loaded.unified_embedding, speaker.unified_embedding, atol=1e-6,
        )
        np.testing.assert_array_equal(
            loaded.acoustic.codes, speaker.acoustic.codes,
        )
        np.testing.assert_array_equal(
            loaded.semantic.codes, speaker.semantic.codes,
        )
        assert loaded.metadata["model_version"] == "v1"

    def test_load_missing_raises(self, tmp_path: Path) -> None:
        config = TokenizerConfig()
        tokenizer = UnifiedTokenizer(config)

        with pytest.raises(FileNotFoundError):
            tokenizer.load_tokenized(tmp_path / "nonexistent.safetensors")


class TestTokenize:
    def test_tokenize_produces_speaker(self, tmp_path: Path) -> None:
        """Integration-style test using patched encode methods."""
        config = TokenizerConfig()
        tokenizer = UnifiedTokenizer(config)

        audio_path = _make_audio(tmp_path)

        acoustic_tokens = _stub_acoustic_tokens()
        semantic_tokens = _stub_semantic_tokens()

        with (
            patch.object(
                tokenizer._acoustic,
                "encode",
                return_value=acoustic_tokens,
            ),
            patch.object(
                tokenizer._semantic,
                "encode",
                return_value=semantic_tokens,
            ),
        ):
            speaker = tokenizer.tokenize(
                audio_path,
                identity_id="bob",
                style_id="formal",
                metadata={"test": "true"},
            )

        assert isinstance(speaker, TokenizedSpeaker)
        assert speaker.identity_id == "bob"
        assert speaker.style_id == "formal"
        assert speaker.unified_embedding.shape == (64 + 128,)
        assert speaker.duration_seconds > 0.0
        assert speaker.metadata == {"test": "true"}

    def test_speaker_similarity_with_patched_encoders(
        self, tmp_path: Path,
    ) -> None:
        config = TokenizerConfig()
        tokenizer = UnifiedTokenizer(config)

        audio_a = _make_audio(tmp_path)
        audio_b = tmp_path / "audio_b.wav"
        sf.write(str(audio_b), np.zeros(24000, dtype=np.float32), 24000)

        acoustic_tokens = _stub_acoustic_tokens()
        semantic_tokens = _stub_semantic_tokens()

        with (
            patch.object(
                tokenizer._acoustic,
                "encode",
                return_value=acoustic_tokens,
            ),
            patch.object(
                tokenizer._semantic,
                "encode",
                return_value=semantic_tokens,
            ),
        ):
            similarity = tokenizer.speaker_similarity(audio_a, audio_b)

        # Same stubs → identical embeddings → similarity ≈ 1.0
        assert similarity > 0.999
