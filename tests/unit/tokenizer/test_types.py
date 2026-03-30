from __future__ import annotations

import numpy as np

from voxid.tokenizer.types import (
    AcousticTokens,
    SemanticTokens,
    TokenizedSpeaker,
)


def _make_acoustic(embed: np.ndarray | None = None) -> AcousticTokens:
    return AcousticTokens(
        codes=np.array([[1, 2, 3]], dtype=np.int64),
        frame_rate=40.0,
        embedding=embed if embed is not None else np.ones(64, dtype=np.float32),
        sample_rate=24000,
    )


def _make_semantic(features: np.ndarray | None = None) -> SemanticTokens:
    return SemanticTokens(
        codes=np.array([10, 20, 30], dtype=np.int64),
        frame_rate=50.0,
        features=(
            features if features is not None
            else np.ones((3, 128), dtype=np.float32)
        ),
    )


class TestAcousticTokens:
    def test_frozen(self) -> None:
        tok = _make_acoustic()
        try:
            tok.frame_rate = 50.0  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_codes_shape(self) -> None:
        tok = _make_acoustic()
        assert tok.codes.shape == (1, 3)

    def test_embedding_dtype(self) -> None:
        tok = _make_acoustic()
        assert tok.embedding.dtype == np.float32


class TestSemanticTokens:
    def test_frozen(self) -> None:
        tok = _make_semantic()
        try:
            tok.frame_rate = 40.0  # type: ignore[misc]
            raised = False
        except AttributeError:
            raised = True
        assert raised

    def test_features_shape(self) -> None:
        tok = _make_semantic()
        assert tok.features.shape == (3, 128)


class TestTokenizedSpeaker:
    def test_speaker_similarity_identical(self) -> None:
        embed = np.random.randn(192).astype(np.float32)
        speaker = TokenizedSpeaker(
            identity_id="alice",
            style_id="conversational",
            acoustic=_make_acoustic(),
            semantic=_make_semantic(),
            unified_embedding=embed,
            duration_seconds=3.5,
        )
        assert speaker.speaker_similarity(speaker) > 0.999

    def test_speaker_similarity_orthogonal(self) -> None:
        embed_a = np.zeros(192, dtype=np.float32)
        embed_a[0] = 1.0
        embed_b = np.zeros(192, dtype=np.float32)
        embed_b[1] = 1.0

        speaker_a = TokenizedSpeaker(
            identity_id="a",
            style_id="s",
            acoustic=_make_acoustic(),
            semantic=_make_semantic(),
            unified_embedding=embed_a,
            duration_seconds=1.0,
        )
        speaker_b = TokenizedSpeaker(
            identity_id="b",
            style_id="s",
            acoustic=_make_acoustic(),
            semantic=_make_semantic(),
            unified_embedding=embed_b,
            duration_seconds=1.0,
        )
        assert abs(speaker_a.speaker_similarity(speaker_b)) < 1e-6

    def test_speaker_similarity_zero_vector(self) -> None:
        embed = np.zeros(192, dtype=np.float32)
        speaker = TokenizedSpeaker(
            identity_id="a",
            style_id="s",
            acoustic=_make_acoustic(),
            semantic=_make_semantic(),
            unified_embedding=embed,
            duration_seconds=1.0,
        )
        assert speaker.speaker_similarity(speaker) == 0.0

    def test_metadata_default(self) -> None:
        speaker = TokenizedSpeaker(
            identity_id="a",
            style_id="s",
            acoustic=_make_acoustic(),
            semantic=_make_semantic(),
            unified_embedding=np.ones(64, dtype=np.float32),
            duration_seconds=1.0,
        )
        assert speaker.metadata == {}

    def test_metadata_preserved(self) -> None:
        speaker = TokenizedSpeaker(
            identity_id="a",
            style_id="s",
            acoustic=_make_acoustic(),
            semantic=_make_semantic(),
            unified_embedding=np.ones(64, dtype=np.float32),
            duration_seconds=1.0,
            metadata={"model": "wavtokenizer-v1"},
        )
        assert speaker.metadata == {"model": "wavtokenizer-v1"}
