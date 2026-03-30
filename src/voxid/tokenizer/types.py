from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class AcousticTokens:
    """Causal acoustic token sequence from WavTokenizer.

    Attributes:
        codes: Integer token IDs, shape (n_codebooks, T).
        frame_rate: Tokens per second (e.g. 40 Hz for WavTokenizer).
        embedding: Continuous speaker embedding derived from temporal
            averaging of codec hidden states, shape (embed_dim,).
        sample_rate: Audio sample rate used during encoding.
    """

    codes: NDArray[np.int64]
    frame_rate: float
    embedding: NDArray[np.float32]
    sample_rate: int


@dataclass(frozen=True)
class SemanticTokens:
    """Quantized semantic token sequence from HuBERT.

    Attributes:
        codes: Cluster IDs from k-means quantization, shape (T,).
        frame_rate: Tokens per second (e.g. 50 Hz for HuBERT).
        features: Raw HuBERT hidden states before quantization,
            shape (T, hidden_dim). Retained for downstream alignment.
    """

    codes: NDArray[np.int64]
    frame_rate: float
    features: NDArray[np.float32]


@dataclass(frozen=True)
class TokenizedSpeaker:
    """Combined acoustic + semantic representation of a speaker sample.

    Attributes:
        identity_id: VoxID identity this sample belongs to.
        style_id: Style within that identity.
        acoustic: Causal acoustic tokens.
        semantic: Semantic tokens.
        unified_embedding: Concatenation of acoustic embedding and
            mean-pooled semantic features, shape (unified_dim,).
        duration_seconds: Duration of the source audio.
        metadata: Arbitrary key-value pairs (model versions, etc.).
    """

    identity_id: str
    style_id: str
    acoustic: AcousticTokens
    semantic: SemanticTokens
    unified_embedding: NDArray[np.float32]
    duration_seconds: float
    metadata: dict[str, str] = field(default_factory=dict)

    def speaker_similarity(self, other: TokenizedSpeaker) -> float:
        """Cosine similarity between unified embeddings."""
        a = self.unified_embedding
        b = other.unified_embedding
        norm_a = float(np.linalg.norm(a))
        norm_b = float(np.linalg.norm(b))
        if norm_a == 0.0 or norm_b == 0.0:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
