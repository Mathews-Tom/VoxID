from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from voxid.serialization import load_prompt, save_prompt

from .acoustic import AcousticTokenizer
from .config import TokenizerConfig
from .semantic import SemanticTokenizer
from .types import AcousticTokens, SemanticTokens, TokenizedSpeaker


class UnifiedTokenizer:
    """Combines acoustic and semantic tokenizers into a single
    engine-agnostic speaker representation.

    Produces a unified embedding by concatenating the acoustic
    speaker embedding with the mean-pooled semantic features.
    """

    def __init__(self, config: TokenizerConfig | None = None) -> None:
        self._config = config or TokenizerConfig()
        self._acoustic = AcousticTokenizer(self._config.acoustic)
        self._semantic = SemanticTokenizer(self._config.semantic)

    @property
    def acoustic(self) -> AcousticTokenizer:
        return self._acoustic

    @property
    def semantic(self) -> SemanticTokenizer:
        return self._semantic

    def tokenize(
        self,
        audio_path: Path,
        identity_id: str,
        style_id: str,
        metadata: dict[str, str] | None = None,
    ) -> TokenizedSpeaker:
        """Tokenize an audio file into a full speaker representation.

        Runs both acoustic and semantic tokenizers, then constructs
        a unified embedding by concatenating acoustic speaker embedding
        with mean-pooled semantic features.
        """
        import soundfile as sf

        info = sf.info(str(audio_path))
        duration_seconds = float(info.duration)

        acoustic_tokens = self._acoustic.encode(audio_path)
        semantic_tokens = self._semantic.encode(audio_path)

        unified_embedding = self._build_unified_embedding(
            acoustic_tokens,
            semantic_tokens,
        )

        return TokenizedSpeaker(
            identity_id=identity_id,
            style_id=style_id,
            acoustic=acoustic_tokens,
            semantic=semantic_tokens,
            unified_embedding=unified_embedding,
            duration_seconds=duration_seconds,
            metadata=metadata or {},
        )

    def _build_unified_embedding(
        self,
        acoustic: AcousticTokens,
        semantic: SemanticTokens,
    ) -> NDArray[np.float32]:
        """Concatenate acoustic embedding with mean-pooled semantic features."""
        semantic_embedding = semantic.features.mean(axis=0).astype(np.float32)
        return np.concatenate(
            [acoustic.embedding, semantic_embedding],
        ).astype(np.float32)

    def speaker_similarity(
        self,
        audio_a: Path,
        audio_b: Path,
    ) -> float:
        """Compute speaker similarity between two audio files.

        Uses the unified embedding (acoustic + semantic) and
        returns cosine similarity in [-1, 1].
        """
        tok_a = self.tokenize(audio_a, identity_id="", style_id="")
        tok_b = self.tokenize(audio_b, identity_id="", style_id="")
        return tok_a.speaker_similarity(tok_b)

    def save_tokenized(
        self,
        speaker: TokenizedSpeaker,
        output_path: Path,
    ) -> None:
        """Save a TokenizedSpeaker to a SafeTensors file.

        Stores the unified embedding, acoustic codes, acoustic embedding,
        semantic codes, and semantic features as named tensors.
        """
        tensors: dict[str, np.ndarray] = {
            "unified_embedding": speaker.unified_embedding,
            "acoustic_codes": speaker.acoustic.codes,
            "acoustic_embedding": speaker.acoustic.embedding,
            "semantic_codes": speaker.semantic.codes.reshape(1, -1),
            "semantic_features": speaker.semantic.features,
        }
        metadata: dict[str, str] = {
            "identity_id": speaker.identity_id,
            "style_id": speaker.style_id,
            "duration_seconds": str(speaker.duration_seconds),
            "acoustic_frame_rate": str(speaker.acoustic.frame_rate),
            "acoustic_sample_rate": str(speaker.acoustic.sample_rate),
            "semantic_frame_rate": str(speaker.semantic.frame_rate),
            **speaker.metadata,
        }
        save_prompt(tensors, output_path, metadata=metadata)

    def load_tokenized(self, path: Path) -> TokenizedSpeaker:
        """Load a TokenizedSpeaker from a SafeTensors file."""
        tensors, metadata = load_prompt(path)

        acoustic = AcousticTokens(
            codes=tensors["acoustic_codes"].astype(np.int64),
            frame_rate=float(metadata.get("acoustic_frame_rate", "40.0")),
            embedding=tensors["acoustic_embedding"].astype(np.float32),
            sample_rate=int(metadata.get("acoustic_sample_rate", "24000")),
        )
        semantic = SemanticTokens(
            codes=tensors["semantic_codes"].reshape(-1).astype(np.int64),
            frame_rate=float(metadata.get("semantic_frame_rate", "50.0")),
            features=tensors["semantic_features"].astype(np.float32),
        )

        return TokenizedSpeaker(
            identity_id=metadata.get("identity_id", ""),
            style_id=metadata.get("style_id", ""),
            acoustic=acoustic,
            semantic=semantic,
            unified_embedding=tensors["unified_embedding"].astype(np.float32),
            duration_seconds=float(metadata.get("duration_seconds", "0.0")),
            metadata={
                k: v
                for k, v in metadata.items()
                if k
                not in {
                    "identity_id",
                    "style_id",
                    "duration_seconds",
                    "acoustic_frame_rate",
                    "acoustic_sample_rate",
                    "semantic_frame_rate",
                }
            },
        )
