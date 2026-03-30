from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class AcousticConfig:
    """Configuration for the acoustic tokenizer."""

    model_id: str = "novateur/WavTokenizer-medium-speech-75token"
    device: str = "cpu"
    frame_rate: float = 40.0
    n_codebooks: int = 1
    sample_rate: int = 24000


@dataclass(frozen=True)
class SemanticConfig:
    """Configuration for the semantic tokenizer."""

    model_id: str = "facebook/hubert-base-ls960"
    device: str = "cpu"
    frame_rate: float = 50.0
    n_clusters: int = 500
    layer: int = 6
    sample_rate: int = 16000
    cluster_path: Path | None = None


@dataclass(frozen=True)
class TokenizerConfig:
    """Top-level configuration for the unified tokenizer."""

    acoustic: AcousticConfig = field(default_factory=AcousticConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    projections_dir: Path = field(
        default_factory=lambda: Path.home() / ".voxid" / "projections",
    )
    cache_tokenized: bool = True
