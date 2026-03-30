from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class SpoofLabel(Enum):
    """Three-class anti-spoofing decision label."""

    GENUINE = "genuine"
    SYNTHETIC = "synthetic"
    UNCERTAIN = "uncertain"


class ArtifactType(Enum):
    """Taxonomy of synthesis artifacts detected in audio."""

    VOCODER = "vocoder"
    AUTOREGRESSIVE = "autoregressive"
    DIFFUSION = "diffusion"
    CONCATENATIVE = "concatenative"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SpoofDecision:
    """Result of anti-spoofing analysis on an audio sample.

    Attributes:
        label: Three-class decision (genuine, synthetic, uncertain).
        score: Ensemble spoofing probability in [0, 1]. Higher = more
            likely synthetic.
        artifact_type: Detected synthesis method when label is SYNTHETIC.
        model_scores: Per-model raw scores for audit/debugging.
        confidence: Agreement ratio among ensemble members in [0, 1].
    """

    label: SpoofLabel
    score: float
    artifact_type: ArtifactType
    model_scores: dict[str, float]
    confidence: float


class SpoofingUnavailableError(RuntimeError):
    """Raised when no anti-spoofing models could be loaded."""
