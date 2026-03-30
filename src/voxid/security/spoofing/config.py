from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class SpoofingConfig:
    """Configuration for the anti-spoofing ensemble detector.

    Attributes:
        weights_dir: Directory containing model weight files. Models
            look for ``aasist.pth``, ``rawnet2.pth``, ``lcnn.pth``.
        ensemble_weights: Per-model contribution to the final score.
            Missing models are excluded and remaining weights are
            renormalized to sum to 1.0.
        synthetic_threshold: Score above which audio is classified
            as SYNTHETIC.
        uncertain_threshold: Score above which (but below
            ``synthetic_threshold``) audio is classified as UNCERTAIN.
        min_agreement: Minimum fraction of models that must agree on
            the synthetic label for an ensemble SYNTHETIC decision.
        sample_rate: Expected sample rate for feature extraction.
        chunk_duration_s: Duration of audio chunks for RawNet2.
        chunk_overlap: Fractional overlap between consecutive chunks.
    """

    weights_dir: Path = field(default_factory=lambda: Path("weights/spoofing"))
    ensemble_weights: dict[str, float] = field(
        default_factory=lambda: {
            "aasist": 0.4,
            "rawnet2": 0.35,
            "lcnn": 0.25,
        }
    )
    synthetic_threshold: float = 0.7
    uncertain_threshold: float = 0.4
    min_agreement: float = 0.5
    sample_rate: int = 16000
    chunk_duration_s: float = 4.0
    chunk_overlap: float = 0.5
