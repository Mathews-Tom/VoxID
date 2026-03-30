from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray

from .config import SpoofingConfig
from .features import resample_if_needed
from .models import (
    AASISTWrapper,
    LCNNWrapper,
    RawNet2Wrapper,
    load_available_models,
)
from .types import ArtifactType, SpoofDecision, SpoofingUnavailableError, SpoofLabel

logger = logging.getLogger(__name__)


class SynthesisDetector:
    """Ensemble anti-spoofing detector combining AASIST, RawNet2, and LCNN.

    Scores audio against all available models, computes a weighted
    average, and applies threshold + agreement logic to produce a
    three-class decision (GENUINE / SYNTHETIC / UNCERTAIN).
    """

    def __init__(self, config: SpoofingConfig | None = None) -> None:
        self._config = config or SpoofingConfig()
        self._models: list[AASISTWrapper | RawNet2Wrapper | LCNNWrapper] = (
            load_available_models(self._config)
        )
        if not self._models:
            raise SpoofingUnavailableError(
                "No anti-spoofing models found. Ensure at least one of "
                f"aasist.pth, rawnet2.pth, lcnn.pth exists in "
                f"{self._config.weights_dir}"
            )

    @property
    def available_models(self) -> list[str]:
        return [m.name for m in self._models]

    def detect(
        self, audio: NDArray[np.floating], sr: int
    ) -> SpoofDecision:
        """Run the anti-spoofing ensemble on an audio sample.

        Args:
            audio: Mono waveform, any float dtype.
            sr: Sample rate in Hz.

        Returns:
            SpoofDecision with label, score, artifact type, and
            per-model scores.
        """
        audio_f32, target_sr = resample_if_needed(audio, sr, self._config)

        model_scores: dict[str, float] = {}
        for model in self._models:
            score = model.predict(audio_f32, target_sr)
            model_scores[model.name] = score

        ensemble_score = self._weighted_score(model_scores)
        agreement = self._compute_agreement(model_scores)
        label = self._classify(ensemble_score, agreement)
        artifact_type = (
            self._infer_artifact_type(model_scores)
            if label == SpoofLabel.SYNTHETIC
            else ArtifactType.UNKNOWN
        )

        return SpoofDecision(
            label=label,
            score=ensemble_score,
            artifact_type=artifact_type,
            model_scores=model_scores,
            confidence=agreement,
        )

    def _weighted_score(self, model_scores: dict[str, float]) -> float:
        """Compute weighted average score, renormalizing for missing models."""
        total_weight = 0.0
        weighted_sum = 0.0

        for name, score in model_scores.items():
            weight = self._config.ensemble_weights.get(name, 0.0)
            weighted_sum += weight * score
            total_weight += weight

        if total_weight == 0.0:
            return float(np.mean(list(model_scores.values())))

        return weighted_sum / total_weight

    def _compute_agreement(self, model_scores: dict[str, float]) -> float:
        """Fraction of models scoring above the synthetic threshold."""
        if not model_scores:
            return 0.0
        above = sum(
            1
            for s in model_scores.values()
            if s >= self._config.synthetic_threshold
        )
        return above / len(model_scores)

    def _classify(self, score: float, agreement: float) -> SpoofLabel:
        """Apply threshold + agreement logic for three-class decision."""
        if (
            score >= self._config.synthetic_threshold
            and agreement >= self._config.min_agreement
        ):
            return SpoofLabel.SYNTHETIC
        if score >= self._config.uncertain_threshold:
            return SpoofLabel.UNCERTAIN
        return SpoofLabel.GENUINE

    def _infer_artifact_type(
        self, model_scores: dict[str, float]
    ) -> ArtifactType:
        """Heuristic artifact type inference based on model score patterns.

        AASIST excels at vocoder detection. RawNet2 catches autoregressive
        patterns. LCNN is sensitive to concatenative artifacts. When all
        models agree strongly, diffusion is likely (produces broad
        spectral artifacts all models detect).
        """
        scores = model_scores
        aasist = scores.get("aasist", 0.0)
        rawnet2 = scores.get("rawnet2", 0.0)
        lcnn = scores.get("lcnn", 0.0)

        # All models strongly agree → likely diffusion
        all_scores = [s for s in scores.values() if s > 0]
        if all_scores and min(all_scores) > 0.8:
            return ArtifactType.DIFFUSION

        # Dominant model heuristics
        if aasist > rawnet2 and aasist > lcnn:
            return ArtifactType.VOCODER
        if rawnet2 > aasist and rawnet2 > lcnn:
            return ArtifactType.AUTOREGRESSIVE
        if lcnn > aasist and lcnn > rawnet2:
            return ArtifactType.CONCATENATIVE

        return ArtifactType.UNKNOWN
