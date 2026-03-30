from __future__ import annotations

from unittest.mock import patch

import numpy as np

from voxid.enrollment.quality_gate import QualityConfig, QualityGate, QualityReport
from voxid.security.spoofing.types import (
    ArtifactType,
    SpoofDecision,
    SpoofingUnavailableError,
    SpoofLabel,
)

# Relaxed config that accepts our test audio
_RELAXED_CONFIG = QualityConfig(min_snr_db=-10.0, warn_snr_db=-10.0)


def _valid_audio(sr: int = 24000, duration_s: float = 3.0) -> np.ndarray:
    """Generate audio that passes ALL basic quality gates.

    Creates a signal with clear speech regions and a quiet noise floor
    section so that the SNR estimator can distinguish signal from noise.
    """
    n_samples = int(sr * duration_s)
    t = np.linspace(0, duration_s, n_samples, endpoint=False)
    # Speech-like tonal signal
    signal = (
        0.15 * np.sin(2 * np.pi * 150 * t)
        + 0.10 * np.sin(2 * np.pi * 300 * t)
        + 0.05 * np.sin(2 * np.pi * 450 * t)
    ).astype(np.float32)
    # Insert a quiet noise-floor section (last 0.5s) so SNR estimator works
    noise_start = int(sr * (duration_s - 0.5))
    rng = np.random.default_rng(42)
    noise = rng.standard_normal(n_samples - noise_start) * 0.001
    signal[noise_start:] = noise.astype(np.float32)
    return signal


class TestQualityGateWithoutSpoofingExtra:
    """Verify existing behavior is unchanged when spoofing is not available."""

    def test_passes_without_spoofing_module(self) -> None:
        gate = QualityGate()
        audio = _valid_audio()
        with patch(
            "voxid.enrollment.quality_gate._SPOOFING_AVAILABLE", False
        ):
            report = gate.validate(audio, 24000)
        assert isinstance(report, QualityReport)
        assert "Synthetic speech detected" not in report.rejection_reasons


class TestQualityGateWithSpoofingGenuine:
    """Verify genuine audio passes the synthesis gate."""

    def test_genuine_passes(self) -> None:
        gate = QualityGate()
        audio = _valid_audio()

        with (
            patch(
                "voxid.enrollment.quality_gate._SPOOFING_AVAILABLE", True
            ),
            patch.object(
                QualityGate,
                "_check_synthesis",
                return_value="pass",
            ),
        ):
            report = gate.validate(audio, 24000)

        assert "Synthetic speech detected" not in report.rejection_reasons


class TestQualityGateWithSpoofingSynthetic:
    """Verify synthetic audio fails the gate."""

    def test_synthetic_fails(self) -> None:
        gate = QualityGate(config=_RELAXED_CONFIG)
        audio = _valid_audio()

        with (
            patch(
                "voxid.enrollment.quality_gate._SPOOFING_AVAILABLE", True
            ),
            patch.object(
                QualityGate,
                "_check_synthesis",
                return_value="fail",
            ),
        ):
            report = gate.validate(audio, 24000)

        assert not report.passed
        assert "Synthetic speech detected" in report.rejection_reasons


class TestQualityGateWithSpoofingUncertain:
    """Verify uncertain result produces warning, not rejection."""

    def test_uncertain_warns(self) -> None:
        gate = QualityGate(config=_RELAXED_CONFIG)
        audio = _valid_audio()

        with (
            patch(
                "voxid.enrollment.quality_gate._SPOOFING_AVAILABLE", True
            ),
            patch.object(
                QualityGate,
                "_check_synthesis",
                return_value="warn",
            ),
        ):
            report = gate.validate(audio, 24000)

        assert "Audio flagged as potentially synthetic" in report.warnings
        assert "Synthetic speech detected" not in report.rejection_reasons


class TestQualityGateSkipsSpoofingOnEarlyFailure:
    """Verify synthesis check is skipped when audio already failed."""

    def test_skips_when_too_short(self) -> None:
        gate = QualityGate()
        audio = np.zeros(100, dtype=np.float32)

        with patch(
            "voxid.enrollment.quality_gate._SPOOFING_AVAILABLE", True
        ):
            report = gate.validate(audio, 24000)

        assert not report.passed
        assert "Synthetic speech detected" not in report.rejection_reasons


class TestCheckSynthesisMethod:
    """Test _check_synthesis directly with patched imports."""

    def test_returns_pass_on_unavailable_error(self) -> None:
        gate = QualityGate()
        audio = _valid_audio()

        with patch(
            "voxid.security.spoofing.SynthesisDetector",
            side_effect=SpoofingUnavailableError("no weights"),
        ):
            result = gate._check_synthesis(audio, 24000)

        assert result == "pass"

    def test_returns_fail_on_synthetic(self) -> None:
        gate = QualityGate()
        audio = _valid_audio()
        synthetic_decision = SpoofDecision(
            label=SpoofLabel.SYNTHETIC,
            score=0.9,
            artifact_type=ArtifactType.VOCODER,
            model_scores={"aasist": 0.9},
            confidence=1.0,
        )

        with patch(
            "voxid.security.spoofing.SynthesisDetector"
        ) as mock_cls:
            mock_cls.return_value.detect.return_value = synthetic_decision
            result = gate._check_synthesis(audio, 24000)

        assert result == "fail"

    def test_returns_warn_on_uncertain(self) -> None:
        gate = QualityGate()
        audio = _valid_audio()
        uncertain_decision = SpoofDecision(
            label=SpoofLabel.UNCERTAIN,
            score=0.5,
            artifact_type=ArtifactType.UNKNOWN,
            model_scores={"aasist": 0.5},
            confidence=0.5,
        )

        with patch(
            "voxid.security.spoofing.SynthesisDetector"
        ) as mock_cls:
            mock_cls.return_value.detect.return_value = uncertain_decision
            result = gate._check_synthesis(audio, 24000)

        assert result == "warn"

    def test_returns_pass_on_genuine(self) -> None:
        gate = QualityGate()
        audio = _valid_audio()
        genuine_decision = SpoofDecision(
            label=SpoofLabel.GENUINE,
            score=0.1,
            artifact_type=ArtifactType.UNKNOWN,
            model_scores={"aasist": 0.1},
            confidence=1.0,
        )

        with patch(
            "voxid.security.spoofing.SynthesisDetector"
        ) as mock_cls:
            mock_cls.return_value.detect.return_value = genuine_decision
            result = gate._check_synthesis(audio, 24000)

        assert result == "pass"
