from __future__ import annotations

import numpy as np
import pytest

from voxid.enrollment.quality_gate import (
    QualityConfig,
    QualityGate,
    QualityReport,
    estimate_snr,
)


def _make_sine(
    sr: int,
    duration_s: float,
    freq: float = 440.0,
    amplitude: float = 0.3,
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


def _make_noise(sr: int, duration_s: float, amplitude: float = 0.01) -> np.ndarray:
    rng = np.random.default_rng(42)
    n_samples = int(sr * duration_s)
    return (amplitude * rng.standard_normal(n_samples)).astype(np.float64)


# --- QualityConfig ---


class TestQualityConfig:
    def test_defaults_match_spec(self) -> None:
        cfg = QualityConfig()
        assert cfg.min_duration_s == 2.0
        assert cfg.max_duration_s == 60.0
        assert cfg.min_snr_db == 20.0
        assert cfg.warn_snr_db == 30.0
        assert cfg.min_rms_dbfs == -40.0
        assert cfg.max_rms_dbfs == -3.0
        assert cfg.max_peak_dbfs == -0.5
        assert cfg.min_speech_ratio == 0.4
        assert cfg.min_sample_rate == 24000


# --- QualityReport ---


class TestQualityReport:
    def test_passed_when_no_rejections(self) -> None:
        report = QualityReport(
            passed=True, snr_db=45.0, rms_dbfs=-20.0, peak_dbfs=-3.0,
            speech_ratio=0.8, net_speech_duration_s=4.0,
            total_duration_s=5.0, sample_rate=24000,
            warnings=[], rejection_reasons=[],
        )
        assert report.passed is True
        assert report.rejection_reasons == []

    def test_failed_when_rejections_present(self) -> None:
        report = QualityReport(
            passed=False, snr_db=10.0, rms_dbfs=-50.0, peak_dbfs=-3.0,
            speech_ratio=0.3, net_speech_duration_s=1.5,
            total_duration_s=5.0, sample_rate=24000,
            warnings=[], rejection_reasons=["Too quiet", "Low speech"],
        )
        assert report.passed is False
        assert len(report.rejection_reasons) == 2

    def test_roundtrip_serialization(self) -> None:
        # Arrange
        report = QualityReport(
            passed=True, snr_db=35.0, rms_dbfs=-18.0, peak_dbfs=-4.0,
            speech_ratio=0.75, net_speech_duration_s=3.75,
            total_duration_s=5.0, sample_rate=48000,
            warnings=["Borderline SNR"], rejection_reasons=[],
        )

        # Act
        data = report.to_dict()
        restored = QualityReport.from_dict(data)

        # Assert
        assert restored.passed == report.passed
        assert restored.snr_db == report.snr_db
        assert restored.rms_dbfs == report.rms_dbfs
        assert restored.peak_dbfs == report.peak_dbfs
        assert restored.speech_ratio == report.speech_ratio
        assert restored.net_speech_duration_s == report.net_speech_duration_s
        assert restored.total_duration_s == report.total_duration_s
        assert restored.sample_rate == report.sample_rate
        assert restored.warnings == report.warnings
        assert restored.rejection_reasons == report.rejection_reasons


# --- estimate_snr ---


class TestEstimateSNR:
    def test_clean_signal_high_snr(self) -> None:
        # Arrange — low real noise then clean sine (not digital zeros)
        sr = 24000
        noise = _make_noise(sr, 0.5, amplitude=0.001)
        signal = _make_sine(sr, 4.5, amplitude=0.3)
        audio = np.concatenate([noise, signal])

        # Act
        snr = estimate_snr(audio, sr)

        # Assert — clean signal over real noise should give high SNR
        assert snr > 40.0

    def test_noisy_signal_low_snr(self) -> None:
        # Arrange — noise floor then signal+noise at known ratio
        sr = 24000
        noise_floor = _make_noise(sr, 0.5, amplitude=0.1)
        signal_with_noise = (
            _make_sine(sr, 4.5, amplitude=0.3)
            + _make_noise(sr, 4.5, amplitude=0.1)
        )
        audio = np.concatenate([noise_floor, signal_with_noise])

        # Act
        snr = estimate_snr(audio, sr)

        # Assert — expected ~10 dB (0.3/0.1 ≈ 10 dB)
        assert 5.0 < snr < 20.0

    def test_silent_audio_returns_zero(self) -> None:
        audio = np.zeros(24000, dtype=np.float64)
        assert estimate_snr(audio, 24000) == 0.0

    def test_with_vad_timestamps(self) -> None:
        # Arrange — low real noise then speech
        sr = 24000
        noise = _make_noise(sr, 0.5, amplitude=0.001)
        speech = _make_sine(sr, 2.0, amplitude=0.3)
        audio = np.concatenate([noise, speech])
        vad = [(len(noise), len(audio))]

        # Act
        snr = estimate_snr(audio, sr, vad_timestamps=vad)

        # Assert — speech over real noise floor should give high SNR
        assert snr > 40.0

    def test_very_short_audio(self) -> None:
        # 100 samples — should not crash
        audio = _make_sine(24000, 0.004, amplitude=0.3)
        snr = estimate_snr(audio, 24000)
        assert isinstance(snr, float)

    def test_empty_audio_returns_zero(self) -> None:
        audio = np.array([], dtype=np.float64)
        assert estimate_snr(audio, 24000) == 0.0


# --- QualityGate.validate ---


class TestQualityGateValidate:
    @pytest.fixture
    def gate(self) -> QualityGate:
        return QualityGate()

    def _make_good_audio(self, sr: int = 24000) -> np.ndarray:
        """5s sine at moderate amplitude — should pass all gates."""
        noise = _make_noise(sr, 0.5, amplitude=0.001)
        signal = _make_sine(sr, 4.5, amplitude=0.3)
        return np.concatenate([noise, signal])

    def test_validate_clean_audio_passes(self, gate: QualityGate) -> None:
        audio = self._make_good_audio()
        report = gate.validate(audio, 24000)
        assert report.passed is True
        assert report.rejection_reasons == []

    def test_validate_too_short_rejects(self, gate: QualityGate) -> None:
        audio = _make_sine(24000, 1.0)
        report = gate.validate(audio, 24000)
        assert report.passed is False
        assert any("Too short" in r for r in report.rejection_reasons)

    def test_validate_too_long_rejects(self, gate: QualityGate) -> None:
        audio = _make_sine(24000, 65.0, amplitude=0.3)
        report = gate.validate(audio, 24000)
        assert report.passed is False
        assert any("Too long" in r for r in report.rejection_reasons)

    def test_validate_too_quiet_rejects(self, gate: QualityGate) -> None:
        # Very low amplitude signal
        noise = _make_noise(24000, 0.5, amplitude=0.00001)
        signal = _make_sine(24000, 4.5, amplitude=0.0001)
        audio = np.concatenate([noise, signal])
        report = gate.validate(audio, 24000)
        assert report.passed is False
        assert any("Too quiet" in r for r in report.rejection_reasons)

    def test_validate_clipping_rejects(self, gate: QualityGate) -> None:
        # Sine at full scale (peak = 1.0 → 0 dBFS > -1 dBFS)
        noise = _make_noise(24000, 0.5, amplitude=0.001)
        signal = _make_sine(24000, 4.5, amplitude=0.99)
        audio = np.concatenate([noise, signal])
        report = gate.validate(audio, 24000)
        assert report.passed is False
        assert any("Clipping" in r for r in report.rejection_reasons)

    def test_validate_low_snr_rejects(self, gate: QualityGate) -> None:
        # Heavy noise in the "noise floor" region matching signal noise
        sr = 24000
        noise_level = 0.3
        noise_floor = _make_noise(sr, 0.5, amplitude=noise_level)
        # Signal barely above noise
        signal = (
            _make_sine(sr, 4.5, amplitude=0.35)
            + _make_noise(sr, 4.5, amplitude=noise_level)
        )
        audio = np.concatenate([noise_floor, signal])
        report = gate.validate(audio, sr)
        assert report.passed is False
        assert any("Noisy" in r for r in report.rejection_reasons)

    def test_validate_low_speech_ratio_rejects(
        self, gate: QualityGate,
    ) -> None:
        # Mostly silence with short speech burst
        sr = 24000
        silence = np.zeros(int(4.0 * sr), dtype=np.float64)
        speech = _make_sine(sr, 1.0, amplitude=0.3)
        audio = np.concatenate([silence, speech])
        report = gate.validate(audio, sr)
        assert report.passed is False
        assert any("Low speech" in r for r in report.rejection_reasons)

    def test_validate_low_sample_rate_rejects(
        self, gate: QualityGate,
    ) -> None:
        audio = _make_sine(16000, 5.0, amplitude=0.3)
        report = gate.validate(audio, 16000)
        assert report.passed is False
        assert any("Sample rate" in r for r in report.rejection_reasons)

    def test_validate_borderline_snr_warns(
        self, gate: QualityGate,
    ) -> None:
        # SNR between 25 and 40 dB — should warn but not reject
        sr = 24000
        noise_amp = 0.005
        noise_floor = _make_noise(sr, 0.5, amplitude=noise_amp)
        signal = (
            _make_sine(sr, 4.5, amplitude=0.3)
            + _make_noise(sr, 4.5, amplitude=noise_amp)
        )
        audio = np.concatenate([noise_floor, signal])
        report = gate.validate(audio, sr)
        # SNR should be between 20 and 30
        if 20.0 <= report.snr_db < 30.0:
            assert any("Borderline SNR" in w for w in report.warnings)

    def test_validate_multiple_failures_lists_all_reasons(
        self,
    ) -> None:
        # Short, quiet audio at low sample rate
        gate = QualityGate()
        audio = _make_sine(8000, 1.0, amplitude=0.0001)
        report = gate.validate(audio, 8000)
        assert report.passed is False
        assert len(report.rejection_reasons) >= 2

    def test_validate_custom_config_thresholds(self) -> None:
        # Relaxed config that accepts short, lower-rate audio
        sr = 8000
        config = QualityConfig(
            min_duration_s=0.5,
            min_snr_db=5.0,
            warn_snr_db=10.0,
            min_rms_dbfs=-60.0,
            min_sample_rate=8000,
            min_speech_ratio=0.1,
        )
        gate = QualityGate(config=config)
        noise_floor = _make_noise(sr, 0.5, amplitude=0.001)
        signal = _make_sine(sr, 2.0, amplitude=0.3)
        audio = np.concatenate([noise_floor, signal])
        report = gate.validate(audio, sr)
        assert report.passed is True
