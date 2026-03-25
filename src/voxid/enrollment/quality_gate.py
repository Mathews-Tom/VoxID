from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

# Research-backed thresholds (scripted-enrollment-pipeline.md §9.1)
_EPSILON = 1e-9


@dataclass
class QualityConfig:
    """Thresholds for audio quality validation during enrollment."""

    min_duration_s: float = 2.0
    max_duration_s: float = 60.0
    min_snr_db: float = 20.0
    warn_snr_db: float = 30.0
    min_rms_dbfs: float = -40.0
    max_rms_dbfs: float = -3.0
    max_peak_dbfs: float = -0.5
    min_speech_ratio: float = 0.4
    min_sample_rate: int = 24000


@dataclass
class QualityReport:
    """Result of quality gate validation on an audio sample."""

    passed: bool
    snr_db: float
    rms_dbfs: float
    peak_dbfs: float
    speech_ratio: float
    net_speech_duration_s: float
    total_duration_s: float
    sample_rate: int
    warnings: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "snr_db": self.snr_db,
            "rms_dbfs": self.rms_dbfs,
            "peak_dbfs": self.peak_dbfs,
            "speech_ratio": self.speech_ratio,
            "net_speech_duration_s": self.net_speech_duration_s,
            "total_duration_s": self.total_duration_s,
            "sample_rate": self.sample_rate,
            "warnings": list(self.warnings),
            "rejection_reasons": list(self.rejection_reasons),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> QualityReport:
        return cls(
            passed=data["passed"],
            snr_db=data["snr_db"],
            rms_dbfs=data["rms_dbfs"],
            peak_dbfs=data["peak_dbfs"],
            speech_ratio=data["speech_ratio"],
            net_speech_duration_s=data["net_speech_duration_s"],
            total_duration_s=data["total_duration_s"],
            sample_rate=data["sample_rate"],
            warnings=list(data.get("warnings", [])),
            rejection_reasons=list(data.get("rejection_reasons", [])),
        )


def estimate_snr(
    audio: np.ndarray,
    sr: int,
    vad_timestamps: list[tuple[int, int]] | None = None,
    noise_window_s: float = 0.5,
) -> float:
    """Estimate signal-to-noise ratio in dB.

    Noise floor is estimated from the quietest window in the recording
    (sliding window of `noise_window_s` duration). This avoids false
    readings from mic startup silence where the first N ms are digital
    zeros.

    When VAD timestamps are provided, non-speech segments are used
    directly as the noise floor instead.
    """
    if len(audio) == 0:
        return 0.0

    audio_f = audio.astype(np.float64)

    if vad_timestamps is not None and len(vad_timestamps) > 0:
        # Use non-speech segments as noise floor
        non_speech_parts: list[np.ndarray] = []
        prev_end = 0
        for start, end in vad_timestamps:
            if start > prev_end:
                non_speech_parts.append(audio_f[prev_end:start])
            prev_end = end
        if prev_end < len(audio_f):
            non_speech_parts.append(audio_f[prev_end:])

        if non_speech_parts:
            noise = np.concatenate(non_speech_parts)
        else:
            noise = _find_quietest_window(audio_f, sr, noise_window_s)

        speech_parts = [
            audio_f[start:end] for start, end in vad_timestamps
        ]
        speech = np.concatenate(speech_parts) if speech_parts else audio_f
    else:
        noise = _find_quietest_window(audio_f, sr, noise_window_s)
        speech = audio_f

    noise_rms = float(np.sqrt(np.mean(noise ** 2)))
    speech_rms = float(np.sqrt(np.mean(speech ** 2)))

    if noise_rms < _EPSILON:
        # Noise floor is digital silence — cannot compute meaningful SNR.
        # Return 0 to force a warning rather than a fake perfect score.
        return 0.0

    snr = float(20.0 * np.log10((speech_rms + _EPSILON) / noise_rms))
    # Cap at a realistic maximum — no consumer mic achieves >80 dB SNR
    return min(snr, 80.0)


def _find_quietest_window(
    audio: np.ndarray,
    sr: int,
    window_s: float,
) -> np.ndarray:
    """Find the window with the lowest RMS energy in the audio.

    Slides a window of `window_s` seconds across the audio in
    non-overlapping hops and returns the one with the smallest RMS.
    Skips windows that are digital silence (all zeros) since those
    indicate mic startup artifacts, not real noise floor.
    """
    window_samples = int(window_s * sr)
    if window_samples <= 0 or len(audio) <= window_samples:
        return audio

    best_window = audio[:window_samples]
    best_rms = float("inf")

    for offset in range(0, len(audio) - window_samples, window_samples):
        chunk = audio[offset : offset + window_samples]
        chunk_max = float(np.max(np.abs(chunk)))
        # Skip digital silence — not a real noise measurement
        if chunk_max < _EPSILON:
            continue
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        if rms < best_rms:
            best_rms = rms
            best_window = chunk

    return best_window


def _compute_speech_ratio(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    energy_threshold_db: float = -40.0,
) -> float:
    """Energy-based VAD: fraction of frames above energy threshold."""
    frame_size = int(sr * frame_ms / 1000)
    if frame_size == 0 or len(audio) == 0:
        return 0.0

    audio_f = audio.astype(np.float64)
    n_frames = len(audio_f) // frame_size
    if n_frames == 0:
        return 0.0

    threshold_linear = 10.0 ** (energy_threshold_db / 20.0)
    speech_frames = 0

    for i in range(n_frames):
        frame = audio_f[i * frame_size : (i + 1) * frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        if rms > threshold_linear:
            speech_frames += 1

    return speech_frames / n_frames


class QualityGate:
    """Validates audio samples against quality thresholds for enrollment.

    Checks (in order): sample rate, duration, RMS level, peak level,
    SNR, and speech ratio. Each failed check is recorded as a rejection
    reason; borderline values produce warnings.
    """

    def __init__(self, config: QualityConfig | None = None) -> None:
        self._config = config or QualityConfig()

    @property
    def config(self) -> QualityConfig:
        return self._config

    def validate(self, audio: np.ndarray, sr: int) -> QualityReport:
        """Run all quality gates on an audio sample."""
        reasons: list[str] = []
        warnings: list[str] = []
        audio_f = audio.astype(np.float64)

        # 1. Sample rate
        if sr < self._config.min_sample_rate:
            reasons.append(
                f"Sample rate {sr} Hz < {self._config.min_sample_rate} Hz"
            )

        # 2. Duration
        duration = len(audio_f) / sr
        if duration < self._config.min_duration_s:
            reasons.append(
                f"Too short: {duration:.1f}s < {self._config.min_duration_s}s"
            )
        elif duration > self._config.max_duration_s:
            reasons.append(
                f"Too long: {duration:.1f}s > {self._config.max_duration_s}s"
            )

        # 3. RMS level
        rms = float(np.sqrt(np.mean(audio_f ** 2)))
        rms_dbfs = float(20.0 * np.log10(rms + _EPSILON))
        if rms_dbfs < self._config.min_rms_dbfs:
            reasons.append(
                f"Too quiet: {rms_dbfs:.1f} dBFS < {self._config.min_rms_dbfs} dBFS"
            )
        elif rms_dbfs > self._config.max_rms_dbfs:
            reasons.append(
                f"Too loud: {rms_dbfs:.1f} dBFS > {self._config.max_rms_dbfs} dBFS"
            )

        # 4. Peak level (clipping detection)
        peak = float(np.max(np.abs(audio_f)))
        peak_dbfs = float(20.0 * np.log10(peak + _EPSILON))
        if peak_dbfs > self._config.max_peak_dbfs:
            reasons.append(
                f"Clipping: peak {peak_dbfs:.1f} dBFS"
                f" > {self._config.max_peak_dbfs} dBFS"
            )

        # 5. SNR
        snr = estimate_snr(audio_f, sr)
        if snr < self._config.min_snr_db:
            reasons.append(
                f"Noisy: SNR {snr:.1f} dB < {self._config.min_snr_db} dB"
            )
        elif snr < self._config.warn_snr_db:
            warnings.append(
                f"Borderline SNR: {snr:.1f} dB < {self._config.warn_snr_db} dB"
            )

        # 6. Speech ratio
        speech_ratio = _compute_speech_ratio(audio_f, sr)
        if speech_ratio < self._config.min_speech_ratio:
            reasons.append(
                f"Low speech: {speech_ratio:.0%} < {self._config.min_speech_ratio:.0%}"
            )

        net_speech = duration * speech_ratio

        return QualityReport(
            passed=len(reasons) == 0,
            snr_db=snr,
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            speech_ratio=speech_ratio,
            net_speech_duration_s=net_speech,
            total_duration_s=duration,
            sample_rate=sr,
            warnings=warnings,
            rejection_reasons=reasons,
        )
