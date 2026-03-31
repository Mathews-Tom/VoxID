from __future__ import annotations

import logging

import numpy as np
import pyloudnorm as pyln

from voxid.audio_utils import resample_linear

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


class AudioPreprocessor:
    """Audio preprocessing pipeline for enrollment recordings.

    Pipeline: resample → mono → trim silence → normalize loudness.
    No noise suppression — per Wildspoof 2026 findings, enhancement
    degrades speaker similarity in voice cloning.
    """

    def __init__(
        self,
        target_sr: int = 24000,
        target_lufs: float = -16.0,
        silence_threshold_db: float = -40.0,
        silence_pad_ms: int = 200,
    ) -> None:
        self._target_sr = target_sr
        self._target_lufs = target_lufs
        self._silence_threshold_db = silence_threshold_db
        self._silence_pad_ms = silence_pad_ms

    @property
    def target_sr(self) -> int:
        return self._target_sr

    @property
    def target_lufs(self) -> float:
        return self._target_lufs

    def process(
        self, audio: np.ndarray, sr: int,
    ) -> tuple[np.ndarray, int]:
        """Full pipeline: mono → resample → trim silence → normalize."""
        result = audio.astype(np.float64)

        if result.ndim > 1:
            result = self.to_mono(result)

        if sr != self._target_sr:
            result = self.resample(result, sr, self._target_sr)
            sr = self._target_sr

        result = self.trim_silence(result, sr)
        result = self.normalize_loudness(result, sr)

        return result, sr

    def resample(
        self,
        audio: np.ndarray,
        sr_in: int,
        sr_out: int,
    ) -> np.ndarray:
        """Resample audio using linear interpolation."""
        if sr_in == sr_out:
            return audio

        duration = len(audio) / sr_in
        n_out = int(duration * sr_out)
        if n_out == 0:
            return np.zeros(0, dtype=audio.dtype)

        return resample_linear(audio, n_out)

    def to_mono(self, audio: np.ndarray) -> np.ndarray:
        """Convert multi-channel audio to mono by averaging channels."""
        if audio.ndim == 1:
            return audio
        result: np.ndarray = audio.mean(axis=1)
        return result

    def trim_silence(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Trim leading and trailing silence with configurable padding."""
        if len(audio) == 0:
            return audio

        threshold = 10.0 ** (self._silence_threshold_db / 20.0)
        frame_ms = 10
        frame_size = max(1, int(sr * frame_ms / 1000))
        n_frames = len(audio) // frame_size

        if n_frames == 0:
            return audio

        # Find first and last frame above threshold — vectorized
        frames = audio[: n_frames * frame_size].reshape(n_frames, frame_size)
        rms_values = np.sqrt(np.mean(frames.astype(np.float64) ** 2, axis=1))
        active = rms_values > threshold

        if not active.any():
            # All silence — return minimal audio (one frame)
            return audio[:frame_size]

        first_active = int(np.argmax(active))
        last_active = int(n_frames - 1 - np.argmax(active[::-1]))

        pad_samples = int(sr * self._silence_pad_ms / 1000)
        start = max(0, first_active * frame_size - pad_samples)
        end = min(len(audio), (last_active + 1) * frame_size + pad_samples)

        return audio[start:end]

    def normalize_loudness(
        self, audio: np.ndarray, sr: int,
    ) -> np.ndarray:
        """Normalize audio to target LUFS using ITU-R BS.1770-4."""
        if len(audio) == 0:
            return audio

        audio_f = audio.astype(np.float64)

        # pyloudnorm requires at least 0.4s of audio
        min_samples = int(0.4 * sr)
        if len(audio_f) < min_samples:
            # For very short audio, use simple peak normalization
            peak = float(np.max(np.abs(audio_f)))
            if peak < _EPSILON:
                return audio_f
            target_peak = 10.0 ** (self._target_lufs / 20.0)
            scaled: np.ndarray = audio_f * (target_peak / peak)
            return scaled

        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio_f)

        if not np.isfinite(loudness):
            return audio_f

        normalized: np.ndarray = pyln.normalize.loudness(
            audio_f, loudness, self._target_lufs,
        )
        return normalized
