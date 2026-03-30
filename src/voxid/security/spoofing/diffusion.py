from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .features import extract_mel_spectrogram


@dataclass(frozen=True)
class DiffusionAnalysis:
    """Result of diffusion-specific artifact analysis.

    Attributes:
        spectral_smoothness: Ratio of high-frequency energy to total
            energy. Low values suggest over-smoothing from diffusion
            denoising. Range [0, 1].
        temporal_discontinuity: Maximum spectral flux jump between
            adjacent frames, normalized. High values indicate chunk
            boundary artifacts.
        harmonic_regularity: Standard deviation of harmonic spacing
            across frames. Unusually low values indicate synthetic
            harmonic structure.
        is_suspicious: True if any metric exceeds its threshold.
    """

    spectral_smoothness: float
    temporal_discontinuity: float
    harmonic_regularity: float
    is_suspicious: bool


class DiffusionArtifactAnalyzer:
    """Detects artifacts specific to diffusion-based TTS systems.

    Uses three complementary signal-processing heuristics:
    1. Spectral smoothing — diffusion denoising over-smooths spectrograms
    2. Temporal boundary — chunk boundaries produce spectral flux spikes
    3. Harmonic regularity — denoising converges to unnaturally perfect harmonics
    """

    def __init__(
        self,
        smoothness_threshold: float = 0.15,
        discontinuity_threshold: float = 3.0,
        regularity_threshold: float = 0.05,
    ) -> None:
        self._smoothness_threshold = smoothness_threshold
        self._discontinuity_threshold = discontinuity_threshold
        self._regularity_threshold = regularity_threshold

    def analyze(
        self, audio: NDArray[np.floating], sr: int
    ) -> DiffusionAnalysis:
        """Analyze audio for diffusion-specific synthesis artifacts.

        Args:
            audio: Mono waveform, any float dtype.
            sr: Sample rate in Hz.

        Returns:
            DiffusionAnalysis with metric values and suspicion flag.
        """
        audio_f32 = audio.astype(np.float32)
        mel = extract_mel_spectrogram(audio_f32, sr)

        smoothness = self._spectral_smoothness(mel)
        discontinuity = self._temporal_discontinuity(mel)
        regularity = self._harmonic_regularity(audio_f32, sr)

        is_suspicious = (
            smoothness < self._smoothness_threshold
            or discontinuity > self._discontinuity_threshold
            or regularity < self._regularity_threshold
        )

        return DiffusionAnalysis(
            spectral_smoothness=smoothness,
            temporal_discontinuity=discontinuity,
            harmonic_regularity=regularity,
            is_suspicious=is_suspicious,
        )

    def _spectral_smoothness(self, mel: NDArray[np.float32]) -> float:
        """Measure high-frequency energy ratio in mel spectrogram.

        Diffusion models tend to over-smooth high frequencies, yielding
        a lower ratio of energy in the upper mel bands.
        """
        n_mels = mel.shape[0]
        if n_mels < 4:
            return 1.0

        # Split into lower 75% and upper 25% of mel bands
        split = int(n_mels * 0.75)
        # Convert from dB back to power for energy comparison
        power = 10.0 ** (mel / 10.0)
        low_energy = float(np.mean(power[:split]))
        high_energy = float(np.mean(power[split:]))
        total = low_energy + high_energy

        if total < 1e-12:
            return 1.0

        return high_energy / total

    def _temporal_discontinuity(self, mel: NDArray[np.float32]) -> float:
        """Detect spectral flux spikes indicating chunk boundaries.

        Computes frame-to-frame spectral flux and returns the maximum
        normalized by the median flux. Chunk boundaries produce large
        spikes relative to the smooth within-chunk transitions.
        """
        if mel.shape[1] < 3:
            return 0.0

        # Spectral flux: L2 norm of frame-to-frame differences
        diff = np.diff(mel, axis=1)
        flux = np.sqrt(np.sum(diff ** 2, axis=0))

        median_flux = float(np.median(flux))
        if median_flux < 1e-9:
            return 0.0

        max_flux = float(np.max(flux))
        return max_flux / median_flux

    def _harmonic_regularity(
        self, audio: NDArray[np.float32], sr: int
    ) -> float:
        """Measure variance of harmonic spacing across short frames.

        Natural speech has variable harmonic spacing due to pitch
        fluctuations. Diffusion synthesis produces unnaturally stable
        harmonic patterns.
        """
        frame_size = int(0.03 * sr)  # 30ms frames
        hop = frame_size // 2
        n_frames = max(1, (len(audio) - frame_size) // hop)

        if n_frames < 4:
            return 1.0

        peak_spacings: list[float] = []
        for i in range(n_frames):
            start = i * hop
            frame = audio[start : start + frame_size]
            spectrum = np.abs(np.fft.rfft(frame * np.hanning(len(frame))))

            # Find peaks in spectrum
            peaks = _find_spectral_peaks(spectrum)
            if len(peaks) >= 2:
                spacings = np.diff(peaks).astype(np.float64)
                peak_spacings.append(float(np.std(spacings)))

        if not peak_spacings:
            return 1.0

        # Return mean of per-frame spacing variability
        return float(np.mean(peak_spacings))


def _find_spectral_peaks(
    spectrum: NDArray[np.floating],
    min_prominence: float = 0.1,
) -> NDArray[np.intp]:
    """Find peaks in a magnitude spectrum using simple local maxima."""
    if len(spectrum) < 3:
        return np.array([], dtype=np.intp)

    threshold = float(np.max(spectrum)) * min_prominence
    is_peak = (
        (spectrum[1:-1] > spectrum[:-2])
        & (spectrum[1:-1] > spectrum[2:])
        & (spectrum[1:-1] > threshold)
    )
    return np.where(is_peak)[0] + 1
