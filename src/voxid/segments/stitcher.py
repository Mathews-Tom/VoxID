from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf  # type: ignore[import-untyped]


@dataclass(frozen=True)
class StitchConfig:
    paragraph_pause_ms: int = 500
    sentence_pause_ms: int = 200
    clause_pause_ms: int = 100
    crossfade_ms: int = 20  # small crossfade to prevent clicks


class AudioStitcher:
    """Concatenate audio segments with adaptive pauses."""

    def __init__(
        self,
        config: StitchConfig | None = None,
    ) -> None:
        self._config = config or StitchConfig()

    def stitch(
        self,
        audio_segments: list[tuple[np.ndarray, int]],
        boundary_types: list[str],
        output_path: Path,
    ) -> tuple[Path, int, int]:
        """Stitch audio segments into a single file.

        Args:
            audio_segments: list of (waveform, sample_rate) tuples
            boundary_types: boundary type for each segment
                (same length as audio_segments)
            output_path: path to write the stitched WAV

        Returns:
            (output_path, total_samples, sample_rate)
        """
        if not audio_segments:
            raise ValueError("audio_segments must not be empty")

        if len(audio_segments) != len(boundary_types):
            raise ValueError(
                f"audio_segments length {len(audio_segments)} != "
                f"boundary_types length {len(boundary_types)}"
            )

        # Validate uniform sample rate
        sample_rates = [sr for _, sr in audio_segments]
        if len(set(sample_rates)) > 1:
            raise ValueError(
                f"All segments must have the same sample rate; "
                f"found: {sorted(set(sample_rates))}"
            )
        sample_rate = sample_rates[0]

        crossfade_samples = int(
            sample_rate * self._config.crossfade_ms / 1000
        )

        # Start with the first segment
        accumulated = audio_segments[0][0].astype(np.float32)

        for i in range(1, len(audio_segments)):
            # Pause duration is determined by the boundary of the NEXT segment
            pause_ms = self._pause_for_boundary(boundary_types[i])
            silence = self._make_silence(pause_ms, sample_rate)

            # Crossfade between end of accumulated and silence
            accumulated = self._crossfade(accumulated, silence, crossfade_samples)

            # Crossfade between end of (accumulated+silence) and next segment
            next_audio = audio_segments[i][0].astype(np.float32)
            accumulated = self._crossfade(accumulated, next_audio, crossfade_samples)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(output_path), accumulated, sample_rate, subtype="PCM_16")

        total_samples = len(accumulated)
        return output_path, total_samples, sample_rate

    def _make_silence(
        self,
        duration_ms: int,
        sample_rate: int,
    ) -> np.ndarray:
        """Generate silence of given duration."""
        n_samples = int(sample_rate * duration_ms / 1000)
        return np.zeros(n_samples, dtype=np.float32)

    def _crossfade(
        self,
        a: np.ndarray,
        b: np.ndarray,
        crossfade_samples: int,
    ) -> np.ndarray:
        """Apply equal-power crossfade between two arrays."""
        if crossfade_samples <= 0 or len(a) == 0 or len(b) == 0:
            return np.concatenate([a, b])

        xf = min(crossfade_samples, len(a), len(b))

        # Equal-power crossfade curve
        t = np.linspace(0, np.pi / 2, xf, dtype=np.float32)
        fade_out = np.cos(t)
        fade_in = np.sin(t)

        result = np.empty(len(a) + len(b) - xf, dtype=np.float32)
        result[: len(a) - xf] = a[: len(a) - xf]
        result[len(a) - xf : len(a)] = (
            a[len(a) - xf :] * fade_out + b[:xf] * fade_in
        )
        result[len(a) :] = b[xf:]
        return result

    def _pause_for_boundary(self, boundary_type: str) -> int:
        """Return pause duration in ms for a boundary type."""
        if boundary_type == "paragraph":
            return self._config.paragraph_pause_ms
        if boundary_type == "sentence":
            return self._config.sentence_pause_ms
        return self._config.clause_pause_ms
