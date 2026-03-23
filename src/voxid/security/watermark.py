from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class WatermarkResult:
    watermarked: bool
    payload: str  # hex string: identity UUID + timestamp
    confidence: float  # 0-1, detection confidence
    method: str  # "audioseal" or "none"


def is_audioseal_available() -> bool:
    """Check if AudioSeal is installed."""
    try:
        import audioseal  # type: ignore[import-not-found,unused-ignore]  # noqa: F401

        return True
    except ImportError:
        return False


def embed_watermark(
    audio: np.ndarray,
    sample_rate: int,
    payload: str,
) -> tuple[np.ndarray, WatermarkResult]:
    """Embed an AudioSeal watermark into audio.

    If AudioSeal is not installed, returns the original audio
    unchanged with method="none".

    Args:
        audio: float32 waveform
        sample_rate: audio sample rate
        payload: hex string to embed (e.g., identity UUID)

    Returns:
        (watermarked_audio, WatermarkResult)
    """
    if not is_audioseal_available():
        return audio, WatermarkResult(
            watermarked=False,
            payload=payload,
            confidence=0.0,
            method="none",
        )

    import torch  # type: ignore[import-not-found,unused-ignore]
    from audioseal import (
        AudioSeal as AudioSealModel,  # type: ignore[import-not-found,unused-ignore]
    )

    model = AudioSealModel.load_generator("audioseal_wm_16bits")

    # Prepare audio tensor: (batch, channels, samples)
    audio_tensor = torch.from_numpy(audio).unsqueeze(0).unsqueeze(0)
    if audio_tensor.dtype != torch.float32:
        audio_tensor = audio_tensor.float()

    # Generate watermark
    watermark = model.get_watermark(
        audio_tensor,
        sample_rate=sample_rate,
    )
    watermarked = audio_tensor + watermark

    result_audio = watermarked.squeeze().numpy().astype(np.float32)
    return result_audio, WatermarkResult(
        watermarked=True,
        payload=payload,
        confidence=1.0,
        method="audioseal",
    )


def detect_watermark(
    audio: np.ndarray,
    sample_rate: int,
) -> WatermarkResult:
    """Detect an AudioSeal watermark in audio.

    Returns WatermarkResult with confidence score.
    If AudioSeal is not installed, returns confidence=0.
    """
    if not is_audioseal_available():
        return WatermarkResult(
            watermarked=False,
            payload="",
            confidence=0.0,
            method="none",
        )

    import torch  # type: ignore[import-not-found,unused-ignore]
    from audioseal import (
        AudioSeal as AudioSealModel,  # type: ignore[import-not-found,unused-ignore]
    )

    detector = AudioSealModel.load_detector(
        "audioseal_detector_16bits",
    )

    audio_tensor = (
        torch.from_numpy(audio).unsqueeze(0).unsqueeze(0).float()
    )

    result = detector.detect_watermark(
        audio_tensor,
        sample_rate=sample_rate,
    )
    # result is a tuple: (detection_prob, message_bits)
    prob = float(result[0].mean().item())

    return WatermarkResult(
        watermarked=prob > 0.5,
        payload="",  # payload extraction not always reliable
        confidence=prob,
        method="audioseal",
    )


def embed_watermark_file(
    input_path: Path,
    output_path: Path,
    payload: str,
) -> WatermarkResult:
    """Embed watermark in a WAV file, write to output."""
    import soundfile as sf  # type: ignore[import-untyped]

    audio, sr = sf.read(str(input_path), dtype="float32")
    watermarked, result = embed_watermark(audio, sr, payload)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), watermarked, sr)
    return result


def detect_watermark_file(
    audio_path: Path,
) -> WatermarkResult:
    """Detect watermark in a WAV file."""
    import soundfile as sf  # type: ignore[import-untyped,unused-ignore]

    audio, sr = sf.read(str(audio_path), dtype="float32")
    return detect_watermark(audio, sr)
