from __future__ import annotations

import importlib.util
import logging
from enum import StrEnum

import numpy as np

from .recorder import detect_speech_energy

logger = logging.getLogger(__name__)


class VADBackend(StrEnum):
    ENERGY = "energy"
    SILERO = "silero"
    WEBRTC = "webrtc"


def _silero_available() -> bool:
    return (
        importlib.util.find_spec("torch") is not None
        and importlib.util.find_spec("torchaudio") is not None
    )


def _webrtc_available() -> bool:
    return importlib.util.find_spec("webrtcvad") is not None


def detect_best_available() -> VADBackend:
    """Return the best available VAD backend."""
    if _silero_available():
        return VADBackend.SILERO
    if _webrtc_available():
        return VADBackend.WEBRTC
    return VADBackend.ENERGY


def detect_speech_silero(
    audio: np.ndarray,
    sr: int,
    threshold: float = 0.5,
) -> list[tuple[int, int]]:
    """Detect speech using Silero VAD (requires torch).

    Model is loaded lazily on first call via torch.hub.
    """
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "Silero VAD requires PyTorch. Install with: uv add torch"
        ) from exc

    model, utils = torch.hub.load(
        repo_or_dir="snakers4/silero-vad",
        model="silero_vad",
        trust_repo=True,
    )
    get_speech_timestamps = utils[0]

    audio_tensor = torch.from_numpy(
        audio.astype(np.float32),
    )

    # Silero requires 16kHz
    if sr != 16000:
        audio_tensor = _resample_tensor(audio_tensor, sr, 16000)
        effective_sr = 16000
    else:
        effective_sr = sr

    timestamps = get_speech_timestamps(
        audio_tensor, model, threshold=threshold,
        sampling_rate=effective_sr,
    )

    # Convert back to original sample rate
    scale = sr / effective_sr
    return [
        (int(t["start"] * scale), int(t["end"] * scale))
        for t in timestamps
    ]


def _resample_tensor(
    tensor: object,
    sr_in: int,
    sr_out: int,
) -> object:
    """Resample a torch tensor using linear interpolation."""
    import torch

    assert isinstance(tensor, torch.Tensor)
    n_out = int(len(tensor) * sr_out / sr_in)
    indices = torch.linspace(0, len(tensor) - 1, n_out)
    return torch.from_numpy(
        np.interp(
            indices.numpy(),
            np.arange(len(tensor)),
            tensor.numpy(),
        ).astype(np.float32),
    )


def detect_speech_webrtc(
    audio: np.ndarray,
    sr: int,
    aggressiveness: int = 2,
    frame_ms: int = 30,
    merge_gap_ms: int = 300,
) -> list[tuple[int, int]]:
    """Detect speech using WebRTC VAD.

    Requires webrtcvad package. Audio is resampled to a supported
    rate (8000/16000/32000/48000 Hz) if needed.
    """
    try:
        import webrtcvad
    except ImportError as exc:
        raise RuntimeError(
            "WebRTC VAD requires webrtcvad. Install with: uv add webrtcvad"
        ) from exc

    supported_rates = {8000, 16000, 32000, 48000}
    if sr not in supported_rates:
        # Resample to 16kHz
        n_out = int(len(audio) * 16000 / sr)
        audio_resampled = np.interp(
            np.linspace(0, len(audio) - 1, n_out),
            np.arange(len(audio)),
            audio.astype(np.float64),
        ).astype(np.float32)
        effective_sr = 16000
        scale = sr / effective_sr
    else:
        audio_resampled = audio.astype(np.float32)
        effective_sr = sr
        scale = 1.0

    vad = webrtcvad.Vad(aggressiveness)
    frame_size = int(effective_sr * frame_ms / 1000)
    n_frames = len(audio_resampled) // frame_size

    # Convert to 16-bit PCM bytes
    pcm = (audio_resampled * 32767).astype(np.int16)

    speech_frames: list[bool] = []
    for i in range(n_frames):
        frame_bytes = pcm[
            i * frame_size : (i + 1) * frame_size
        ].tobytes()
        speech_frames.append(vad.is_speech(frame_bytes, effective_sr))

    # Build segments
    segments: list[tuple[int, int]] = []
    in_speech = False
    seg_start = 0

    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            seg_start = int(i * frame_size * scale)
            in_speech = True
        elif not is_speech and in_speech:
            segments.append((seg_start, int(i * frame_size * scale)))
            in_speech = False

    if in_speech:
        segments.append(
            (seg_start, int(n_frames * frame_size * scale)),
        )

    # Merge close segments
    if len(segments) <= 1:
        return segments

    merge_gap_samples = int(sr * merge_gap_ms / 1000)
    merged: list[tuple[int, int]] = [segments[0]]

    for start, end in segments[1:]:
        prev_start, prev_end = merged[-1]
        if start - prev_end <= merge_gap_samples:
            merged[-1] = (prev_start, end)
        else:
            merged.append((start, end))

    return merged


def detect_speech(
    audio: np.ndarray,
    sr: int,
    backend: VADBackend | None = None,
    **kwargs: object,
) -> list[tuple[int, int]]:
    """Detect speech using the specified or best available backend."""
    if backend is None:
        backend = detect_best_available()

    if backend == VADBackend.SILERO:
        try:
            return detect_speech_silero(audio, sr, **kwargs)  # type: ignore[arg-type]
        except (RuntimeError, ModuleNotFoundError):
            logger.warning("Silero VAD failed, falling back to energy")
            return detect_speech_energy(audio, sr)

    if backend == VADBackend.WEBRTC:
        try:
            return detect_speech_webrtc(audio, sr, **kwargs)  # type: ignore[arg-type]
        except (RuntimeError, ModuleNotFoundError):
            logger.warning("WebRTC VAD failed, falling back to energy")
            return detect_speech_energy(audio, sr)

    return detect_speech_energy(audio, sr)
