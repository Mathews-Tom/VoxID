from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
import soundfile as sf

logger = logging.getLogger(__name__)

_EPSILON = 1e-9


@dataclass(frozen=True)
class RecordingMetrics:
    rms_dbfs: float
    peak_dbfs: float
    is_speech: bool
    elapsed_s: float


class AudioRecorder:
    """Callback-based audio recorder with real-time metrics.

    Uses sounddevice.InputStream for low-latency capture. Audio frames
    are buffered in a thread-safe deque; metrics are computed in the
    callback with minimal overhead.
    """

    def __init__(
        self,
        sample_rate: int = 48000,
        channels: int = 1,
        device: int | str | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._channels = channels
        self._device = device
        self._stream: sd.InputStream | None = None
        self._buffers: deque[np.ndarray] = deque()
        self._latest_metrics: RecordingMetrics | None = None
        self._start_time: float | None = None
        self._speech_threshold = 10.0 ** (-35.0 / 20.0)

    @property
    def sample_rate(self) -> int:
        return self._sample_rate

    @property
    def channels(self) -> int:
        return self._channels

    def _callback(
        self,
        indata: np.ndarray,
        frames: int,
        time_info: Any,
        status: sd.CallbackFlags,
    ) -> None:
        """Audio stream callback — runs on real-time thread.

        Minimal work: copy buffer, compute RMS/peak from the frame.
        """
        self._buffers.append(indata.copy())

        audio = indata[:, 0] if indata.ndim > 1 else indata.ravel()
        audio_f = audio.astype(np.float64)

        rms = float(np.sqrt(np.mean(audio_f ** 2)))
        peak = float(np.max(np.abs(audio_f)))
        rms_dbfs = float(20.0 * np.log10(rms + _EPSILON))
        peak_dbfs = float(20.0 * np.log10(peak + _EPSILON))
        is_speech = rms > self._speech_threshold

        elapsed = 0.0
        if self._start_time is not None:
            elapsed = time.monotonic() - self._start_time

        self._latest_metrics = RecordingMetrics(
            rms_dbfs=rms_dbfs,
            peak_dbfs=peak_dbfs,
            is_speech=is_speech,
            elapsed_s=elapsed,
        )

    def start(self) -> None:
        """Begin recording from the input device."""
        if self._stream is not None:
            raise RuntimeError("Recording already in progress")

        self._buffers.clear()
        self._latest_metrics = None
        self._start_time = time.monotonic()

        self._stream = sd.InputStream(
            samplerate=self._sample_rate,
            channels=self._channels,
            device=self._device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> np.ndarray:
        """Stop recording and return the captured audio as a numpy array."""
        if self._stream is None:
            raise RuntimeError("No recording in progress")

        self._stream.stop()
        self._stream.close()
        self._stream = None

        if not self._buffers:
            return np.zeros(0, dtype=np.float32)

        return np.concatenate(list(self._buffers))

    def is_recording(self) -> bool:
        return self._stream is not None

    def elapsed_seconds(self) -> float:
        if self._start_time is None:
            return 0.0
        return time.monotonic() - self._start_time

    def get_metrics(self) -> RecordingMetrics | None:
        return self._latest_metrics

    @staticmethod
    def list_devices() -> list[dict[str, Any]]:
        """List available audio devices."""
        devices = sd.query_devices()
        if isinstance(devices, dict):
            return [devices]
        result: list[dict[str, Any]] = [dict(d) for d in devices]
        return result

    @staticmethod
    def default_input_device() -> dict[str, Any]:
        """Return info about the default input device."""
        idx = sd.default.device[0]
        info = sd.query_devices(idx)
        result: dict[str, Any] = dict(info)
        return result


def detect_speech_energy(
    audio: np.ndarray,
    sr: int,
    frame_ms: int = 30,
    threshold_db: float = -35.0,
    merge_gap_ms: int = 300,
) -> list[tuple[int, int]]:
    """Detect speech segments using per-frame energy thresholding.

    Returns a list of (start_sample, end_sample) tuples for segments
    where RMS exceeds the threshold. Adjacent segments closer than
    merge_gap_ms are merged.
    """
    if len(audio) == 0:
        return []

    frame_size = max(1, int(sr * frame_ms / 1000))
    n_frames = len(audio) // frame_size
    if n_frames == 0:
        return []

    threshold_linear = 10.0 ** (threshold_db / 20.0)
    audio_f = audio.astype(np.float64)

    # Find speech frames
    speech_frames: list[bool] = []
    for i in range(n_frames):
        frame = audio_f[i * frame_size : (i + 1) * frame_size]
        rms = float(np.sqrt(np.mean(frame ** 2)))
        speech_frames.append(rms > threshold_linear)

    # Build segments from consecutive speech frames
    segments: list[tuple[int, int]] = []
    in_speech = False
    seg_start = 0

    for i, is_speech in enumerate(speech_frames):
        if is_speech and not in_speech:
            seg_start = i * frame_size
            in_speech = True
        elif not is_speech and in_speech:
            segments.append((seg_start, i * frame_size))
            in_speech = False

    if in_speech:
        segments.append((seg_start, n_frames * frame_size))

    # Merge segments closer than merge_gap_ms
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


def play_audio(audio: np.ndarray, sr: int) -> None:
    """Play audio through the default output device and block until done."""
    sd.play(audio, samplerate=sr)
    sd.wait()


def save_recording(
    audio: np.ndarray,
    sr: int,
    output_path: Path,
    fmt: str = "WAV",
    subtype: str = "PCM_16",
) -> Path:
    """Save audio to a file as 16-bit PCM WAV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), audio, sr, format=fmt, subtype=subtype)
    return output_path
