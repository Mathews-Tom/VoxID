from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import soundfile as sf

from voxid.enrollment.recorder import (
    AudioRecorder,
    RecordingMetrics,
    detect_speech_energy,
    save_recording,
)


def _make_sine(
    sr: int,
    duration_s: float,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float32)


class MockInputStream:
    """Simulates sounddevice.InputStream for testing."""

    def __init__(
        self,
        samplerate: int = 48000,
        channels: int = 1,
        device: int | str | None = None,
        callback: Any = None,
        **kwargs: Any,
    ) -> None:
        self._callback = callback
        self._samplerate = samplerate
        self._channels = channels

    def start(self) -> None:
        # Simulate a callback with synthetic audio
        if self._callback is not None:
            chunk = _make_sine(
                self._samplerate, 0.1, amplitude=0.3,
            ).reshape(-1, self._channels)
            flags = MagicMock()
            self._callback(chunk, len(chunk), None, flags)

    def stop(self) -> None:
        pass

    def close(self) -> None:
        pass


# --- RecordingMetrics ---


class TestRecordingMetrics:
    def test_dataclass_fields(self) -> None:
        m = RecordingMetrics(
            rms_dbfs=-20.0, peak_dbfs=-3.0,
            is_speech=True, elapsed_s=1.5,
        )
        assert m.rms_dbfs == -20.0
        assert m.peak_dbfs == -3.0
        assert m.is_speech is True
        assert m.elapsed_s == 1.5

    def test_frozen(self) -> None:
        m = RecordingMetrics(
            rms_dbfs=-20.0, peak_dbfs=-3.0,
            is_speech=True, elapsed_s=1.5,
        )
        with pytest.raises(AttributeError):
            m.rms_dbfs = -10.0  # type: ignore[misc]


# --- AudioRecorder ---


class TestAudioRecorder:
    def test_init_default_params(self) -> None:
        recorder = AudioRecorder()
        assert recorder.sample_rate == 48000
        assert recorder.channels == 1
        assert recorder.is_recording() is False

    def test_init_custom_params(self) -> None:
        recorder = AudioRecorder(
            sample_rate=24000, channels=2, device="test",
        )
        assert recorder.sample_rate == 24000
        assert recorder.channels == 2

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_start_stop_returns_ndarray(self) -> None:
        recorder = AudioRecorder()
        recorder.start()
        assert recorder.is_recording() is True
        audio = recorder.stop()
        assert isinstance(audio, np.ndarray)
        assert len(audio) > 0
        assert recorder.is_recording() is False

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_is_recording_true_after_start(self) -> None:
        recorder = AudioRecorder()
        recorder.start()
        assert recorder.is_recording() is True
        recorder.stop()

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_is_recording_false_after_stop(self) -> None:
        recorder = AudioRecorder()
        recorder.start()
        recorder.stop()
        assert recorder.is_recording() is False

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_elapsed_seconds_increases(self) -> None:
        recorder = AudioRecorder()
        assert recorder.elapsed_seconds() == 0.0
        recorder.start()
        elapsed = recorder.elapsed_seconds()
        assert elapsed >= 0.0
        recorder.stop()

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_get_metrics_returns_dataclass(self) -> None:
        recorder = AudioRecorder()
        recorder.start()
        metrics = recorder.get_metrics()
        recorder.stop()
        assert metrics is not None
        assert isinstance(metrics, RecordingMetrics)

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_metrics_rms_in_valid_range(self) -> None:
        recorder = AudioRecorder()
        recorder.start()
        metrics = recorder.get_metrics()
        recorder.stop()
        assert metrics is not None
        # Sine at 0.3 amplitude: RMS should be around -13 dBFS
        assert -30.0 < metrics.rms_dbfs < 0.0
        assert metrics.peak_dbfs <= 0.0

    @patch("voxid.enrollment.recorder.sd.InputStream", MockInputStream)
    def test_start_twice_raises(self) -> None:
        recorder = AudioRecorder()
        recorder.start()
        with pytest.raises(RuntimeError, match="already in progress"):
            recorder.start()
        recorder.stop()

    def test_stop_without_start_raises(self) -> None:
        recorder = AudioRecorder()
        with pytest.raises(RuntimeError, match="No recording"):
            recorder.stop()

    def test_get_metrics_none_before_start(self) -> None:
        recorder = AudioRecorder()
        assert recorder.get_metrics() is None

    @patch("voxid.enrollment.recorder.sd.query_devices")
    def test_list_devices_returns_list(
        self, mock_query: MagicMock,
    ) -> None:
        mock_query.return_value = [
            {"name": "Device 1", "max_input_channels": 2},
            {"name": "Device 2", "max_input_channels": 1},
        ]
        devices = AudioRecorder.list_devices()
        assert isinstance(devices, list)
        assert len(devices) == 2

    @patch("voxid.enrollment.recorder.sd.query_devices")
    def test_list_devices_single_device_dict(
        self, mock_query: MagicMock,
    ) -> None:
        mock_query.return_value = {
            "name": "Only Device", "max_input_channels": 1,
        }
        devices = AudioRecorder.list_devices()
        assert isinstance(devices, list)
        assert len(devices) == 1

    @patch("voxid.enrollment.recorder.sd.query_devices")
    @patch("voxid.enrollment.recorder.sd.default")
    def test_default_input_device(
        self,
        mock_default: MagicMock,
        mock_query: MagicMock,
    ) -> None:
        mock_default.device = (0, 1)
        mock_query.return_value = {
            "name": "Built-in Mic", "max_input_channels": 2,
        }
        info = AudioRecorder.default_input_device()
        assert info["name"] == "Built-in Mic"


# --- detect_speech_energy ---


class TestDetectSpeechEnergy:
    def test_finds_speech_in_sine_plus_silence(self) -> None:
        # Arrange — 1s silence + 2s sine + 1s silence
        sr = 24000
        silence1 = np.zeros(sr, dtype=np.float64)
        speech = _make_sine(sr, 2.0, amplitude=0.3).astype(np.float64)
        silence2 = np.zeros(sr, dtype=np.float64)
        audio = np.concatenate([silence1, speech, silence2])

        # Act
        segments = detect_speech_energy(audio, sr)

        # Assert — should find at least one segment
        assert len(segments) >= 1
        start, end = segments[0]
        # Speech starts around sample 24000, ends around 72000
        assert start < sr * 1.5  # within first half
        assert end > sr * 2.0  # extends past 2s mark

    def test_all_silence_returns_empty(self) -> None:
        audio = np.zeros(24000, dtype=np.float64)
        segments = detect_speech_energy(audio, 24000)
        assert segments == []

    def test_all_speech_returns_single_segment(self) -> None:
        audio = _make_sine(24000, 2.0, amplitude=0.3).astype(np.float64)
        segments = detect_speech_energy(audio, 24000)
        assert len(segments) == 1
        start, end = segments[0]
        assert start == 0
        # end is n_frames * frame_size (may exclude remainder samples)
        assert end >= len(audio) - 1000

    def test_merges_close_segments(self) -> None:
        # Arrange — speech + short gap (100ms) + speech
        sr = 24000
        speech1 = _make_sine(sr, 1.0, amplitude=0.3).astype(np.float64)
        gap = np.zeros(int(sr * 0.1), dtype=np.float64)  # 100ms < 300ms
        speech2 = _make_sine(sr, 1.0, amplitude=0.3).astype(np.float64)
        audio = np.concatenate([speech1, gap, speech2])

        # Act
        segments = detect_speech_energy(audio, sr, merge_gap_ms=300)

        # Assert — should merge into single segment
        assert len(segments) == 1

    def test_does_not_merge_distant_segments(self) -> None:
        # Arrange — speech + long gap (500ms) + speech
        sr = 24000
        speech1 = _make_sine(sr, 1.0, amplitude=0.3).astype(np.float64)
        gap = np.zeros(int(sr * 0.5), dtype=np.float64)  # 500ms > 300ms
        speech2 = _make_sine(sr, 1.0, amplitude=0.3).astype(np.float64)
        audio = np.concatenate([speech1, gap, speech2])

        # Act
        segments = detect_speech_energy(audio, sr, merge_gap_ms=300)

        # Assert — should be two separate segments
        assert len(segments) == 2

    def test_respects_threshold(self) -> None:
        # Very quiet signal below threshold
        sr = 24000
        quiet = _make_sine(sr, 1.0, amplitude=0.001).astype(np.float64)
        segments = detect_speech_energy(quiet, sr, threshold_db=-35.0)
        assert segments == []

    def test_empty_audio_returns_empty(self) -> None:
        audio = np.array([], dtype=np.float64)
        assert detect_speech_energy(audio, 24000) == []

    def test_very_short_audio(self) -> None:
        audio = np.array([0.5, 0.3], dtype=np.float64)
        segments = detect_speech_energy(audio, 24000)
        # Too short for even one frame
        assert segments == []


# --- save_recording ---


class TestSaveRecording:
    def test_creates_wav_file(self, tmp_path: Path) -> None:
        audio = _make_sine(24000, 1.0)
        path = save_recording(audio, 24000, tmp_path / "test.wav")
        assert path.exists()
        assert path.suffix == ".wav"

    def test_readable_by_soundfile(self, tmp_path: Path) -> None:
        audio = _make_sine(24000, 1.0)
        path = save_recording(audio, 24000, tmp_path / "test.wav")
        data, sr = sf.read(str(path))
        assert sr == 24000
        assert len(data) == len(audio)

    def test_correct_sample_rate(self, tmp_path: Path) -> None:
        audio = _make_sine(48000, 0.5)
        path = save_recording(audio, 48000, tmp_path / "test.wav")
        info = sf.info(str(path))
        assert info.samplerate == 48000

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        deep_path = tmp_path / "a" / "b" / "c" / "test.wav"
        audio = _make_sine(24000, 0.5)
        path = save_recording(audio, 24000, deep_path)
        assert path.exists()

    def test_pcm16_subtype(self, tmp_path: Path) -> None:
        audio = _make_sine(24000, 0.5)
        path = save_recording(audio, 24000, tmp_path / "test.wav")
        info = sf.info(str(path))
        assert info.subtype == "PCM_16"
