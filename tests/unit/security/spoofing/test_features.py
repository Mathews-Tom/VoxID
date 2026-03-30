from __future__ import annotations

import numpy as np
import pytest

from voxid.security.spoofing.config import SpoofingConfig
from voxid.security.spoofing.features import (
    _frame_signal,
    _linear_filterbank,
    extract_cqt,
    extract_lfcc,
    extract_mel_spectrogram,
    resample_if_needed,
)


@pytest.fixture
def sine_wave_16k() -> tuple[np.ndarray, int]:
    """440 Hz sine wave at 16 kHz, 2 seconds."""
    sr = 16000
    t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
    audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    return audio, sr


@pytest.fixture
def short_audio_16k() -> tuple[np.ndarray, int]:
    """Very short audio (256 samples) at 16 kHz."""
    sr = 16000
    audio = np.random.default_rng(42).standard_normal(256).astype(np.float32)
    return audio, sr


class TestExtractMelSpectrogram:
    def test_output_shape_default(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        mel = extract_mel_spectrogram(audio, sr)
        assert mel.shape[0] == 80
        assert mel.shape[1] > 0
        assert mel.dtype == np.float32

    def test_custom_n_mels(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        mel = extract_mel_spectrogram(audio, sr, n_mels=40)
        assert mel.shape[0] == 40

    def test_values_finite(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        mel = extract_mel_spectrogram(audio, sr)
        assert np.all(np.isfinite(mel))


class TestExtractCqt:
    def test_output_shape_default(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        cqt = extract_cqt(audio, sr)
        assert cqt.shape[0] == 84
        assert cqt.shape[1] > 0
        assert cqt.dtype == np.float32

    def test_custom_n_bins(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        cqt = extract_cqt(audio, sr, n_bins=48)
        assert cqt.shape[0] == 48


class TestExtractLfcc:
    def test_output_shape_default(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        lfcc = extract_lfcc(audio, sr)
        assert lfcc.shape[0] == 60
        assert lfcc.shape[1] > 0
        assert lfcc.dtype == np.float32

    def test_custom_n_lfcc(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        lfcc = extract_lfcc(audio, sr, n_lfcc=30)
        assert lfcc.shape[0] == 30

    def test_values_finite(
        self, sine_wave_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = sine_wave_16k
        lfcc = extract_lfcc(audio, sr)
        assert np.all(np.isfinite(lfcc))

    def test_short_audio_pads(
        self, short_audio_16k: tuple[np.ndarray, int]
    ) -> None:
        audio, sr = short_audio_16k
        lfcc = extract_lfcc(audio, sr)
        assert lfcc.shape[0] == 60
        assert lfcc.shape[1] >= 1


class TestFrameSignal:
    def test_normal_framing(self) -> None:
        signal = np.ones(1024, dtype=np.float32)
        frames = _frame_signal(signal, 256, 128)
        expected_n_frames = 1 + (1024 - 256) // 128
        assert frames.shape == (expected_n_frames, 256)

    def test_short_signal_pads(self) -> None:
        signal = np.ones(100, dtype=np.float32)
        frames = _frame_signal(signal, 256, 128)
        assert frames.shape == (1, 256)
        assert frames[0, 100] == 0.0


class TestLinearFilterbank:
    def test_shape(self) -> None:
        fb = _linear_filterbank(128, 513, 16000, 1024)
        assert fb.shape == (128, 513)
        assert fb.dtype == np.float32

    def test_non_negative(self) -> None:
        fb = _linear_filterbank(128, 513, 16000, 1024)
        assert np.all(fb >= 0)


class TestResampleIfNeeded:
    def test_no_resample_same_rate(self) -> None:
        config = SpoofingConfig(sample_rate=16000)
        audio = np.ones(16000, dtype=np.float32)
        result, sr_out = resample_if_needed(audio, 16000, config)
        assert sr_out == 16000
        np.testing.assert_array_equal(result, audio)

    def test_resample_different_rate(self) -> None:
        config = SpoofingConfig(sample_rate=16000)
        audio = np.ones(48000, dtype=np.float32)
        result, sr_out = resample_if_needed(audio, 48000, config)
        assert sr_out == 16000
        assert len(result) == 16000
