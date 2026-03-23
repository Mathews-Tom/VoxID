from __future__ import annotations

import numpy as np
import pyloudnorm as pyln

from voxid.enrollment.preprocessor import AudioPreprocessor


def _make_sine(
    sr: int,
    duration_s: float,
    freq: float = 440.0,
    amplitude: float = 0.5,
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * freq * t)).astype(np.float64)


class TestProperties:
    def test_target_sr_property(self) -> None:
        preprocessor = AudioPreprocessor(target_sr=48000)
        assert preprocessor.target_sr == 48000

    def test_target_lufs_property(self) -> None:
        preprocessor = AudioPreprocessor(target_lufs=-23.0)
        assert preprocessor.target_lufs == -23.0


class TestResample:
    def test_process_48khz_to_24khz_resamples(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=24000)
        audio = _make_sine(sr=48000, duration_s=1.0)

        # Act
        result, sr_out = preprocessor.process(audio, sr=48000)

        # Assert
        assert sr_out == 24000
        assert len(result) == 24000

    def test_resample_identity_when_same_rate(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=24000)
        audio = _make_sine(sr=24000, duration_s=0.5)

        # Act
        result = preprocessor.resample(audio, sr_in=24000, sr_out=24000)

        # Assert
        np.testing.assert_array_equal(result, audio)


class TestToMono:
    def test_process_stereo_to_mono(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=16000)
        left = _make_sine(sr=16000, duration_s=1.0, freq=440.0)
        right = _make_sine(sr=16000, duration_s=1.0, freq=880.0)
        stereo = np.stack([left, right], axis=1)

        # Act
        result, sr_out = preprocessor.process(stereo, sr=16000)

        # Assert
        assert result.ndim == 1
        assert sr_out == 16000

    def test_to_mono_already_mono_no_change(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor()
        audio = _make_sine(sr=24000, duration_s=0.5)

        # Act
        result = preprocessor.to_mono(audio)

        # Assert
        np.testing.assert_array_equal(result, audio)


class TestTrimSilence:
    def test_process_trims_leading_silence(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=16000, silence_pad_ms=50)
        sr = 16000
        silence = np.zeros(sr)  # 1 second of silence
        tone = _make_sine(sr=sr, duration_s=1.0, amplitude=0.5)
        audio = np.concatenate([silence, tone])
        leading_silence_samples = len(silence)

        # Act
        result = preprocessor.trim_silence(audio, sr)

        # Assert — result should be shorter than original due to trimmed leading silence
        assert len(result) < len(audio)
        # The start of the result should be near the tone onset (within pad + frame tolerance)
        pad_samples = int(sr * 50 / 1000)
        frame_size = max(1, int(sr * 10 / 1000))
        max_start_offset = pad_samples + frame_size
        assert len(result) >= len(tone) - frame_size
        assert len(result) <= len(tone) + max_start_offset

    def test_process_trims_trailing_silence(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=16000, silence_pad_ms=50)
        sr = 16000
        tone = _make_sine(sr=sr, duration_s=1.0, amplitude=0.5)
        silence = np.zeros(sr)  # 1 second of silence
        audio = np.concatenate([tone, silence])

        # Act
        result = preprocessor.trim_silence(audio, sr)

        # Assert — result should be shorter than original due to trimmed trailing silence
        assert len(result) < len(audio)
        pad_samples = int(sr * 50 / 1000)
        frame_size = max(1, int(sr * 10 / 1000))
        assert len(result) <= len(tone) + pad_samples + frame_size

    def test_process_preserves_speech_content(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=16000, silence_pad_ms=100)
        sr = 16000
        silence_lead = np.zeros(sr // 2)
        tone = _make_sine(sr=sr, duration_s=2.0, amplitude=0.5)
        silence_trail = np.zeros(sr // 2)
        audio = np.concatenate([silence_lead, tone, silence_trail])

        # Act
        result = preprocessor.trim_silence(audio, sr)

        # Assert — the tone portion must be fully contained in the result
        # Compute RMS of result to verify speech energy is preserved
        rms_result = float(np.sqrt(np.mean(result**2)))
        rms_tone = float(np.sqrt(np.mean(tone**2)))
        # RMS should be close (result includes some silence padding but is dominated by tone)
        assert rms_result > rms_tone * 0.5

    def test_trim_silence_all_silence_returns_minimal(self) -> None:
        # Arrange
        preprocessor = AudioPreprocessor(target_sr=16000)
        sr = 16000
        audio = np.zeros(sr)  # 1 second of pure silence

        # Act
        result = preprocessor.trim_silence(audio, sr)

        # Assert — returns exactly one frame of audio
        frame_size = max(1, int(sr * 10 / 1000))
        assert len(result) == frame_size


class TestNormalizeLoudness:
    def test_process_normalizes_loudness_to_target(self) -> None:
        # Arrange
        target_lufs = -16.0
        preprocessor = AudioPreprocessor(target_sr=24000, target_lufs=target_lufs)
        sr = 24000
        # Use a longer signal so pyloudnorm can measure accurately
        audio = _make_sine(sr=sr, duration_s=2.0, amplitude=0.1)

        # Act
        result = preprocessor.normalize_loudness(audio, sr)

        # Assert — measured loudness should be within 1 dB of target
        meter = pyln.Meter(sr)
        measured_lufs = meter.integrated_loudness(result)
        assert abs(measured_lufs - target_lufs) < 1.0

    def test_normalize_loudness_already_at_target_no_change(self) -> None:
        # Arrange
        target_lufs = -16.0
        preprocessor = AudioPreprocessor(target_sr=24000, target_lufs=target_lufs)
        sr = 24000
        audio = _make_sine(sr=sr, duration_s=2.0, amplitude=0.3)

        # Pre-normalize to target so the input is already at target LUFS
        meter = pyln.Meter(sr)
        loudness = meter.integrated_loudness(audio)
        audio = pyln.normalize.loudness(audio, loudness, target_lufs)

        # Act
        result = preprocessor.normalize_loudness(audio, sr)

        # Assert — output should be nearly identical to input
        max_diff = float(np.max(np.abs(result - audio)))
        assert max_diff < 1e-6


class TestFullPipeline:
    def test_process_full_pipeline_output_matches_spec(self) -> None:
        # Arrange
        target_sr = 24000
        preprocessor = AudioPreprocessor(target_sr=target_sr, target_lufs=-16.0)
        sr_in = 48000
        # Build mono audio with leading/trailing silence at 48kHz
        # Note: resample runs before to_mono in the pipeline, so np.interp
        # requires 1D input when sr differs from target.
        silence = np.zeros(sr_in // 2)
        tone = _make_sine(sr=sr_in, duration_s=2.0, amplitude=0.3)
        audio = np.concatenate([silence, tone, silence])

        # Act
        result, sr_out = preprocessor.process(audio, sr=sr_in)

        # Assert — mono
        assert result.ndim == 1
        # Assert — target sample rate
        assert sr_out == target_sr
        # Assert — non-silent (RMS well above silence threshold)
        rms = float(np.sqrt(np.mean(result**2)))
        assert rms > 1e-4
        # Assert — loudness near target
        meter = pyln.Meter(sr_out)
        measured = meter.integrated_loudness(result)
        assert abs(measured - (-16.0)) < 1.5


class TestEdgeCases:
    def test_resample_very_short_to_zero_samples(self) -> None:
        # Arrange — audio so short that target produces 0 samples
        preprocessor = AudioPreprocessor()
        audio = np.array([0.1], dtype=np.float64)

        # Act
        result = preprocessor.resample(audio, sr_in=48000, sr_out=1)

        # Assert
        assert len(result) == 0

    def test_trim_silence_empty_audio(self) -> None:
        preprocessor = AudioPreprocessor()
        audio = np.array([], dtype=np.float64)
        result = preprocessor.trim_silence(audio, 24000)
        assert len(result) == 0

    def test_normalize_loudness_empty_audio(self) -> None:
        preprocessor = AudioPreprocessor()
        audio = np.array([], dtype=np.float64)
        result = preprocessor.normalize_loudness(audio, 24000)
        assert len(result) == 0

    def test_normalize_loudness_very_short_audio(self) -> None:
        # Less than 0.4s — uses peak normalization fallback
        preprocessor = AudioPreprocessor(target_lufs=-16.0)
        sr = 24000
        audio = _make_sine(sr, 0.1, amplitude=0.5)

        result = preprocessor.normalize_loudness(audio, sr)
        assert len(result) == len(audio)
        assert float(np.max(np.abs(result))) > 0

    def test_normalize_loudness_silent_short_audio(self) -> None:
        # Very short all-zeros — peak < epsilon
        preprocessor = AudioPreprocessor()
        audio = np.zeros(100, dtype=np.float64)
        result = preprocessor.normalize_loudness(audio, 24000)
        np.testing.assert_array_equal(result, audio)

    def test_trim_silence_very_short_single_frame(self) -> None:
        # Audio shorter than one frame
        preprocessor = AudioPreprocessor()
        audio = np.array([0.5, 0.3, 0.1], dtype=np.float64)
        result = preprocessor.trim_silence(audio, 24000)
        assert len(result) == len(audio)
