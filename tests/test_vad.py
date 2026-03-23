from __future__ import annotations

from unittest.mock import patch

import numpy as np

from voxid.enrollment.vad import (
    VADBackend,
    detect_best_available,
    detect_speech,
    detect_speech_energy,
)


def _make_sine(
    sr: int, duration_s: float, amplitude: float = 0.3,
) -> np.ndarray:
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float64)


class TestVADBackend:
    def test_enum_values(self) -> None:
        assert VADBackend.ENERGY == "energy"
        assert VADBackend.SILERO == "silero"
        assert VADBackend.WEBRTC == "webrtc"


class TestDetectBestAvailable:
    @patch("voxid.enrollment.vad._silero_available", return_value=True)
    def test_prefers_silero(self, mock: object) -> None:
        assert detect_best_available() == VADBackend.SILERO

    @patch("voxid.enrollment.vad._silero_available", return_value=False)
    @patch("voxid.enrollment.vad._webrtc_available", return_value=True)
    def test_falls_back_to_webrtc(
        self, mock_webrtc: object, mock_silero: object,
    ) -> None:
        assert detect_best_available() == VADBackend.WEBRTC

    @patch("voxid.enrollment.vad._silero_available", return_value=False)
    @patch("voxid.enrollment.vad._webrtc_available", return_value=False)
    def test_falls_back_to_energy(
        self, mock_webrtc: object, mock_silero: object,
    ) -> None:
        assert detect_best_available() == VADBackend.ENERGY


class TestDetectSpeech:
    def test_energy_backend_finds_speech(self) -> None:
        sr = 24000
        silence = np.zeros(sr, dtype=np.float64)
        speech = _make_sine(sr, 2.0)
        audio = np.concatenate([silence, speech, silence])

        segments = detect_speech(audio, sr, backend=VADBackend.ENERGY)
        assert len(segments) >= 1

    def test_energy_backend_all_silence(self) -> None:
        audio = np.zeros(24000, dtype=np.float64)
        segments = detect_speech(audio, 24000, backend=VADBackend.ENERGY)
        assert segments == []

    def test_silero_fallback_to_energy_on_import_error(self) -> None:
        sr = 24000
        audio = _make_sine(sr, 2.0)

        with patch(
            "voxid.enrollment.vad.detect_speech_silero",
            side_effect=RuntimeError("No torch"),
        ):
            segments = detect_speech(
                audio, sr, backend=VADBackend.SILERO,
            )
            # Falls back to energy — should still find speech
            assert len(segments) >= 1

    def test_webrtc_fallback_to_energy_on_import_error(self) -> None:
        sr = 24000
        audio = _make_sine(sr, 2.0)

        with patch(
            "voxid.enrollment.vad.detect_speech_webrtc",
            side_effect=RuntimeError("No webrtcvad"),
        ):
            segments = detect_speech(
                audio, sr, backend=VADBackend.WEBRTC,
            )
            assert len(segments) >= 1

    def test_auto_backend_returns_segments(self) -> None:
        sr = 24000
        silence = np.zeros(sr, dtype=np.float64)
        speech = _make_sine(sr, 2.0)
        audio = np.concatenate([silence, speech])

        segments = detect_speech(audio, sr)
        assert len(segments) >= 1

    def test_detect_speech_energy_reexported(self) -> None:
        # Verify the energy function is accessible through vad module
        sr = 24000
        audio = _make_sine(sr, 1.0)
        segments = detect_speech_energy(audio, sr)
        assert isinstance(segments, list)
