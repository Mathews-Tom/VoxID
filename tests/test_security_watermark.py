from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxid.security.watermark import (
    WatermarkResult,
    detect_watermark,
    detect_watermark_file,
    embed_watermark,
    embed_watermark_file,
    is_audioseal_available,
)


@pytest.fixture
def test_audio() -> tuple[np.ndarray, int]:
    sr = 24000
    t = np.linspace(0, 1.0, sr, dtype=np.float32)
    return 0.3 * np.sin(2 * np.pi * 440 * t), sr


@pytest.fixture
def test_audio_file(tmp_path: Path, test_audio: tuple[np.ndarray, int]) -> Path:
    audio, sr = test_audio
    path = tmp_path / "input.wav"
    sf.write(str(path), audio, sr)
    return path


def test_is_audioseal_available_returns_bool() -> None:
    result = is_audioseal_available()
    assert isinstance(result, bool)


def test_embed_watermark_without_audioseal_returns_original(
    test_audio: tuple[np.ndarray, int],
) -> None:
    if is_audioseal_available():
        pytest.skip("AudioSeal is installed; degradation path not exercised")

    audio, sr = test_audio
    result_audio, result = embed_watermark(audio, sr, payload="test-payload")

    assert np.array_equal(result_audio, audio)
    assert result.method == "none"


def test_embed_watermark_result_fields(
    test_audio: tuple[np.ndarray, int],
) -> None:
    audio, sr = test_audio
    _, result = embed_watermark(audio, sr, payload="test-uuid")

    assert hasattr(result, "watermarked")
    assert hasattr(result, "payload")
    assert hasattr(result, "confidence")
    assert hasattr(result, "method")
    assert isinstance(result.watermarked, bool)
    assert isinstance(result.payload, str)
    assert isinstance(result.confidence, float)
    assert isinstance(result.method, str)


def test_detect_watermark_without_audioseal(
    test_audio: tuple[np.ndarray, int],
) -> None:
    if is_audioseal_available():
        pytest.skip("AudioSeal is installed; degradation path not exercised")

    audio, sr = test_audio
    result = detect_watermark(audio, sr)

    assert result.confidence == 0.0
    assert result.method == "none"


def test_embed_watermark_file_creates_output(
    tmp_path: Path,
    test_audio_file: Path,
) -> None:
    output_path = tmp_path / "output.wav"
    result = embed_watermark_file(test_audio_file, output_path, payload="uuid-001")

    assert output_path.exists()
    assert isinstance(result, WatermarkResult)


def test_detect_watermark_file_returns_result(
    test_audio_file: Path,
) -> None:
    result = detect_watermark_file(test_audio_file)
    assert isinstance(result, WatermarkResult)


def test_watermark_payload_preserved(
    test_audio: tuple[np.ndarray, int],
) -> None:
    audio, sr = test_audio
    _, result = embed_watermark(audio, sr, payload="test-uuid")
    assert result.payload == "test-uuid"
