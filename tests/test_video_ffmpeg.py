from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

from voxid.video.ffmpeg import check_ffmpeg, composite_video_audio, concat_audio_files


def test_check_ffmpeg_returns_bool() -> None:
    # Act
    result = check_ffmpeg()

    # Assert — value depends on system; only type matters
    assert isinstance(result, bool)


def test_composite_missing_ffmpeg_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — make shutil.which return None so check_ffmpeg() returns False
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)

    video_path = tmp_path / "video.mp4"
    audio_path = tmp_path / "audio.wav"
    output_path = tmp_path / "out.mp4"

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="ffmpeg"):
        composite_video_audio(video_path, audio_path, output_path)


def test_composite_missing_video_raises(tmp_path: Path) -> None:
    # Arrange — only skip if ffmpeg is not installed
    if not check_ffmpeg():
        pytest.skip("ffmpeg not installed")

    video_path = tmp_path / "nonexistent_video.mp4"
    audio_path = tmp_path / "audio.wav"
    audio = np.zeros(24000, dtype=np.float32)
    sf.write(str(audio_path), audio, 24000)
    output_path = tmp_path / "out.mp4"

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Video file not found"):
        composite_video_audio(video_path, audio_path, output_path)


def test_composite_missing_audio_raises(tmp_path: Path) -> None:
    # Arrange — only skip if ffmpeg is not installed
    if not check_ffmpeg():
        pytest.skip("ffmpeg not installed")

    video_path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=black:size=2x2:duration=1:rate=30",
            "-c:v", "libx264", str(video_path),
        ],
        check=True,
        capture_output=True,
    )
    audio_path = tmp_path / "nonexistent_audio.wav"
    output_path = tmp_path / "out.mp4"

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="Audio file not found"):
        composite_video_audio(video_path, audio_path, output_path)


def test_concat_missing_ffmpeg_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Arrange — make shutil.which return None so check_ffmpeg() returns False
    import shutil

    monkeypatch.setattr(shutil, "which", lambda _: None)

    audio_paths = [tmp_path / "a.wav", tmp_path / "b.wav"]
    output_path = tmp_path / "out.wav"

    # Act / Assert
    with pytest.raises(FileNotFoundError, match="ffmpeg"):
        concat_audio_files(audio_paths, output_path)


@pytest.mark.skipif(not check_ffmpeg(), reason="ffmpeg not installed")
def test_composite_creates_output(tmp_path: Path) -> None:
    # Arrange — create minimal test video (1s, 2x2 px)
    video_path = tmp_path / "video.mp4"
    subprocess.run(
        [
            "ffmpeg", "-y", "-f", "lavfi",
            "-i", "color=black:size=2x2:duration=1:rate=30",
            "-c:v", "libx264", str(video_path),
        ],
        check=True,
        capture_output=True,
    )

    # Create a tiny silent WAV
    audio_path = tmp_path / "audio.wav"
    audio = np.zeros(24000, dtype=np.float32)
    sf.write(str(audio_path), audio, 24000)

    output_path = tmp_path / "composited.mp4"

    # Act
    result = composite_video_audio(video_path, audio_path, output_path, overwrite=True)

    # Assert
    assert result == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
