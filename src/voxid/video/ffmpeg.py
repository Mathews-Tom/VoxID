from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def check_ffmpeg() -> bool:
    """Check if ffmpeg is available on PATH."""
    return shutil.which("ffmpeg") is not None


def composite_video_audio(
    video_path: Path,
    audio_path: Path,
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Composite video and audio into a single MP4.

    Uses ffmpeg: copies video stream, encodes audio as AAC.
    Raises FileNotFoundError if ffmpeg is not installed.
    Raises subprocess.CalledProcessError on ffmpeg failure.
    """
    if not check_ffmpeg():
        raise FileNotFoundError(
            "ffmpeg not found on PATH. Install with: "
            "brew install ffmpeg (macOS) or "
            "apt install ffmpeg (Linux)"
        )
    if not video_path.exists():
        msg = f"Video file not found: {video_path}"
        raise FileNotFoundError(msg)
    if not audio_path.exists():
        msg = f"Audio file not found: {audio_path}"
        raise FileNotFoundError(msg)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(audio_path),
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
    ]
    if overwrite:
        cmd.append("-y")
    cmd.append(str(output_path))

    subprocess.run(cmd, check=True, capture_output=True)
    return output_path


def concat_audio_files(
    audio_paths: list[Path],
    output_path: Path,
    overwrite: bool = False,
) -> Path:
    """Concatenate multiple audio files using ffmpeg.

    Creates a concat demuxer file and runs ffmpeg.
    Useful for stitching scene audio without re-encoding.
    """
    if not check_ffmpeg():
        raise FileNotFoundError(
            "ffmpeg not found on PATH."
        )
    for p in audio_paths:
        if not p.exists():
            msg = f"Audio file not found: {p}"
            raise FileNotFoundError(msg)

    # Write concat list
    list_path = output_path.parent / "concat_list.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as f:
        for p in audio_paths:
            # ffmpeg concat requires escaped single quotes
            safe = str(p.resolve()).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")

    cmd = [
        "ffmpeg",
        "-f", "concat",
        "-safe", "0",
        "-i", str(list_path),
        "-c", "copy",
    ]
    if overwrite:
        cmd.append("-y")
    cmd.append(str(output_path))

    try:
        subprocess.run(cmd, check=True, capture_output=True)
    finally:
        list_path.unlink(missing_ok=True)

    return output_path
