from __future__ import annotations

import numpy as np
import pytest
import soundfile as sf

from voxid.segments.stitcher import AudioStitcher, StitchConfig


@pytest.fixture
def sine_segment():
    """Factory for sine wave audio segments."""

    def _make(freq: float = 440, duration: float = 0.5, sr: int = 24000):
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        return (0.3 * np.sin(2 * np.pi * freq * t)).astype(np.float32), sr

    return _make


def test_stitch_single_segment(sine_segment, tmp_path):
    # Arrange
    audio, sr = sine_segment()
    stitcher = AudioStitcher()
    output = tmp_path / "out.wav"

    # Act
    result_path, total_samples, result_sr = stitcher.stitch(
        audio_segments=[(audio, sr)],
        boundary_types=["paragraph"],
        output_path=output,
    )

    # Assert — single segment produces output equal to input length (no silence added)
    assert result_path == output
    assert result_sr == sr
    assert total_samples == len(audio)


def test_stitch_two_segments_adds_silence(sine_segment, tmp_path):
    # Arrange
    seg1, sr = sine_segment(freq=440, duration=0.5)
    seg2, _ = sine_segment(freq=880, duration=0.5)
    stitcher = AudioStitcher()
    output = tmp_path / "out.wav"

    # Act
    _, total_samples, result_sr = stitcher.stitch(
        audio_segments=[(seg1, sr), (seg2, sr)],
        boundary_types=["paragraph", "sentence"],
        output_path=output,
    )

    # Assert — total length > sum of individual segments (silence was inserted)
    naive_sum = len(seg1) + len(seg2)
    assert total_samples > naive_sum


def test_stitch_paragraph_pause_longer_than_sentence(sine_segment, tmp_path):
    # Arrange
    seg, sr = sine_segment(duration=0.5)
    stitcher = AudioStitcher()

    out_para = tmp_path / "para.wav"
    out_sent = tmp_path / "sent.wav"

    # Act — stitch identical audio with different boundary types on segment 2
    _, samples_para, _ = stitcher.stitch(
        audio_segments=[(seg, sr), (seg, sr)],
        boundary_types=["paragraph", "paragraph"],
        output_path=out_para,
    )
    _, samples_sent, _ = stitcher.stitch(
        audio_segments=[(seg, sr), (seg, sr)],
        boundary_types=["paragraph", "sentence"],
        output_path=out_sent,
    )

    # Assert
    assert samples_para > samples_sent


def test_stitch_output_is_valid_wav(sine_segment, tmp_path):
    # Arrange
    seg, sr = sine_segment()
    stitcher = AudioStitcher()
    output = tmp_path / "valid.wav"

    # Act
    stitcher.stitch(
        audio_segments=[(seg, sr)],
        boundary_types=["paragraph"],
        output_path=output,
    )

    # Assert — can be read back with soundfile
    data, read_sr = sf.read(str(output))
    assert read_sr == sr
    assert len(data) > 0


def test_stitch_mismatched_sample_rates_raises(sine_segment, tmp_path):
    # Arrange
    seg1, _ = sine_segment(sr=24000)
    seg2, _ = sine_segment(sr=16000)
    stitcher = AudioStitcher()
    output = tmp_path / "out.wav"

    # Act / Assert
    with pytest.raises(ValueError, match="sample rate"):
        stitcher.stitch(
            audio_segments=[(seg1, 24000), (seg2, 16000)],
            boundary_types=["paragraph", "sentence"],
            output_path=output,
        )


def test_stitch_crossfade_prevents_clicks(sine_segment, tmp_path):
    # Arrange — two segments, default crossfade_ms=20 at 24kHz = 480 samples
    seg1, sr = sine_segment(freq=440, duration=0.5)
    seg2, _ = sine_segment(freq=440, duration=0.5)
    config = StitchConfig(crossfade_ms=20, sentence_pause_ms=200)
    stitcher = AudioStitcher(config=config)
    output = tmp_path / "out.wav"

    # Act
    stitcher.stitch(
        audio_segments=[(seg1, sr), (seg2, sr)],
        boundary_types=["paragraph", "sentence"],
        output_path=output,
    )
    data, _ = sf.read(str(output), dtype="float32")

    # Assert — max absolute diff between adjacent samples in the full output < 0.1
    diffs = np.abs(np.diff(data))
    assert float(diffs.max()) < 0.1


def test_stitch_custom_pause_durations(sine_segment, tmp_path):
    # Arrange — custom config with known pause durations
    seg, sr = sine_segment(duration=0.5)
    config = StitchConfig(
        paragraph_pause_ms=1000,
        sentence_pause_ms=50,
        crossfade_ms=0,  # disable crossfade to simplify length math
    )
    stitcher = AudioStitcher(config=config)
    out_para = tmp_path / "para.wav"
    out_sent = tmp_path / "sent.wav"

    # Act
    _, samples_para, _ = stitcher.stitch(
        audio_segments=[(seg, sr), (seg, sr)],
        boundary_types=["paragraph", "paragraph"],
        output_path=out_para,
    )
    _, samples_sent, _ = stitcher.stitch(
        audio_segments=[(seg, sr), (seg, sr)],
        boundary_types=["paragraph", "sentence"],
        output_path=out_sent,
    )

    # Assert — para pause is 1000ms, sentence is 50ms; difference should reflect this
    expected_diff = int(sr * (1000 - 50) / 1000)
    actual_diff = samples_para - samples_sent
    assert actual_diff == pytest.approx(expected_diff, abs=10)


def test_stitch_empty_segments_raises(tmp_path):
    # Arrange
    stitcher = AudioStitcher()
    output = tmp_path / "out.wav"

    # Act / Assert
    with pytest.raises(ValueError):
        stitcher.stitch(
            audio_segments=[],
            boundary_types=[],
            output_path=output,
        )
