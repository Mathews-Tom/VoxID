from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from voxid.schemas import GeneratedScene, GenerationResult
from voxid.video.remotion import (
    build_remotion_props,
    build_remotion_scenes,
    export_remotion_props,
)


@pytest.fixture
def sample_result() -> GenerationResult:
    return GenerationResult(
        manifest_id="test-manifest",
        scenes=[
            GeneratedScene(
                scene_id="intro",
                audio_path="/tmp/intro.wav",
                duration_ms=3000,
                word_timings=[
                    ("Welcome", 0, 500),
                    ("to", 500, 700),
                    ("the", 700, 900),
                    ("demo", 900, 3000),
                ],
                style_used="narration",
                engine_used="stub",
            ),
            GeneratedScene(
                scene_id="technical",
                audio_path="/tmp/technical.wav",
                duration_ms=5000,
                word_timings=[
                    ("The", 0, 300),
                    ("pipeline", 300, 1200),
                    ("processes", 1200, 2500),
                    ("queries", 2500, 5000),
                ],
                style_used="technical",
                engine_used="stub",
            ),
            GeneratedScene(
                scene_id="closing",
                audio_path="/tmp/closing.wav",
                duration_ms=2000,
                word_timings=[
                    ("Thanks", 0, 800),
                    ("everyone", 800, 2000),
                ],
                style_used="conversational",
                engine_used="stub",
            ),
        ],
        total_duration_ms=10000,
    )


def test_build_remotion_scenes_frame_count(sample_result: GenerationResult) -> None:
    # Arrange — intro is 3.0s at default 30fps
    scenes = build_remotion_scenes(sample_result, fps=30)

    # Act
    intro = next(s for s in scenes if s.scene_id == "intro")

    # Assert — 3.0s * 30fps = 90 frames
    assert intro.duration_in_frames == 90


def test_build_remotion_scenes_word_timings_format(
    sample_result: GenerationResult,
) -> None:
    # Act
    scenes = build_remotion_scenes(sample_result)
    intro = next(s for s in scenes if s.scene_id == "intro")

    # Assert — each timing dict has the expected keys
    assert len(intro.word_timings) == 4
    for timing in intro.word_timings:
        assert "word" in timing
        assert "startMs" in timing
        assert "endMs" in timing
        assert isinstance(timing["word"], str)
        assert isinstance(timing["startMs"], int)
        assert isinstance(timing["endMs"], int)


def test_build_remotion_props_structure(sample_result: GenerationResult) -> None:
    # Act
    props = build_remotion_props(sample_result)

    # Assert
    assert "fps" in props
    assert "totalDurationInFrames" in props
    assert "scenes" in props
    assert isinstance(props["scenes"], list)


def test_build_remotion_props_total_frames(sample_result: GenerationResult) -> None:
    # Arrange
    fps = 30
    scenes = build_remotion_scenes(sample_result, fps=fps)
    expected_total = sum(s.duration_in_frames for s in scenes)

    # Act
    props = build_remotion_props(sample_result, fps=fps)

    # Assert
    assert props["totalDurationInFrames"] == expected_total


def test_build_remotion_props_custom_fps(sample_result: GenerationResult) -> None:
    # Act
    props_30 = build_remotion_props(sample_result, fps=30)
    props_60 = build_remotion_props(sample_result, fps=60)

    # Assert — double fps → double frames
    assert props_60["totalDurationInFrames"] == props_30["totalDurationInFrames"] * 2
    assert props_60["fps"] == 60


def test_export_remotion_props_creates_file(
    sample_result: GenerationResult, tmp_path: Path
) -> None:
    # Arrange
    output_path = tmp_path / "props.json"

    # Act
    returned_path = export_remotion_props(sample_result, output_path)

    # Assert
    assert returned_path == output_path
    assert output_path.exists()


def test_export_remotion_props_valid_json(
    sample_result: GenerationResult, tmp_path: Path
) -> None:
    # Arrange
    output_path = tmp_path / "props.json"
    export_remotion_props(sample_result, output_path)

    # Act
    content = output_path.read_text(encoding="utf-8")
    parsed = json.loads(content)

    # Assert
    assert "fps" in parsed
    assert "totalDurationInFrames" in parsed
    assert "scenes" in parsed
    assert isinstance(parsed["scenes"], list)
    assert len(parsed["scenes"]) == 3


def test_remotion_frame_accuracy(sample_result: GenerationResult) -> None:
    # Arrange — closing scene is 2000ms = 2.0s at 30fps
    fps = 30
    expected_frames = math.ceil(2.0 * fps)  # 60

    # Act
    scenes = build_remotion_scenes(sample_result, fps=fps)
    closing = next(s for s in scenes if s.scene_id == "closing")

    # Assert — within ±1 frame tolerance
    assert abs(closing.duration_in_frames - expected_frames) <= 1
