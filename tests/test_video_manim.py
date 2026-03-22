from __future__ import annotations

import pytest

from voxid.schemas import GeneratedScene, GenerationResult
from voxid.video.manim import (
    ManimSceneTiming,
    build_manim_config,
    build_manim_scenes,
    build_scene_timings,
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


def test_build_scene_timings_returns_dict(sample_result: GenerationResult) -> None:
    # Act
    result = build_scene_timings(sample_result)

    # Assert
    assert isinstance(result, dict)
    assert all(isinstance(k, str) for k in result)
    assert all(isinstance(v, float) for v in result.values())


def test_build_scene_timings_correct_values(sample_result: GenerationResult) -> None:
    # Act
    result = build_scene_timings(sample_result)

    # Assert
    assert result["intro"] == pytest.approx(3.0)
    assert result["technical"] == pytest.approx(5.0)
    assert result["closing"] == pytest.approx(2.0)


def test_build_manim_scenes_returns_list(sample_result: GenerationResult) -> None:
    # Act
    result = build_manim_scenes(sample_result)

    # Assert
    assert isinstance(result, list)
    assert all(isinstance(s, ManimSceneTiming) for s in result)


def test_build_manim_scenes_correct_fields(sample_result: GenerationResult) -> None:
    # Act
    result = build_manim_scenes(sample_result)

    # Assert
    intro = next(s for s in result if s.scene_id == "intro")
    assert intro.duration_seconds == pytest.approx(3.0)
    assert intro.audio_path == "/tmp/intro.wav"
    assert intro.style == "narration"

    technical = next(s for s in result if s.scene_id == "technical")
    assert technical.duration_seconds == pytest.approx(5.0)
    assert technical.audio_path == "/tmp/technical.wav"
    assert technical.style == "technical"

    closing = next(s for s in result if s.scene_id == "closing")
    assert closing.duration_seconds == pytest.approx(2.0)
    assert closing.audio_path == "/tmp/closing.wav"
    assert closing.style == "conversational"


def test_build_manim_config_has_required_keys(sample_result: GenerationResult) -> None:
    # Act
    result = build_manim_config(sample_result)

    # Assert
    assert "scene_timings" in result
    assert "audio_paths" in result
    assert "styles" in result
    assert "total_duration_seconds" in result


def test_build_manim_config_total_duration(sample_result: GenerationResult) -> None:
    # Act
    result = build_manim_config(sample_result)

    # Assert
    assert result["total_duration_seconds"] == pytest.approx(10.0)
