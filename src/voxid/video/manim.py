from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from voxid.schemas import GenerationResult


@dataclass(frozen=True)
class ManimSceneTiming:
    scene_id: str
    duration_seconds: float
    audio_path: str
    style: str


def build_scene_timings(
    result: GenerationResult,
) -> dict[str, float]:
    """Build scene_timings dict for Manim self.wait() calls.

    Returns {scene_id: duration_seconds} mapping.
    Used in Manim scenes as:
        self.wait(scene_timings["intro"])
    """
    return {
        scene.scene_id: scene.duration_ms / 1000.0
        for scene in result.scenes
    }


def build_manim_scenes(
    result: GenerationResult,
) -> list[ManimSceneTiming]:
    """Build detailed scene timing objects for Manim integration."""
    return [
        ManimSceneTiming(
            scene_id=scene.scene_id,
            duration_seconds=scene.duration_ms / 1000.0,
            audio_path=scene.audio_path,
            style=scene.style_used,
        )
        for scene in result.scenes
    ]


def build_manim_config(
    result: GenerationResult,
) -> dict[str, Any]:
    """Build a complete Manim configuration dict.

    Includes scene_timings, audio_paths, and style info.
    Suitable for passing as Manim scene kwargs or writing
    to a JSON config file.
    """
    return {
        "scene_timings": build_scene_timings(result),
        "audio_paths": {
            s.scene_id: s.audio_path
            for s in result.scenes
        },
        "styles": {
            s.scene_id: s.style_used
            for s in result.scenes
        },
        "total_duration_seconds": (
            result.total_duration_ms / 1000.0
        ),
    }
