from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from voxid.schemas import GenerationResult


@dataclass(frozen=True)
class RemotionScene:
    scene_id: str
    style: str
    audio_src: str
    duration_seconds: float
    duration_in_frames: int
    word_timings: list[dict[str, Any]]
    # word_timings format: [{"word": "hello", "startMs": 0, "endMs": 300}]


def build_remotion_scenes(
    result: GenerationResult,
    fps: int = 30,
) -> list[RemotionScene]:
    """Convert GenerationResult to Remotion-compatible scene list.

    Each scene maps to a Remotion <Sequence> component:
    - durationInFrames = ceil(duration_seconds * fps)
    - wordTimings formatted for subtitle rendering
    """
    scenes: list[RemotionScene] = []
    for gs in result.scenes:
        duration_s = gs.duration_ms / 1000.0
        frames = math.ceil(duration_s * fps)

        word_timing_dicts: list[dict[str, Any]] = [
            {
                "word": word,
                "startMs": start,
                "endMs": end,
            }
            for word, start, end in gs.word_timings
        ]

        scenes.append(RemotionScene(
            scene_id=gs.scene_id,
            style=gs.style_used,
            audio_src=gs.audio_path,
            duration_seconds=duration_s,
            duration_in_frames=frames,
            word_timings=word_timing_dicts,
        ))
    return scenes


def build_remotion_props(
    result: GenerationResult,
    fps: int = 30,
) -> dict[str, Any]:
    """Build complete Remotion composition props.

    Output format matches the TypeScript interface:
    {
        scenes: VoxIDScene[],
        fps: number,
        totalDurationInFrames: number,
    }
    """
    scenes = build_remotion_scenes(result, fps)
    total_frames = sum(s.duration_in_frames for s in scenes)

    return {
        "fps": fps,
        "totalDurationInFrames": total_frames,
        "scenes": [
            {
                "sceneId": s.scene_id,
                "style": s.style,
                "audioSrc": s.audio_src,
                "durationSeconds": s.duration_seconds,
                "durationInFrames": s.duration_in_frames,
                "wordTimings": s.word_timings,
            }
            for s in scenes
        ],
    }


def export_remotion_props(
    result: GenerationResult,
    output_path: Path,
    fps: int = 30,
) -> Path:
    """Export Remotion props to JSON file."""
    props = build_remotion_props(result, fps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(props, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_path
