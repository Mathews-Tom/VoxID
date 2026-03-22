from __future__ import annotations

from voxid.video.ffmpeg import (
    check_ffmpeg,
    composite_video_audio,
    concat_audio_files,
)
from voxid.video.manim import (
    ManimSceneTiming,
    build_manim_config,
    build_manim_scenes,
    build_scene_timings,
)
from voxid.video.remotion import (
    RemotionScene,
    build_remotion_props,
    build_remotion_scenes,
    export_remotion_props,
)
from voxid.video.timing import (
    WordTiming,
    estimate_word_timings,
    timings_to_tuples,
)

__all__ = [
    # timing
    "WordTiming",
    "estimate_word_timings",
    "timings_to_tuples",
    # manim
    "ManimSceneTiming",
    "build_manim_config",
    "build_manim_scenes",
    "build_scene_timings",
    # remotion
    "RemotionScene",
    "build_remotion_props",
    "build_remotion_scenes",
    "export_remotion_props",
    # ffmpeg
    "check_ffmpeg",
    "composite_video_audio",
    "concat_audio_files",
]
