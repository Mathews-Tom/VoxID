from __future__ import annotations

from .adapters import (
    EngineCapabilities,
    TTSEngineAdapter,
    get_adapter,
    list_adapters,
    register_adapter,
)
from .config import VoxIDConfig, load_config
from .models import ConsentRecord, Identity, Style
from .router import RouteDecision, StyleRouter
from .schemas import GeneratedScene, GenerationResult, SceneManifest, SceneNarration
from .segments import (
    SegmentGenerationResult,
    SegmentPlanItem,
    SegmentResult,
    build_segment_plan,
    export_plan,
)
from .store import VoicePromptStore
from .video import (
    WordTiming,
    build_manim_config,
    build_remotion_props,
    build_scene_timings,
    check_ffmpeg,
    composite_video_audio,
    estimate_word_timings,
    export_remotion_props,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    # models
    "ConsentRecord",
    "Identity",
    "Style",
    # schemas
    "SceneNarration",
    "SceneManifest",
    "GeneratedScene",
    "GenerationResult",
    # store
    "VoicePromptStore",
    # config
    "VoxIDConfig",
    "load_config",
    # adapters
    "EngineCapabilities",
    "TTSEngineAdapter",
    "register_adapter",
    "get_adapter",
    "list_adapters",
    # router
    "StyleRouter",
    "RouteDecision",
    # segments
    "SegmentGenerationResult",
    "SegmentPlanItem",
    "SegmentResult",
    "build_segment_plan",
    "export_plan",
    # video
    "WordTiming",
    "estimate_word_timings",
    "build_scene_timings",
    "build_manim_config",
    "build_remotion_props",
    "export_remotion_props",
    "check_ffmpeg",
    "composite_video_audio",
]
