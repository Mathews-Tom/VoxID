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
]
