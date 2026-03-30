from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SegmentHistory:
    """Prosodic features captured from a generated segment."""

    text: str
    style: str
    duration_ms: int
    final_f0: float  # Hz — trailing fundamental frequency
    final_energy: float  # RMS energy of trailing window
    speaking_rate: float  # words per second


@dataclass(frozen=True)
class GenerationContext:
    """Full context for the next segment generation."""

    history: list[SegmentHistory]
    doc_position: float  # 0.0 (start) to 1.0 (end)
    total_segments: int
    style_sequence: list[str]  # styles assigned to all segments


@dataclass(frozen=True)
class StitchParams:
    """Context-derived stitching parameters for one boundary."""

    pause_ms: int
    crossfade_ms: int = 20


@dataclass(frozen=True)
class ConditioningResult:
    """Output of the ContextConditioner for one segment."""

    ssml_prefix: str = ""  # text-level SSML hint prepended to segment
    ssml_suffix: str = ""  # text-level SSML hint appended to segment
    context_params: dict[str, float] = field(default_factory=dict)
    stitch: StitchParams = field(default_factory=lambda: StitchParams(pause_ms=200))
