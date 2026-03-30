from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voxid.router import RouteDecision, StyleRouter
from voxid.segments.segmenter import TextSegment, TextSegmenter
from voxid.segments.smoother import SmoothedDecision, StyleSmoother
from voxid.segments.stitcher import AudioStitcher, StitchConfig


@dataclass
class SegmentPlanItem:
    """One segment in a generation plan."""

    index: int
    text: str
    style: str
    confidence: float
    tier: str
    boundary_type: str
    was_smoothed: bool
    sentence_count: int


@dataclass
class SegmentResult:
    """Result of generating one segment."""

    index: int
    text: str
    style: str
    audio_path: Path
    duration_ms: int
    sample_rate: int
    boundary_type: str


@dataclass
class SegmentGenerationResult:
    """Full result of batch segment generation."""

    segments: list[SegmentResult]
    stitched_path: Path | None
    total_duration_ms: int
    plan: list[SegmentPlanItem]


def build_segment_plan(
    text: str,
    router: StyleRouter,
    available_styles: list[str],
    default_style: str = "conversational",
    segmenter: TextSegmenter | None = None,
    smoother: StyleSmoother | None = None,
) -> tuple[list[TextSegment], list[SegmentPlanItem]]:
    """Build a generation plan: segment text, route each, smooth transitions.

    Returns (segments, plan_items).
    """
    seg = segmenter or TextSegmenter()
    sm = smoother or StyleSmoother()

    segments = seg.segment(text)
    if not segments:
        return [], []

    # Route each segment (with context for semantic classifier)
    segment_texts = [s.text for s in segments]
    decisions: list[RouteDecision] = []
    for i, s in enumerate(segments):
        context_window = 2
        context_texts = [
            segment_texts[j]
            for j in range(
                max(0, i - context_window),
                min(len(segment_texts), i + context_window + 1),
            )
            if j != i
        ]
        decision = router.route(
            s.text,
            available_styles,
            default_style,
            context_texts=context_texts,
        )
        decisions.append(decision)

    # Smooth
    sentence_counts = [s.sentence_count for s in segments]
    smoothed = sm.smooth(decisions, sentence_counts)

    # Build plan
    plan: list[SegmentPlanItem] = []
    for i, (seg_item, sm_item) in enumerate(zip(segments, smoothed)):
        plan.append(
            SegmentPlanItem(
                index=i,
                text=seg_item.text,
                style=sm_item.style,
                confidence=sm_item.confidence,
                tier=decisions[i].tier,
                boundary_type=seg_item.boundary_type,
                was_smoothed=sm_item.was_smoothed,
                sentence_count=seg_item.sentence_count,
            )
        )

    return segments, plan


def export_plan(
    plan: list[SegmentPlanItem],
    output_path: Path,
) -> None:
    """Export generation plan to JSON."""
    data = []
    for item in plan:
        data.append(
            {
                "index": item.index,
                "text": item.text,
                "style": item.style,
                "confidence": item.confidence,
                "tier": item.tier,
                "boundary_type": item.boundary_type,
                "was_smoothed": item.was_smoothed,
                "sentence_count": item.sentence_count,
            }
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


__all__ = [
    "AudioStitcher",
    "SegmentGenerationResult",
    "SegmentPlanItem",
    "SegmentResult",
    "SmoothedDecision",
    "StitchConfig",
    "StyleSmoother",
    "TextSegment",
    "TextSegmenter",
    "build_segment_plan",
    "export_plan",
]
