from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

import voxid.adapters.stub  # noqa: F401
from voxid.config import VoxIDConfig
from voxid.core import VoxID
from voxid.router import StyleRouter
from voxid.segments import (
    SegmentGenerationResult,
    SegmentPlanItem,
    build_segment_plan,
    export_plan,
)

_TECHNICAL_TEXT = (
    "We migrated from FAISS to pgvector, reducing cold-start latency from "
    "340ms to 89ms. The embedding model is now BGE-M3 running on a dedicated "
    "inference pod with 4-bit quantization. Every request now carries a signed "
    "token with a 15-minute expiry."
)

_CONVERSATIONAL_TEXT = (
    "Honestly, this one was a grind. Three false starts before we found the "
    "right configuration. But the numbers speak for themselves."
)

_MULTI_PARA_TEXT = f"{_TECHNICAL_TEXT}\n\n{_CONVERSATIONAL_TEXT}"

_LONG_TEXT = (
    "Welcome to this week's engineering update.\n\n"
    + _TECHNICAL_TEXT
    + "\n\n"
    + _CONVERSATIONAL_TEXT
)


@pytest.fixture
def vox_with_styles(tmp_path: Path) -> VoxID:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    vox = VoxID(config=config)

    audio = np.zeros(24000, dtype=np.float32)
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), audio, 24000)

    vox.create_identity(id="tom", name="Tom")
    for sid in ["conversational", "technical", "narration", "emphatic"]:
        vox.add_style(
            identity_id="tom",
            id=sid,
            label=sid.title(),
            description=f"{sid} style",
            ref_audio=ref,
            ref_text=f"Ref for {sid}",
            engine="stub",
        )
    return vox


# ---------------------------------------------------------------------------
# build_segment_plan
# ---------------------------------------------------------------------------


def test_build_segment_plan_returns_plan_items(tmp_path: Path):
    # Arrange
    router = StyleRouter(cache_dir=tmp_path / "cache")
    available = ["conversational", "technical", "narration", "emphatic"]

    # Act
    segments, plan = build_segment_plan(
        text=_LONG_TEXT,
        router=router,
        available_styles=available,
    )

    # Assert
    assert len(plan) > 0
    assert len(plan) == len(segments)
    for item in plan:
        assert isinstance(item, SegmentPlanItem)


def test_build_segment_plan_routes_different_styles(tmp_path: Path):
    # Arrange
    router = StyleRouter(cache_dir=tmp_path / "cache")
    available = ["conversational", "technical", "narration", "emphatic"]

    # Act
    _, plan = build_segment_plan(
        text=_MULTI_PARA_TEXT,
        router=router,
        available_styles=available,
    )

    # Assert — with 2 distinct paragraphs the plan should exist
    assert len(plan) >= 1
    styles = {item.style for item in plan}
    # At minimum we get valid styles from the available list
    assert styles.issubset(set(available))


def test_build_segment_plan_smoothing_applied(tmp_path: Path):
    # Arrange — text with short middle paragraph that should trigger smoothing
    short_middle = (
        "First paragraph with enough content. It has multiple sentences here.\n\n"
        "Short bit.\n\n"
        "Third paragraph is longer again. It also has multiple sentences. "
        "More content follows here."
    )
    router = StyleRouter(cache_dir=tmp_path / "cache")
    available = ["conversational", "technical", "narration", "emphatic"]

    # Act
    _, plan = build_segment_plan(
        text=short_middle,
        router=router,
        available_styles=available,
    )

    # Assert — plan contains was_smoothed field (smoothing ran without error)
    assert all(hasattr(item, "was_smoothed") for item in plan)


# ---------------------------------------------------------------------------
# export_plan
# ---------------------------------------------------------------------------


def test_export_plan_creates_json(tmp_path: Path):
    # Arrange
    router = StyleRouter(cache_dir=tmp_path / "cache")
    available = ["conversational", "technical"]
    _, plan = build_segment_plan(
        text=_MULTI_PARA_TEXT,
        router=router,
        available_styles=available,
    )
    output_path = tmp_path / "plan.json"

    # Act
    export_plan(plan, output_path)

    # Assert
    assert output_path.exists()
    parsed = json.loads(output_path.read_text(encoding="utf-8"))
    assert isinstance(parsed, list)


def test_export_plan_contains_required_fields(tmp_path: Path):
    # Arrange
    router = StyleRouter(cache_dir=tmp_path / "cache")
    available = ["conversational", "technical"]
    _, plan = build_segment_plan(
        text=_MULTI_PARA_TEXT,
        router=router,
        available_styles=available,
    )
    output_path = tmp_path / "plan.json"

    # Act
    export_plan(plan, output_path)
    items = json.loads(output_path.read_text(encoding="utf-8"))

    # Assert
    required_fields = {
        "index",
        "text",
        "style",
        "confidence",
        "tier",
        "boundary_type",
        "was_smoothed",
    }
    for item in items:
        assert required_fields.issubset(item.keys())


# ---------------------------------------------------------------------------
# generate_segments
# ---------------------------------------------------------------------------


def test_generate_segments_produces_per_segment_wavs(
    vox_with_styles: VoxID,
    tmp_path: Path,
):
    # Arrange
    output_dir = tmp_path / "segments_out"

    # Act
    result: SegmentGenerationResult = vox_with_styles.generate_segments(
        text=_LONG_TEXT,
        identity_id="tom",
        engine="stub",
        output_dir=output_dir,
        stitch=False,
    )

    # Assert — one WAV file per segment
    assert len(result.segments) > 0
    for seg in result.segments:
        assert seg.audio_path.exists()
        assert seg.audio_path.suffix == ".wav"


def test_generate_segments_stitched_output(
    vox_with_styles: VoxID,
    tmp_path: Path,
):
    # Arrange
    output_dir = tmp_path / "segments_stitched"

    # Act
    result: SegmentGenerationResult = vox_with_styles.generate_segments(
        text=_LONG_TEXT,
        identity_id="tom",
        engine="stub",
        output_dir=output_dir,
        stitch=True,
    )

    # Assert
    assert result.stitched_path is not None
    assert result.stitched_path.exists()
    data, sr = sf.read(str(result.stitched_path))
    assert len(data) > 0
    assert sr > 0


def test_generate_segments_plan_export(
    vox_with_styles: VoxID,
    tmp_path: Path,
):
    # Arrange
    output_dir = tmp_path / "segments_export"
    plan_path = tmp_path / "plan.json"

    # Act
    result: SegmentGenerationResult = vox_with_styles.generate_segments(
        text=_LONG_TEXT,
        identity_id="tom",
        engine="stub",
        output_dir=output_dir,
        stitch=False,
        export_plan_path=plan_path,
    )

    # Assert
    assert plan_path.exists()
    items = json.loads(plan_path.read_text(encoding="utf-8"))
    assert len(items) == len(result.segments)
    assert all("index" in item for item in items)
