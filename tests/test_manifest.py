from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import voxid.adapters.stub  # noqa: F401 — registers StubAdapter
from voxid.config import VoxIDConfig
from voxid.core import VoxID
from voxid.schemas import (
    GeneratedScene,
    GenerationResult,
    SceneManifest,
    SceneNarration,
)


@pytest.fixture
def vox(tmp_path: Path) -> VoxID:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    return VoxID(config=config)


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def identity_with_styles(vox: VoxID, ref_audio: Path) -> str:
    vox.create_identity(id="tom", name="Tom")
    for style_id in ["conversational", "technical", "narration", "emphatic"]:
        vox.add_style(
            identity_id="tom",
            id=style_id,
            label=style_id.title(),
            description=f"{style_id} style",
            ref_audio=ref_audio,
            ref_text=f"Reference text for {style_id}",
            engine="stub",
        )
    return "tom"


@pytest.fixture
def test_manifest() -> SceneManifest:
    return SceneManifest(
        identity_id="tom",
        scenes=[
            SceneNarration(
                scene_id="intro",
                text="Welcome to this breakdown of RAG.",
                style=None,  # auto-route
            ),
            SceneNarration(
                scene_id="technical",
                text="The pipeline processes queries in three stages.",
                style="technical",  # explicit
            ),
            SceneNarration(
                scene_id="closing",
                text="Honestly, once you see the numbers, there is no going back.",
                style=None,  # auto-route
            ),
        ],
    )


# ── generate_from_manifest tests ──────────────────────────────────────────────


def test_generate_from_manifest_produces_per_scene_wavs(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert — one WAV per scene
    for scene in result.scenes:
        assert Path(scene.audio_path).exists()
        assert Path(scene.audio_path).suffix == ".wav"


def test_generate_from_manifest_returns_generation_result(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert
    assert isinstance(result, GenerationResult)
    assert isinstance(result.manifest_id, str)
    assert isinstance(result.scenes, list)
    assert isinstance(result.total_duration_ms, int)
    for scene in result.scenes:
        assert isinstance(scene, GeneratedScene)


def test_generate_from_manifest_scene_ids_match(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert
    expected_ids = {s.scene_id for s in test_manifest.scenes}
    actual_ids = {s.scene_id for s in result.scenes}
    assert actual_ids == expected_ids


def test_generate_from_manifest_word_timings_present(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert
    for scene in result.scenes:
        assert len(scene.word_timings) > 0
        for word, start, end in scene.word_timings:
            assert isinstance(word, str)
            assert isinstance(start, int)
            assert isinstance(end, int)


def test_generate_from_manifest_duration_positive(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert
    for scene in result.scenes:
        assert scene.duration_ms > 0


def test_generate_from_manifest_stitched_output(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Arrange
    output_dir = tmp_path / "out"

    # Act
    vox.generate_from_manifest(test_manifest, output_dir=output_dir, stitch=True)

    # Assert — stitched file written by AudioStitcher
    stitched_path = output_dir / "stitched.wav"
    assert stitched_path.exists()


def test_generate_from_manifest_explicit_style_override(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert — the technical scene must use the explicit style
    technical_scene = next(s for s in result.scenes if s.scene_id == "technical")
    assert technical_scene.style_used == "technical"


def test_generate_from_manifest_auto_routes_when_style_none(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert — auto-routed scenes have a non-empty style_used
    auto_scene_ids = {
        s.scene_id
        for s in test_manifest.scenes
        if s.style is None
    }
    for scene in result.scenes:
        if scene.scene_id in auto_scene_ids:
            assert scene.style_used != ""


# ── plan_from_manifest tests ──────────────────────────────────────────────────


def test_plan_from_manifest_no_audio(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
) -> None:
    # Act
    result = vox.plan_from_manifest(test_manifest)

    # Assert — dry-run produces no audio files and zero durations
    for scene in result.scenes:
        assert scene.audio_path == ""
        assert scene.duration_ms == 0


def test_plan_from_manifest_routes_styles(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
) -> None:
    # Act
    result = vox.plan_from_manifest(test_manifest)

    # Assert — every scene has a non-empty style_used
    assert len(result.scenes) == 3
    for scene in result.scenes:
        assert scene.style_used != ""


def test_manifest_total_duration(
    identity_with_styles: str,
    vox: VoxID,
    test_manifest: SceneManifest,
    tmp_path: Path,
) -> None:
    # Act
    result = vox.generate_from_manifest(
        test_manifest, output_dir=tmp_path / "out", stitch=False
    )

    # Assert
    expected_total = sum(s.duration_ms for s in result.scenes)
    assert result.total_duration_ms == expected_total
