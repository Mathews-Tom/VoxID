from __future__ import annotations

from voxid.schemas import (
    GeneratedScene,
    GenerationResult,
    SceneManifest,
    SceneNarration,
)


def test_scene_narration_minimal() -> None:
    # Arrange / Act
    narration = SceneNarration(scene_id="s1", text="Hello world")

    # Assert
    assert narration.scene_id == "s1"
    assert narration.text == "Hello world"
    assert narration.style is None
    assert narration.duration_hint is None
    assert narration.language is None


def test_scene_manifest_validates() -> None:
    # Arrange / Act
    manifest = SceneManifest(
        identity_id="tom",
        engine="qwen3-tts",
        scenes=[
            SceneNarration(scene_id="s1", text="First scene"),
            SceneNarration(scene_id="s2", text="Second scene", style="technical"),
        ],
        metadata={"project": "demo"},
    )

    # Assert
    assert manifest.identity_id == "tom"
    assert manifest.engine == "qwen3-tts"
    assert len(manifest.scenes) == 2
    assert manifest.metadata["project"] == "demo"


def test_scene_manifest_empty_scenes_validates() -> None:
    # Arrange / Act
    manifest = SceneManifest(identity_id="tom", scenes=[])

    # Assert
    assert manifest.scenes == []


def test_generated_scene_all_fields() -> None:
    # Arrange
    scene = GeneratedScene(
        scene_id="s1",
        audio_path="/tmp/s1.wav",
        duration_ms=3200,
        word_timings=[("Hello", 0, 500), ("world", 600, 1200)],
        style_used="conversational",
        engine_used="qwen3-tts",
    )

    # Act
    data = scene.model_dump()

    # Assert
    assert data["scene_id"] == "s1"
    assert data["audio_path"] == "/tmp/s1.wav"
    assert data["duration_ms"] == 3200
    assert data["word_timings"] == [("Hello", 0, 500), ("world", 600, 1200)]
    assert data["style_used"] == "conversational"
    assert data["engine_used"] == "qwen3-tts"


def test_generation_result_total_duration() -> None:
    # Arrange
    result = GenerationResult(
        manifest_id="m1",
        scenes=[
            GeneratedScene(
                scene_id="s1",
                audio_path="/tmp/s1.wav",
                duration_ms=1500,
                style_used="conversational",
                engine_used="qwen3-tts",
            )
        ],
        total_duration_ms=1500,
    )

    # Assert
    assert result.total_duration_ms == 1500
    assert result.manifest_id == "m1"
