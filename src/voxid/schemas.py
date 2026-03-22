from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict


class SceneNarration(BaseModel):
    model_config = ConfigDict(strict=True)

    scene_id: str
    text: str
    style: str | None = None
    duration_hint: float | None = None
    language: str | None = None


class SceneManifest(BaseModel):
    model_config = ConfigDict(strict=True)

    identity_id: str
    engine: str | None = None
    scenes: list[SceneNarration]
    metadata: dict[str, Any] = {}


class GeneratedScene(BaseModel):
    model_config = ConfigDict(strict=True)

    scene_id: str
    audio_path: str
    duration_ms: int
    word_timings: list[tuple[str, int, int]] = []
    style_used: str
    engine_used: str


class GenerationResult(BaseModel):
    model_config = ConfigDict(strict=True)

    manifest_id: str
    scenes: list[GeneratedScene]
    total_duration_ms: int
