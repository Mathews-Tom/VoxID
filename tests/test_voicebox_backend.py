from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import voxid.adapters.stub  # noqa: F401
from voxid.config import VoxIDConfig
from voxid.core import VoxID
from voxid.plugins.voicebox.backend import TTSBackend, VoxIDBackend
from voxid.plugins.voicebox.models import (
    VoiceBoxGenerateRequest,
    VoiceBoxStory,
    VoiceBoxTrack,
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
def seeded_vox(vox: VoxID, ref_audio: Path) -> VoxID:
    vox.create_identity(id="tom", name="Tom")
    for sid in ["conversational", "technical", "narration", "emphatic"]:
        vox.add_style(
            identity_id="tom",
            id=sid,
            label=sid.title(),
            description=f"{sid} style",
            ref_audio=ref_audio,
            ref_text=f"Ref for {sid}",
            engine="stub",
        )
    return vox


@pytest.fixture
def backend(seeded_vox: VoxID) -> VoxIDBackend:
    return VoxIDBackend(voxid=seeded_vox)


# ── Protocol / identity tests ─────────────────────────────────────────────────


def test_backend_engine_name(backend: VoxIDBackend) -> None:
    assert backend.engine_name == "voxid"


def test_backend_supported_languages(backend: VoxIDBackend) -> None:
    languages = backend.supported_languages

    assert "en" in languages
    assert "zh" in languages
    assert "ja" in languages


def test_backend_is_available(backend: VoxIDBackend) -> None:
    assert backend.is_available() is True


def test_backend_satisfies_protocol(backend: VoxIDBackend) -> None:
    assert isinstance(backend, TTSBackend)


# ── list_voices ───────────────────────────────────────────────────────────────


def test_backend_list_voices(backend: VoxIDBackend) -> None:
    voices = backend.list_voices()

    assert "tom:conversational" in voices
    assert "tom:technical" in voices
    assert "tom:narration" in voices
    assert "tom:emphatic" in voices


def test_backend_list_voices_empty(vox: VoxID) -> None:
    empty_backend = VoxIDBackend(voxid=vox)

    assert empty_backend.list_voices() == []


# ── generate ─────────────────────────────────────────────────────────────────


def test_backend_generate_auto_route(backend: VoxIDBackend) -> None:
    request = VoiceBoxGenerateRequest(
        text="Hello, how are you?",
        profile_name="tom",
    )

    result = backend.generate(request)

    assert result.audio_path
    assert Path(result.audio_path).exists()
    assert result.sample_rate == 24000
    assert result.duration_seconds > 0


def test_backend_generate_explicit_style(backend: VoxIDBackend) -> None:
    request = VoiceBoxGenerateRequest(
        text="Let me explain the algorithm.",
        profile_name="tom:technical",
    )

    result = backend.generate(request)

    assert result.metadata["style"] == "technical"
    assert result.metadata["identity_id"] == "tom"


def test_backend_generate_with_output_path(
    backend: VoxIDBackend,
    tmp_path: Path,
) -> None:
    dest = str(tmp_path / "output" / "final.wav")
    request = VoiceBoxGenerateRequest(
        text="Copy me.",
        profile_name="tom:conversational",
        output_path=dest,
    )

    result = backend.generate(request)

    assert result.audio_path == dest
    assert Path(dest).exists()


# ── generate_story ────────────────────────────────────────────────────────────


def test_backend_generate_story_multi_track(backend: VoxIDBackend) -> None:
    story = VoiceBoxStory(
        story_id="s1",
        name="Three Tracks",
        tracks=[
            VoiceBoxTrack(
                track_id="t1",
                text="First sentence.",
                profile_name="tom:conversational",
            ),
            VoiceBoxTrack(
                track_id="t2",
                text="Second sentence.",
                profile_name="tom:technical",
            ),
            VoiceBoxTrack(
                track_id="t3",
                text="Third sentence.",
                profile_name="tom:narration",
            ),
        ],
    )

    results = backend.generate_story(story)

    assert len(results) == 3


def test_backend_generate_story_per_track_style(backend: VoxIDBackend) -> None:
    story = VoiceBoxStory(
        story_id="s2",
        name="Style Per Track",
        tracks=[
            VoiceBoxTrack(
                track_id="a",
                text="Technical content.",
                profile_name="tom:technical",
            ),
            VoiceBoxTrack(
                track_id="b",
                text="Narrated passage.",
                profile_name="tom:narration",
            ),
        ],
    )

    results = backend.generate_story(story)

    style_used_a = results[0].metadata["style_used"]
    style_used_b = results[1].metadata["style_used"]
    assert style_used_a == "technical"
    assert style_used_b == "narration"


def test_backend_generate_story_empty_tracks(backend: VoxIDBackend) -> None:
    story = VoiceBoxStory(story_id="empty", name="No Tracks")

    results = backend.generate_story(story)

    assert results == []


# ── _parse_profile ────────────────────────────────────────────────────────────


def test_backend_parse_profile_with_style() -> None:
    identity_id, style = VoxIDBackend._parse_profile("tom:technical")

    assert identity_id == "tom"
    assert style == "technical"


def test_backend_parse_profile_without_style() -> None:
    identity_id, style = VoxIDBackend._parse_profile("tom")

    assert identity_id == "tom"
    assert style is None
