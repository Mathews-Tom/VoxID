from __future__ import annotations

from voxid.plugins.voicebox.models import (
    VoiceBoxGenerateRequest,
    VoiceBoxGenerateResult,
    VoiceBoxProfile,
    VoiceBoxStory,
    VoiceBoxTrack,
)


def test_voicebox_profile_defaults() -> None:
    profile = VoiceBoxProfile(name="Tom", audio_files=["/tmp/ref.wav"])

    assert profile.language == "en"
    assert profile.description == ""
    assert profile.tags == []
    assert profile.metadata == {}


def test_voicebox_generate_request_defaults() -> None:
    request = VoiceBoxGenerateRequest(text="Hello world", profile_name="tom")

    assert request.language == "en"
    assert request.output_path == ""
    assert request.options == {}


def test_voicebox_generate_result_fields() -> None:
    result = VoiceBoxGenerateResult(
        audio_path="/tmp/out.wav",
        sample_rate=24000,
        duration_seconds=1.5,
        metadata={"engine": "voxid"},
    )

    assert result.audio_path == "/tmp/out.wav"
    assert result.sample_rate == 24000
    assert result.duration_seconds == 1.5
    assert result.metadata["engine"] == "voxid"


def test_voicebox_track_style_optional() -> None:
    track = VoiceBoxTrack(
        track_id="t1",
        text="Narrate this.",
        profile_name="tom",
    )

    assert track.style == ""


def test_voicebox_story_empty_tracks() -> None:
    story = VoiceBoxStory(story_id="s1", name="Empty Story")

    assert story.tracks == []
    assert story.metadata == {}
