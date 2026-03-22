from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class VoiceBoxProfile:
    """VoiceBox voice profile — their internal representation."""

    name: str
    audio_files: list[str]  # paths to reference audio
    language: str = "en"
    description: str = ""
    tags: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceBoxGenerateRequest:
    """What VoiceBox sends to a TTSBackend."""

    text: str
    profile_name: str
    language: str = "en"
    output_path: str = ""
    options: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceBoxGenerateResult:
    """What VoiceBox expects back from a TTSBackend."""

    audio_path: str
    sample_rate: int
    duration_seconds: float
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class VoiceBoxTrack:
    """A track in VoiceBox's Stories editor."""

    track_id: str
    text: str
    profile_name: str
    style: str = ""  # VoxID extension: style per track
    language: str = "en"


@dataclass
class VoiceBoxStory:
    """A VoiceBox Story — multi-track composition."""

    story_id: str
    name: str
    tracks: list[VoiceBoxTrack] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
