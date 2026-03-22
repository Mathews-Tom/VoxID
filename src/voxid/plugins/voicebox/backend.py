from __future__ import annotations

import shutil
from pathlib import Path
from typing import Protocol, runtime_checkable

import soundfile as sf  # type: ignore[import-untyped]

from voxid.core import VoxID
from voxid.schemas import SceneManifest, SceneNarration

from .models import (
    VoiceBoxGenerateRequest,
    VoiceBoxGenerateResult,
    VoiceBoxStory,
)


@runtime_checkable
class TTSBackend(Protocol):
    """VoiceBox TTSBackend protocol (defined locally).

    Matches VoiceBox's expected interface for TTS engine integration.
    VoiceBox calls these methods when VoxID is registered as a backend.
    """

    @property
    def engine_name(self) -> str: ...

    @property
    def supported_languages(self) -> list[str]: ...

    def generate(
        self,
        request: VoiceBoxGenerateRequest,
    ) -> VoiceBoxGenerateResult: ...

    def list_voices(self) -> list[str]: ...

    def is_available(self) -> bool: ...


class VoxIDBackend:
    """VoxID adapter for VoiceBox's TTSBackend protocol.

    Maps VoiceBox concepts to VoxID concepts:
    - VoiceBox "profile" → VoxID "identity"
    - VoiceBox "voice"   → VoxID "identity:style"
    - VoiceBox "generate" → VoxID generate with routing
    """

    def __init__(self, voxid: VoxID | None = None) -> None:
        self._voxid = voxid or VoxID()

    @property
    def engine_name(self) -> str:
        return "voxid"

    @property
    def supported_languages(self) -> list[str]:
        return [
            "en", "zh", "ja", "ko", "de",
            "fr", "ru", "pt", "es", "it",
        ]

    def is_available(self) -> bool:
        """Check if VoxID is operational."""
        return True

    def list_voices(self) -> list[str]:
        """List available voices as identity:style pairs."""
        voices: list[str] = []
        for identity_id in self._voxid.list_identities():
            for style_id in self._voxid.list_styles(identity_id):
                voices.append(f"{identity_id}:{style_id}")
        return voices

    def generate(
        self,
        request: VoiceBoxGenerateRequest,
    ) -> VoiceBoxGenerateResult:
        """Generate audio from a VoiceBox request.

        Profile name format:
        - "identity_id"          → auto-route style
        - "identity_id:style_id" → explicit style
        """
        identity_id, style = self._parse_profile(request.profile_name)

        audio_path, sr = self._voxid.generate(
            text=request.text,
            identity_id=identity_id,
            style=style,
        )

        info = sf.info(str(audio_path))

        final_path: Path = audio_path
        if request.output_path:
            dest = Path(request.output_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(audio_path, dest)
            final_path = dest

        return VoiceBoxGenerateResult(
            audio_path=str(final_path),
            sample_rate=sr,
            duration_seconds=info.duration,
            metadata={
                "identity_id": identity_id,
                "style": style if style is not None else "auto-routed",
                "engine": "voxid",
            },
        )

    def generate_story(
        self,
        story: VoiceBoxStory,
        output_dir: Path | None = None,
    ) -> list[VoiceBoxGenerateResult]:
        """Generate audio for all tracks in a VoiceBox Story.

        Maps tracks to VoxID's manifest-based generation.
        Per-track style assignment is supported via the style field.
        Uses the first track's identity as the manifest identity
        (VoiceBox stories typically use one identity per story).
        """
        if not story.tracks:
            return []

        primary_identity, _ = self._parse_profile(story.tracks[0].profile_name)

        scenes: list[SceneNarration] = []
        for track in story.tracks:
            _identity_id, profile_style = self._parse_profile(track.profile_name)
            effective_style = track.style or profile_style
            scenes.append(
                SceneNarration(
                    scene_id=track.track_id,
                    text=track.text,
                    style=effective_style,
                    language=track.language,
                )
            )

        manifest = SceneManifest(
            identity_id=primary_identity,
            scenes=scenes,
            metadata={"story_id": story.story_id},
        )

        result = self._voxid.generate_from_manifest(
            manifest,
            output_dir=output_dir,
            stitch=False,
        )

        return [
            VoiceBoxGenerateResult(
                audio_path=scene.audio_path,
                sample_rate=24000,  # assumed from VoxID generation pipeline
                duration_seconds=scene.duration_ms / 1000.0,
                metadata={
                    "scene_id": scene.scene_id,
                    "style_used": scene.style_used,
                    "engine_used": scene.engine_used,
                },
            )
            for scene in result.scenes
        ]

    @staticmethod
    def _parse_profile(profile_name: str) -> tuple[str, str | None]:
        """Parse 'identity:style' or 'identity' format."""
        if ":" in profile_name:
            parts = profile_name.split(":", 1)
            return parts[0], parts[1]
        return profile_name, None
