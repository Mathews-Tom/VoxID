from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from voxid.core import VoxID
from voxid.models import Identity

from .models import VoiceBoxProfile


class ProfileSync:
    """Bidirectional sync between VoiceBox profiles and VoxID identities."""

    def __init__(self, voxid: VoxID) -> None:
        self._voxid = voxid

    def import_from_voicebox(
        self,
        profile: VoiceBoxProfile,
        default_style: str = "conversational",
    ) -> Identity:
        """Import a VoiceBox profile as a VoxID identity.

        Creates the identity with one style per audio file.
        The first audio file becomes the default style.
        """
        identity = self._voxid.create_identity(
            id=self._slugify(profile.name),
            name=profile.name,
            description=profile.description,
            default_style=default_style,
            metadata={
                "source": "voicebox",
                "tags": profile.tags,
                **profile.metadata,
            },
        )

        for i, audio_path in enumerate(profile.audio_files):
            style_id = default_style if i == 0 else f"style-{i}"
            self._voxid.add_style(
                identity_id=identity.id,
                id=style_id,
                label=style_id.replace("-", " ").title(),
                description=f"Imported from VoiceBox: {profile.name}",
                ref_audio=audio_path,
                ref_text="",  # VoiceBox does not require transcripts
                language=profile.language,
            )

        return identity

    def export_to_voicebox(self, identity_id: str) -> VoiceBoxProfile:
        """Export a VoxID identity as a VoiceBox profile.

        Collects ref_audio paths from all registered styles.
        """
        store = self._voxid._store
        identity = store.get_identity(identity_id)
        style_ids = store.list_styles(identity_id)

        audio_files: list[str] = []
        tags: list[str] = []
        for style_id in style_ids:
            style = store.get_style(identity_id, style_id)
            audio_files.append(style.ref_audio_path)
            tags.append(style_id)

        return VoiceBoxProfile(
            name=identity.name,
            audio_files=audio_files,
            language="en",
            description=identity.description or "",
            tags=tags,
            metadata={
                "voxid_identity_id": identity_id,
                "default_style": identity.default_style,
            },
        )

    def export_to_json(
        self,
        identity_id: str,
        output_path: Path,
    ) -> Path:
        """Export a VoxID identity as a VoiceBox-format JSON file."""
        profile = self.export_to_voicebox(identity_id)
        data = {
            "name": profile.name,
            "audio_files": profile.audio_files,
            "language": profile.language,
            "description": profile.description,
            "tags": profile.tags,
            "metadata": profile.metadata,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return output_path

    def import_from_json(
        self,
        json_path: Path,
        default_style: str = "conversational",
    ) -> Identity:
        """Import a VoiceBox profile from a JSON file."""
        data: dict[str, Any] = json.loads(
            json_path.read_text(encoding="utf-8")
        )
        profile = VoiceBoxProfile(
            name=str(data["name"]),
            audio_files=list(data["audio_files"]),
            language=str(data.get("language", "en")),
            description=str(data.get("description", "")),
            tags=list(data.get("tags", [])),
            metadata=dict(data.get("metadata", {})),
        )
        return self.import_from_voicebox(profile, default_style)

    @staticmethod
    def _slugify(name: str) -> str:
        """Convert a display name to a URL-safe slug."""
        slug = name.lower().strip()
        slug = re.sub(r"[^\w\s-]", "", slug)
        slug = re.sub(r"[\s_]+", "-", slug)
        slug = re.sub(r"-+", "-", slug)
        return slug.strip("-")
