from __future__ import annotations

import hashlib
import hmac
import json
import tempfile
import zipfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import tomli
import tomli_w

from .models import ConsentRecord, Identity, Style
from .security.consent import check_export_consent, check_import_consent
from .store import VoicePromptStore


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class ArchiveExporter:
    """Export a VoxID identity to a portable .voxid archive."""

    def __init__(self, store: VoicePromptStore) -> None:
        self._store = store

    def export(
        self,
        identity_id: str,
        output_path: Path,
        signing_key: bytes | None = None,
        target_scope: str = "personal",
    ) -> Path:
        """Export identity to a .voxid ZIP archive.

        Includes identity.toml, consent.json, and all styles
        with ref_audio + ref_text. Engine-specific prompts
        are NOT included (prompt-as-cache principle).

        If signing_key is provided, computes HMAC-SHA256
        over all file hashes and stores in manifest.json.
        """
        identity = self._store.get_identity(identity_id)

        # Validate consent allows export
        consent_result = check_export_consent(
            identity,
            target_scope=target_scope,
        )
        if not consent_result.valid:
            raise ValueError(
                "Export blocked by consent policy: "
                + "; ".join(consent_result.errors)
            )

        styles = self._store.list_styles(identity_id)

        output_path.parent.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(
            output_path, "w", zipfile.ZIP_DEFLATED
        ) as zf:
            # Identity metadata
            identity_toml = tomli_w.dumps(identity.to_toml()).encode()
            zf.writestr("identity.toml", identity_toml)

            # Consent record
            consent_json = json.dumps(
                identity.consent_record.to_dict(), indent=2
            ).encode()
            zf.writestr("consent.json", consent_json)

            # Collect file hashes for manifest
            file_hashes: dict[str, str] = {
                "identity.toml": hashlib.sha256(identity_toml).hexdigest(),
                "consent.json": hashlib.sha256(consent_json).hexdigest(),
            }

            # Styles
            for style_id in styles:
                style = self._store.get_style(identity_id, style_id)
                style_prefix = f"styles/{style_id}"

                # style.toml
                style_toml = tomli_w.dumps(style.to_toml()).encode()
                zf.writestr(f"{style_prefix}/style.toml", style_toml)
                file_hashes[f"{style_prefix}/style.toml"] = (
                    hashlib.sha256(style_toml).hexdigest()
                )

                # ref_audio
                ref_audio = Path(style.ref_audio_path)
                if ref_audio.exists():
                    zf.write(
                        ref_audio,
                        f"{style_prefix}/ref_audio{ref_audio.suffix}",
                    )
                    file_hashes[
                        f"{style_prefix}/ref_audio{ref_audio.suffix}"
                    ] = _file_sha256(ref_audio)

                # ref_text
                ref_text = style.ref_text.encode("utf-8")
                zf.writestr(f"{style_prefix}/ref_text.txt", ref_text)
                file_hashes[f"{style_prefix}/ref_text.txt"] = (
                    hashlib.sha256(ref_text).hexdigest()
                )

                # unified.safetensors (engine-agnostic identity tokens)
                unified_path = self._store.get_unified_path(
                    identity_id, style_id,
                )
                if unified_path is not None and unified_path.exists():
                    zf.write(
                        unified_path,
                        f"{style_prefix}/unified.safetensors",
                    )
                    file_hashes[
                        f"{style_prefix}/unified.safetensors"
                    ] = _file_sha256(unified_path)

            # Manifest
            manifest: dict[str, Any] = {
                "version": "1.0",
                "identity_id": identity_id,
                "exported_at": datetime.now(UTC).isoformat(),
                "file_hashes": file_hashes,
            }

            if signing_key is not None:
                hash_payload = json.dumps(
                    file_hashes, sort_keys=True
                ).encode()
                manifest["hmac"] = hmac.new(
                    signing_key,
                    hash_payload,
                    hashlib.sha256,
                ).hexdigest()

            zf.writestr(
                "manifest.json",
                json.dumps(manifest, indent=2),
            )

        return output_path


class ArchiveImporter:
    """Import a .voxid archive into a VoicePromptStore."""

    def __init__(self, store: VoicePromptStore) -> None:
        self._store = store

    def import_archive(
        self,
        archive_path: Path,
        signing_key: bytes | None = None,
    ) -> Identity:
        """Import a .voxid archive.

        Verifies HMAC if signing_key provided.
        Creates identity and styles in the store.
        Engine prompts are NOT imported — they'll be
        rebuilt on first generation (prompt-as-cache).
        """
        if not archive_path.exists():
            msg = f"Archive not found: {archive_path}"
            raise FileNotFoundError(msg)

        with zipfile.ZipFile(archive_path, "r") as zf:
            # Read manifest
            manifest: dict[str, Any] = json.loads(zf.read("manifest.json"))

            # Verify HMAC if key provided
            if signing_key is not None:
                stored_hmac = manifest.get("hmac")
                if stored_hmac is None:
                    raise ValueError("Archive has no HMAC signature")
                hash_payload = json.dumps(
                    manifest["file_hashes"], sort_keys=True
                ).encode()
                expected = hmac.new(
                    signing_key,
                    hash_payload,
                    hashlib.sha256,
                ).hexdigest()
                if not hmac.compare_digest(stored_hmac, expected):
                    raise ValueError(
                        "HMAC verification failed — archive may be tampered"
                    )

            # Verify file hashes
            file_hashes: dict[str, str] = manifest["file_hashes"]
            for name, expected_hash in file_hashes.items():
                data = zf.read(name)
                actual = hashlib.sha256(data).hexdigest()
                if actual != expected_hash:
                    raise ValueError(
                        f"Hash mismatch for {name}: "
                        f"expected {expected_hash}, got {actual}"
                    )

            # Parse identity
            identity_data: dict[str, Any] = tomli.loads(
                zf.read("identity.toml").decode()
            )
            consent_data: dict[str, Any] = json.loads(zf.read("consent.json"))
            consent = ConsentRecord.from_dict(consent_data)
            identity = Identity.from_toml(identity_data, consent)

            # Advisory consent validation — does not block import
            check_import_consent(consent)

            # Create identity in store
            self._store.create_identity(identity)

            # Import styles
            style_dirs = {
                name.split("/")[1]
                for name in zf.namelist()
                if name.startswith("styles/") and "/" in name[7:]
            }

            for style_id in sorted(style_dirs):
                prefix = f"styles/{style_id}"
                style_toml = zf.read(f"{prefix}/style.toml").decode()
                style_data: dict[str, Any] = tomli.loads(style_toml)
                style = Style.from_toml(style_data)

                audio_names = [
                    n
                    for n in zf.namelist()
                    if n.startswith(f"{prefix}/ref_audio")
                ]
                if not audio_names:
                    msg = f"No ref_audio in archive for style {style_id}"
                    raise ValueError(msg)

                with tempfile.TemporaryDirectory() as td:
                    audio_name = audio_names[0]
                    ext = Path(audio_name).suffix
                    temp_audio = Path(td) / f"ref_audio{ext}"
                    temp_audio.write_bytes(zf.read(audio_name))

                    self._store.add_style(style, temp_audio)

                    # Restore unified.safetensors if present in archive
                    unified_name = f"{prefix}/unified.safetensors"
                    if unified_name in zf.namelist():
                        style_dir = (
                            self._store._style_dir(  # noqa: SLF001
                                identity.id, style_id,
                            )
                        )
                        unified_dest = style_dir / "unified.safetensors"
                        unified_dest.write_bytes(zf.read(unified_name))

        return identity
