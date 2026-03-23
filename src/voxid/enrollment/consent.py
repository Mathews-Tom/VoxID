from __future__ import annotations

import datetime
import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf

from ..models import ConsentRecord

logger = logging.getLogger(__name__)

CONSENT_STATEMENT = (
    "I, {name}, am aware that recordings of my voice will be used "
    "to train and create a synthetic version of my voice for use in {scope}."
)


class ConsentManager:
    """Manages consent recording, storage, and verification for enrollment."""

    def __init__(self, store_path: Path) -> None:
        self._store_path = store_path

    def _identity_dir(self, identity_id: str) -> Path:
        return self._store_path / "identities" / identity_id

    def _consent_audio_path(self, identity_id: str) -> Path:
        return self._identity_dir(identity_id) / "consent_audio.wav"

    def _consent_json_path(self, identity_id: str) -> Path:
        return self._identity_dir(identity_id) / "consent.json"

    def generate_statement(
        self,
        name: str,
        scope: str = "text-to-speech generation",
    ) -> str:
        """Generate a consent statement with the speaker's name and scope."""
        return CONSENT_STATEMENT.format(name=name, scope=scope)

    def record_consent(
        self,
        identity_id: str,
        audio: np.ndarray,
        sr: int,
        scope: str,
        jurisdiction: str = "US",
    ) -> ConsentRecord:
        """Save consent audio and create a ConsentRecord.

        1. Write consent audio as 16-bit PCM WAV
        2. Compute SHA-256 hash of the saved file
        3. Create and persist ConsentRecord
        """
        audio_path = self._consent_audio_path(identity_id)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(audio_path), audio, sr, format="WAV", subtype="PCM_16")

        file_hash = self._compute_file_hash(audio_path)

        record = ConsentRecord(
            timestamp=datetime.datetime.now(
                tz=datetime.UTC,
            ).isoformat(),
            scope=scope,
            jurisdiction=jurisdiction,
            transferable=False,
            document_hash=file_hash,
        )

        consent_json_path = self._consent_json_path(identity_id)
        consent_json_path.write_text(
            json.dumps(record.to_dict(), indent=2),
            encoding="utf-8",
        )

        return record

    def verify_consent_exists(self, identity_id: str) -> bool:
        """Check whether a valid consent record exists for the identity."""
        audio_path = self._consent_audio_path(identity_id)
        json_path = self._consent_json_path(identity_id)
        return audio_path.exists() and json_path.exists()

    def load_consent(self, identity_id: str) -> ConsentRecord:
        """Load an existing consent record."""
        json_path = self._consent_json_path(identity_id)
        if not json_path.exists():
            raise FileNotFoundError(
                f"No consent record for identity '{identity_id}'"
            )
        with open(json_path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        return ConsentRecord.from_dict(data)

    @staticmethod
    def _compute_file_hash(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
