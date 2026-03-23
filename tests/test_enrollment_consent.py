from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxid.enrollment.consent import CONSENT_STATEMENT, ConsentManager


def _make_audio(sr: int = 24000, duration_s: float = 3.0) -> np.ndarray:
    """Generate a short sine wave for consent audio tests."""
    t = np.linspace(0, duration_s, int(sr * duration_s), endpoint=False)
    return (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)


class TestGenerateStatement:
    def test_includes_name(self) -> None:
        mgr = ConsentManager(Path("/tmp"))
        statement = mgr.generate_statement("Alice")
        assert "Alice" in statement

    def test_includes_scope(self) -> None:
        mgr = ConsentManager(Path("/tmp"))
        statement = mgr.generate_statement(
            "Bob", scope="commercial voice synthesis",
        )
        assert "commercial voice synthesis" in statement

    def test_default_scope(self) -> None:
        mgr = ConsentManager(Path("/tmp"))
        statement = mgr.generate_statement("Alice")
        assert "text-to-speech generation" in statement

    def test_matches_template(self) -> None:
        expected = CONSENT_STATEMENT.format(
            name="Alice", scope="text-to-speech generation",
        )
        mgr = ConsentManager(Path("/tmp"))
        assert mgr.generate_statement("Alice") == expected


class TestRecordConsent:
    @pytest.fixture
    def mgr(self, tmp_path: Path) -> ConsentManager:
        return ConsentManager(tmp_path)

    def test_saves_audio_file(self, mgr: ConsentManager, tmp_path: Path) -> None:
        # Arrange
        audio = _make_audio()

        # Act
        mgr.record_consent("alice", audio, 24000, scope="tts")

        # Assert
        audio_path = tmp_path / "identities" / "alice" / "consent_audio.wav"
        assert audio_path.exists()
        data, sr = sf.read(str(audio_path))
        assert sr == 24000
        assert len(data) == len(audio)

    def test_creates_consent_record(
        self, mgr: ConsentManager,
    ) -> None:
        audio = _make_audio()
        record = mgr.record_consent(
            "alice", audio, 24000, scope="tts", jurisdiction="EU",
        )

        assert record.scope == "tts"
        assert record.jurisdiction == "EU"
        assert record.transferable is False
        assert len(record.document_hash) == 64  # SHA-256 hex
        assert record.timestamp != ""

    def test_hash_matches_file(
        self, mgr: ConsentManager, tmp_path: Path,
    ) -> None:
        audio = _make_audio()
        record = mgr.record_consent("alice", audio, 24000, scope="tts")

        # Independently compute hash of the saved file
        audio_path = tmp_path / "identities" / "alice" / "consent_audio.wav"
        h = hashlib.sha256()
        with open(audio_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        expected_hash = h.hexdigest()

        assert record.document_hash == expected_hash

    def test_writes_consent_json(
        self, mgr: ConsentManager, tmp_path: Path,
    ) -> None:
        audio = _make_audio()
        mgr.record_consent("alice", audio, 24000, scope="tts")

        json_path = tmp_path / "identities" / "alice" / "consent.json"
        assert json_path.exists()


class TestVerifyConsentExists:
    @pytest.fixture
    def mgr(self, tmp_path: Path) -> ConsentManager:
        return ConsentManager(tmp_path)

    def test_true_after_recording(self, mgr: ConsentManager) -> None:
        audio = _make_audio()
        mgr.record_consent("alice", audio, 24000, scope="tts")
        assert mgr.verify_consent_exists("alice") is True

    def test_false_for_unknown(self, mgr: ConsentManager) -> None:
        assert mgr.verify_consent_exists("nonexistent") is False

    def test_false_when_only_audio_exists(
        self, mgr: ConsentManager, tmp_path: Path,
    ) -> None:
        # Create audio file but no JSON
        identity_dir = tmp_path / "identities" / "alice"
        identity_dir.mkdir(parents=True)
        sf.write(
            str(identity_dir / "consent_audio.wav"),
            _make_audio(), 24000,
        )
        assert mgr.verify_consent_exists("alice") is False


class TestLoadConsent:
    def test_load_after_record(self, tmp_path: Path) -> None:
        mgr = ConsentManager(tmp_path)
        audio = _make_audio()
        original = mgr.record_consent(
            "alice", audio, 24000, scope="tts",
        )
        loaded = mgr.load_consent("alice")
        assert loaded.scope == original.scope
        assert loaded.document_hash == original.document_hash
        assert loaded.timestamp == original.timestamp

    def test_load_nonexistent_raises(self, tmp_path: Path) -> None:
        mgr = ConsentManager(tmp_path)
        with pytest.raises(FileNotFoundError, match="No consent record"):
            mgr.load_consent("nonexistent")
