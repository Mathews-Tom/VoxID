from __future__ import annotations

from pathlib import Path

import pytest

from voxid.models import ConsentRecord, Identity, Style
from voxid.store import VoicePromptStore


def test_create_identity_creates_directory_and_files(
    store: VoicePromptStore,
    sample_identity: Identity,
) -> None:
    # Act
    idir = store.create_identity(sample_identity)

    # Assert
    assert (idir / "identity.toml").exists()
    assert (idir / "consent.json").exists()


def test_create_identity_duplicate_raises(
    store: VoicePromptStore,
    sample_identity: Identity,
) -> None:
    # Arrange
    store.create_identity(sample_identity)

    # Act / Assert
    with pytest.raises(ValueError):
        store.create_identity(sample_identity)


def test_get_identity_roundtrip(
    store: VoicePromptStore,
    sample_identity: Identity,
) -> None:
    # Arrange
    store.create_identity(sample_identity)

    # Act
    retrieved = store.get_identity(sample_identity.id)

    # Assert
    assert retrieved.id == sample_identity.id
    assert retrieved.name == sample_identity.name
    assert retrieved.description == sample_identity.description
    assert retrieved.default_style == sample_identity.default_style
    assert retrieved.created_at == sample_identity.created_at
    assert retrieved.metadata == sample_identity.metadata
    rc = retrieved.consent_record
    ec = sample_identity.consent_record
    assert rc.timestamp == ec.timestamp
    assert rc.scope == ec.scope
    assert rc.jurisdiction == ec.jurisdiction
    assert rc.transferable == ec.transferable
    assert rc.document_hash == ec.document_hash


def test_get_identity_not_found_raises(store: VoicePromptStore) -> None:
    # Act / Assert
    with pytest.raises(FileNotFoundError):
        store.get_identity("nonexistent")


def test_list_identities_empty(store: VoicePromptStore) -> None:
    # Act
    result = store.list_identities()

    # Assert
    assert result == []


def test_list_identities_returns_sorted(
    store: VoicePromptStore,
    consent_record: ConsentRecord,
) -> None:
    # Arrange
    bob = Identity(
        id="bob",
        name="Bob",
        description=None,
        default_style="default",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=consent_record,
    )
    alice = Identity(
        id="alice",
        name="Alice",
        description=None,
        default_style="default",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=consent_record,
    )

    # Act
    store.create_identity(bob)
    store.create_identity(alice)
    result = store.list_identities()

    # Assert
    assert result == ["alice", "bob"]


def test_delete_identity_removes_directory(
    store: VoicePromptStore,
    sample_identity: Identity,
) -> None:
    # Arrange
    idir = store.create_identity(sample_identity)

    # Act
    store.delete_identity(sample_identity.id)

    # Assert
    assert not idir.exists()


def test_add_style_copies_audio_and_creates_files(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)

    # Act
    sdir = store.add_style(sample_style, ref_audio_file)

    # Assert
    assert (sdir / "ref_audio.wav").exists()
    assert (sdir / "style.toml").exists()
    assert (sdir / "ref_text.txt").exists()
    assert (sdir / "prompts").is_dir()


def test_add_style_without_identity_raises(
    store: VoicePromptStore,
    sample_style: Style,
    ref_audio_file: Path,
) -> None:
    # Act / Assert
    with pytest.raises(FileNotFoundError):
        store.add_style(sample_style, ref_audio_file)


def test_get_style_roundtrip(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    store.add_style(sample_style, ref_audio_file)

    # Act
    retrieved = store.get_style(sample_identity.id, sample_style.id)

    # Assert
    assert retrieved.id == sample_style.id
    assert retrieved.identity_id == sample_style.identity_id
    assert retrieved.label == sample_style.label
    assert retrieved.description == sample_style.description
    assert retrieved.default_engine == sample_style.default_engine
    assert retrieved.ref_text == sample_style.ref_text
    assert retrieved.language == sample_style.language
    assert retrieved.metadata == sample_style.metadata


def test_list_styles_returns_sorted(
    store: VoicePromptStore,
    sample_identity: Identity,
    ref_audio_file: Path,
    consent_record: ConsentRecord,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    technical = Style(
        id="technical",
        identity_id="tom",
        label="Technical",
        description="Precise and clear",
        default_engine="qwen3-tts",
        ref_audio_path="/tmp/tech.wav",
        ref_text="Let me explain the algorithm",
        language="en-US",
        metadata={},
    )
    conversational = Style(
        id="conversational",
        identity_id="tom",
        label="Conversational",
        description="Relaxed and warm",
        default_engine="qwen3-tts",
        ref_audio_path="/tmp/conv.wav",
        ref_text="So last week I was thinking",
        language="en-US",
        metadata={},
    )

    # Act
    store.add_style(technical, ref_audio_file)
    store.add_style(conversational, ref_audio_file)
    result = store.list_styles(sample_identity.id)

    # Assert
    assert result == ["conversational", "technical"]


def test_delete_style_removes_directory(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    sdir = store.add_style(sample_style, ref_audio_file)

    # Act
    store.delete_style(sample_identity.id, sample_style.id)

    # Assert
    assert not sdir.exists()


def test_get_prompt_path_returns_none_when_missing(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    store.add_style(sample_style, ref_audio_file)

    # Act
    result = store.get_prompt_path(sample_identity.id, sample_style.id, "qwen3-tts")

    # Assert
    assert result is None


def test_get_prompt_path_returns_path_when_exists(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
    tmp_path: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    store.add_style(sample_style, ref_audio_file)
    dummy_prompt = tmp_path / "dummy.safetensors"
    dummy_prompt.write_bytes(b"fake safetensors")
    iid = sample_identity.id
    sid = sample_style.id
    store.set_prompt_path(iid, sid, "qwen3-tts", dummy_prompt)

    # Act
    result = store.get_prompt_path(iid, sid, "qwen3-tts")

    # Assert
    assert result is not None
    assert result.exists()
    assert result.name == "qwen3-tts.safetensors"


def test_invalidate_prompt_cache_single_engine(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
    tmp_path: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    store.add_style(sample_style, ref_audio_file)
    dummy = tmp_path / "dummy.safetensors"
    dummy.write_bytes(b"fake")
    iid = sample_identity.id
    sid = sample_style.id
    store.set_prompt_path(iid, sid, "qwen3-tts", dummy)

    # Act
    store.invalidate_prompt_cache(iid, sid, engine="qwen3-tts")

    # Assert
    assert store.get_prompt_path(iid, sid, "qwen3-tts") is None


def test_invalidate_prompt_cache_all_engines(
    store: VoicePromptStore,
    sample_identity: Identity,
    sample_style: Style,
    ref_audio_file: Path,
    tmp_path: Path,
) -> None:
    # Arrange
    store.create_identity(sample_identity)
    store.add_style(sample_style, ref_audio_file)
    dummy = tmp_path / "dummy.safetensors"
    dummy.write_bytes(b"fake")
    iid = sample_identity.id
    sid = sample_style.id
    store.set_prompt_path(iid, sid, "engine-a", dummy)
    store.set_prompt_path(iid, sid, "engine-b", dummy)

    # Act
    store.invalidate_prompt_cache(iid, sid)

    # Assert
    assert store.get_prompt_path(iid, sid, "engine-a") is None
    assert store.get_prompt_path(iid, sid, "engine-b") is None
