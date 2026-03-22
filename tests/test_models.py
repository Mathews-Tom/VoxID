from __future__ import annotations

import pytest

from voxid.models import ConsentRecord, Identity, Style


def test_consent_record_to_dict_roundtrip() -> None:
    # Arrange
    record = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="US",
        transferable=False,
        document_hash="abc123",
    )

    # Act
    data = record.to_dict()
    restored = ConsentRecord.from_dict(data)

    # Assert
    assert restored.timestamp == record.timestamp
    assert restored.scope == record.scope
    assert restored.jurisdiction == record.jurisdiction
    assert restored.transferable == record.transferable
    assert restored.document_hash == record.document_hash


def test_identity_to_toml_roundtrip(consent_record: ConsentRecord) -> None:
    # Arrange
    identity = Identity(
        id="tom",
        name="Tom",
        description="Test identity",
        default_style="conversational",
        created_at="2026-03-22T00:00:00Z",
        metadata={"locale": "en-US"},
        consent_record=consent_record,
    )

    # Act
    data = identity.to_toml()
    restored = Identity.from_toml(data, consent_record)

    # Assert
    assert restored.id == identity.id
    assert restored.name == identity.name
    assert restored.description == identity.description
    assert restored.default_style == identity.default_style
    assert restored.created_at == identity.created_at
    assert restored.metadata == identity.metadata
    assert restored.consent_record == identity.consent_record


def test_identity_to_toml_omits_none_description(
    consent_record: ConsentRecord,
) -> None:
    # Arrange
    identity = Identity(
        id="anon",
        name="Anonymous",
        description=None,
        default_style="default",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=consent_record,
    )

    # Act
    data = identity.to_toml()

    # Assert
    assert "description" not in data


def test_style_to_toml_roundtrip() -> None:
    # Arrange
    style = Style(
        id="conversational",
        identity_id="tom",
        label="Conversational",
        description="Relaxed, warm, peer-to-peer",
        default_engine="qwen3-tts",
        ref_audio_path="/tmp/test_ref.wav",
        ref_text="So last week I tried making dosa from scratch",
        language="en-US",
        metadata={"energy_level": "medium"},
    )

    # Act
    data = style.to_toml()
    restored = Style.from_toml(data)

    # Assert
    assert restored.id == style.id
    assert restored.identity_id == style.identity_id
    assert restored.label == style.label
    assert restored.description == style.description
    assert restored.default_engine == style.default_engine
    assert restored.ref_audio_path == style.ref_audio_path
    assert restored.ref_text == style.ref_text
    assert restored.language == style.language
    assert restored.metadata == style.metadata


def test_consent_record_from_dict_missing_field_raises() -> None:
    # Arrange
    incomplete = {
        "timestamp": "2026-03-22T00:00:00Z",
        "scope": "personal",
        # jurisdiction missing
        "transferable": False,
        "document_hash": "abc123",
    }

    # Act / Assert
    with pytest.raises((KeyError, TypeError)):
        ConsentRecord.from_dict(incomplete)
