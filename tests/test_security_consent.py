from __future__ import annotations

import pytest

from voxid.models import ConsentRecord, Identity
from voxid.security.consent import (
    check_export_consent,
    check_import_consent,
    validate_consent,
)


@pytest.fixture
def valid_consent() -> ConsentRecord:
    return ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="commercial",
        jurisdiction="US",
        transferable=True,
        document_hash="sha256:abc123",
    )


@pytest.fixture
def valid_identity(valid_consent: ConsentRecord) -> Identity:
    return Identity(
        id="tom",
        name="Tom",
        description="Test",
        default_style="conversational",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=valid_consent,
    )


def test_validate_consent_valid(valid_consent: ConsentRecord) -> None:
    result = validate_consent(valid_consent)
    assert result.valid is True
    assert result.errors == []


def test_validate_consent_invalid_scope() -> None:
    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="invalid",
        jurisdiction="US",
        transferable=True,
        document_hash="sha256:abc123",
    )
    result = validate_consent(consent)
    assert result.valid is False
    assert any("scope" in e.lower() for e in result.errors)


def test_validate_consent_unknown_jurisdiction_warns() -> None:
    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="XX",
        transferable=True,
        document_hash="sha256:abc123",
    )
    result = validate_consent(consent)
    assert result.valid is True
    assert len(result.warnings) > 0


def test_validate_consent_empty_timestamp_fails() -> None:
    consent = ConsentRecord(
        timestamp="",
        scope="personal",
        jurisdiction="US",
        transferable=True,
        document_hash="sha256:abc123",
    )
    result = validate_consent(consent)
    assert result.valid is False
    assert any("timestamp" in e.lower() for e in result.errors)


def test_validate_consent_empty_hash_fails() -> None:
    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="US",
        transferable=True,
        document_hash="",
    )
    result = validate_consent(consent)
    assert result.valid is False
    assert any("document_hash" in e.lower() for e in result.errors)


def test_check_export_consent_transferable_passes(
    valid_identity: Identity,
) -> None:
    result = check_export_consent(valid_identity, target_scope="personal")
    assert result.valid is True


def test_check_export_consent_non_transferable_fails() -> None:
    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="commercial",
        jurisdiction="US",
        transferable=False,
        document_hash="sha256:abc123",
    )
    identity = Identity(
        id="locked",
        name="Locked",
        description=None,
        default_style="default",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=consent,
    )
    result = check_export_consent(identity, target_scope="personal")
    assert result.valid is False
    assert any("transferable" in e.lower() for e in result.errors)


def test_check_export_consent_personal_for_commercial_fails() -> None:
    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="US",
        transferable=True,
        document_hash="sha256:abc123",
    )
    identity = Identity(
        id="personal-user",
        name="Personal",
        description=None,
        default_style="default",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=consent,
    )
    result = check_export_consent(identity, target_scope="commercial")
    assert result.valid is False
    assert any("scope" in e.lower() for e in result.errors)


def test_check_export_consent_commercial_for_personal_passes(
    valid_identity: Identity,
) -> None:
    # valid_identity has scope="commercial", transferable=True
    result = check_export_consent(valid_identity, target_scope="personal")
    assert result.valid is True


def test_check_import_consent_valid(valid_consent: ConsentRecord) -> None:
    result = check_import_consent(valid_consent)
    assert result.valid is True


def test_check_import_consent_none_hash_warns() -> None:
    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="US",
        transferable=True,
        document_hash="none",
    )
    result = check_import_consent(consent)
    assert len(result.warnings) > 0
