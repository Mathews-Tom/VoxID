from __future__ import annotations

from dataclasses import dataclass

from voxid.models import ConsentRecord, Identity


@dataclass(frozen=True)
class ConsentValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]


VALID_SCOPES: frozenset[str] = frozenset({"personal", "commercial"})
VALID_JURISDICTIONS: frozenset[str] = frozenset(
    {
        "US",
        "EU",
        "UK",
        "CA",
        "AU",
        "JP",
        "KR",
        "BR",
        "IN",
    }
)


def validate_consent(
    consent: ConsentRecord,
) -> ConsentValidationResult:
    """Validate a consent record.

    Checks:
    - scope is one of: personal, commercial
    - jurisdiction is a known value
    - timestamp is non-empty
    - document_hash is non-empty
    """
    errors: list[str] = []
    warnings: list[str] = []

    if consent.scope not in VALID_SCOPES:
        errors.append(
            f"Invalid scope {consent.scope!r}. "
            f"Must be one of: {set(VALID_SCOPES)}"
        )

    if consent.jurisdiction not in VALID_JURISDICTIONS:
        warnings.append(
            f"Unknown jurisdiction {consent.jurisdiction!r}. "
            f"Known: {set(VALID_JURISDICTIONS)}"
        )

    if not consent.timestamp:
        errors.append("Consent timestamp is required")

    if not consent.document_hash:
        errors.append("Consent document_hash is required")

    return ConsentValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def check_export_consent(
    identity: Identity,
    target_scope: str = "personal",
) -> ConsentValidationResult:
    """Check if an identity's consent allows export.

    Rules:
    - transferable must be True for any export
    - scope must be >= target_scope
      (commercial covers personal, but not vice versa)
    - consent must be valid
    """
    errors: list[str] = []
    warnings: list[str] = []

    # Basic validation first
    base = validate_consent(identity.consent_record)
    errors.extend(base.errors)
    warnings.extend(base.warnings)

    consent = identity.consent_record

    if not consent.transferable:
        errors.append(
            f"Identity {identity.id!r} is not transferable. "
            "Export requires transferable=True in consent."
        )

    # Scope hierarchy: commercial > personal
    scope_level: dict[str, int] = {"personal": 1, "commercial": 2}
    consent_level = scope_level.get(consent.scope, 0)
    target_level = scope_level.get(target_scope, 0)

    if consent_level < target_level:
        errors.append(
            f"Consent scope {consent.scope!r} is insufficient "
            f"for target scope {target_scope!r}. "
            "Commercial use requires commercial consent."
        )

    return ConsentValidationResult(
        valid=len(errors) == 0,
        errors=errors,
        warnings=warnings,
    )


def check_import_consent(
    consent: ConsentRecord,
) -> ConsentValidationResult:
    """Validate consent record from an imported archive.

    Same as validate_consent but may add import-specific checks.
    """
    result = validate_consent(consent)
    warnings = list(result.warnings)

    if consent.document_hash == "none":
        warnings.append(
            "Consent document_hash is 'none' — "
            "this identity was created without explicit consent. "
            "Consider re-recording consent."
        )

    return ConsentValidationResult(
        valid=result.valid,
        errors=list(result.errors),
        warnings=warnings,
    )
