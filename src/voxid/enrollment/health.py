from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from voxid.core import VoxID

_MAX_AGE_DAYS = 1095  # 3 years


@dataclass(frozen=True)
class EnrollmentHealthReport:
    identity_id: str
    age_days: int
    drift_detected: bool
    re_enrollment_recommended: bool
    reasons: list[str] = field(default_factory=list)


def check_enrollment_health(
    voxid: VoxID,
    identity_id: str,
) -> EnrollmentHealthReport:
    """Assess whether an identity's enrollment should be refreshed.

    Re-enrollment is recommended when:
    - Enrollment age exceeds 3 years (1095 days)
    - Voice drift is detected via existing drift analysis
    """
    identity = voxid._store.get_identity(identity_id)

    # Compute age
    created = datetime.datetime.fromisoformat(identity.created_at)
    now = datetime.datetime.now(tz=datetime.UTC)
    age_days = (now - created).days

    # Check drift across all styles
    drift_detected = False
    reasons: list[str] = []
    styles = voxid.list_styles(identity_id)

    for style_id in styles:
        style_dir = (
            voxid._store._root / "identities" / identity_id / "styles" / style_id
        )
        if not style_dir.exists():
            continue

        from voxid.security.drift import check_drift

        drift_report = check_drift(
            style_dir, identity_id, style_id,
        )
        if drift_report.below_threshold:
            drift_detected = True
            reasons.append(
                f"Voice drift detected for style '{style_id}' "
                f"(similarity: {drift_report.current_similarity:.2f})",
            )

    if age_days > _MAX_AGE_DAYS:
        reasons.append(
            f"Enrollment age ({age_days} days) exceeds {_MAX_AGE_DAYS}-day threshold",
        )

    re_enrollment = drift_detected or age_days > _MAX_AGE_DAYS

    return EnrollmentHealthReport(
        identity_id=identity_id,
        age_days=age_days,
        drift_detected=drift_detected,
        re_enrollment_recommended=re_enrollment,
        reasons=reasons,
    )
