from __future__ import annotations

from .audit import AuditFinding, AuditReport, scan_for_input_validation, scan_source
from .consent import (
    ConsentValidationResult,
    check_export_consent,
    check_import_consent,
    validate_consent,
)
from .drift import DriftReport, check_drift, cosine_similarity
from .watermark import (
    WatermarkResult,
    detect_watermark,
    detect_watermark_file,
    embed_watermark,
    embed_watermark_file,
    is_audioseal_available,
)

__all__ = [
    # watermark
    "WatermarkResult",
    "embed_watermark",
    "embed_watermark_file",
    "detect_watermark",
    "detect_watermark_file",
    "is_audioseal_available",
    # consent
    "ConsentValidationResult",
    "validate_consent",
    "check_export_consent",
    "check_import_consent",
    # drift
    "DriftReport",
    "check_drift",
    "cosine_similarity",
    # audit
    "AuditReport",
    "AuditFinding",
    "scan_source",
    "scan_for_input_validation",
]
