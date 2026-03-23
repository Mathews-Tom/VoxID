from __future__ import annotations

from typing import Any

from pydantic import BaseModel

# ── Identity ──────────────────────────────


class CreateIdentityRequest(BaseModel):
    id: str
    name: str
    description: str | None = None
    default_style: str = "conversational"
    metadata: dict[str, Any] = {}


class IdentityResponse(BaseModel):
    id: str
    name: str
    description: str | None
    default_style: str
    created_at: str
    metadata: dict[str, Any]


class IdentityListResponse(BaseModel):
    identities: list[str]


# ── Style ─────────────────────────────────


class AddStyleRequest(BaseModel):
    id: str
    label: str
    description: str
    ref_audio_path: str  # path on server filesystem
    ref_text: str
    engine: str | None = None
    language: str = "en-US"
    metadata: dict[str, Any] = {}


class StyleResponse(BaseModel):
    id: str
    identity_id: str
    label: str
    description: str
    default_engine: str
    language: str
    metadata: dict[str, Any]


class StyleListResponse(BaseModel):
    styles: list[str]


# ── Generate ──────────────────────────────


class GenerateRequest(BaseModel):
    text: str
    identity_id: str
    style: str | None = None
    engine: str | None = None


class GenerateResponse(BaseModel):
    audio_path: str
    sample_rate: int
    style_used: str
    identity_id: str


class GenerateSegmentsRequest(BaseModel):
    text: str
    identity_id: str
    engine: str | None = None
    stitch: bool = True


class SegmentItemResponse(BaseModel):
    index: int
    text: str
    style: str
    audio_path: str
    duration_ms: int
    boundary_type: str


class GenerateSegmentsResponse(BaseModel):
    segments: list[SegmentItemResponse]
    stitched_path: str | None
    total_duration_ms: int


# ── Route ─────────────────────────────────


class RouteRequest(BaseModel):
    text: str
    identity_id: str


class RouteResponse(BaseModel):
    style: str
    confidence: float
    tier: str
    scores: dict[str, float]


# ── Health ────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    version: str


# ── Enrollment ───────────────────────────


class CreateEnrollSessionRequest(BaseModel):
    identity_id: str
    styles: list[str]
    prompts_per_style: int = 5


class EnrollPromptResponse(BaseModel):
    text: str
    style: str
    unique_phoneme_count: int
    nasal_count: int
    affricate_count: int


class EnrollSessionResponse(BaseModel):
    session_id: str
    identity_id: str
    styles: list[str]
    status: str
    prompts_per_style: int
    current_style: str | None
    current_prompt: EnrollPromptResponse | None
    progress: dict[str, Any]


class QualityReportResponse(BaseModel):
    passed: bool
    snr_db: float
    rms_dbfs: float
    peak_dbfs: float
    speech_ratio: float
    net_speech_duration_s: float
    total_duration_s: float
    sample_rate: int
    warnings: list[str]
    rejection_reasons: list[str]


class UploadSampleResponse(BaseModel):
    accepted: bool
    quality_report: QualityReportResponse
    next_prompt: EnrollPromptResponse | None


class EnrollPromptsResponse(BaseModel):
    style: str
    prompts: list[EnrollPromptResponse]


class CompleteSessionResponse(BaseModel):
    session_id: str
    styles_registered: list[str]
