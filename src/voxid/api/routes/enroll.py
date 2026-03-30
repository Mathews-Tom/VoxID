from __future__ import annotations

import datetime
import io
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf
from fastapi import APIRouter, Depends, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse

from voxid.api.deps import get_voxid
from voxid.api.models import (
    CompleteSessionResponse,
    ConsentRecordResponse,
    ConsentStatementResponse,
    ConsentStatusResponse,
    CreateEnrollSessionRequest,
    EnrollPromptResponse,
    EnrollPromptsResponse,
    EnrollSessionResponse,
    QualityReportResponse,
    UploadSampleResponse,
)
from voxid.core import VoxID
from voxid.enrollment.consent import ConsentManager
from voxid.enrollment.multilingual import (
    MultilingualScriptGenerator,
    get_language_config,
)
from voxid.enrollment.preprocessor import AudioPreprocessor
from voxid.enrollment.quality_gate import QualityGate
from voxid.enrollment.recorder import save_recording
from voxid.enrollment.script_generator import EnrollmentPrompt, ScriptGenerator
from voxid.enrollment.session import (
    EnrollmentSample,
    EnrollmentSession,
    SessionStatus,
    SessionStore,
)

router = APIRouter(prefix="/enroll", tags=["enrollment"])

_gate = QualityGate()
_preprocessor = AudioPreprocessor()
_generator = ScriptGenerator()
_ml_generator = MultilingualScriptGenerator()


def _get_session_store(vox: VoxID = Depends(get_voxid)) -> SessionStore:
    return SessionStore(vox._store._root)


def _prompt_to_response(
    prompt: object,
) -> EnrollPromptResponse:
    from voxid.enrollment.script_generator import EnrollmentPrompt

    assert isinstance(prompt, EnrollmentPrompt)
    return EnrollPromptResponse(
        text=prompt.text,
        style=prompt.style,
        unique_phoneme_count=prompt.unique_phoneme_count,
        nasal_count=prompt.nasal_count,
        affricate_count=prompt.affricate_count,
    )


def _session_to_response(session: EnrollmentSession) -> EnrollSessionResponse:
    current = session.current_prompt()
    return EnrollSessionResponse(
        session_id=session.session_id,
        identity_id=session.identity_id,
        styles=session.styles,
        status=session.status.value,
        prompts_per_style=session.prompts_per_style,
        current_style=session.current_style(),
        current_prompt=(
            _prompt_to_response(current) if current is not None else None
        ),
        progress=session.progress_summary(),
    )


def _load_session(
    session_id: str,
    store: SessionStore,
) -> EnrollmentSession:
    try:
        return store.load(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        ) from exc


@router.post("/sessions", response_model=EnrollSessionResponse, status_code=201)
async def create_session(
    req: CreateEnrollSessionRequest,
    vox: VoxID = Depends(get_voxid),
    store: SessionStore = Depends(_get_session_store),
) -> EnrollSessionResponse:
    if req.identity_id not in vox.list_identities():
        raise HTTPException(
            status_code=404,
            detail=f"Identity '{req.identity_id}' not found",
        )

    is_multilingual = req.language is not None and req.language != "en"

    if is_multilingual and req.language is not None:
        try:
            get_language_config(req.language)
        except KeyError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        ml_prompts = {
            s: _ml_generator.select_prompts(
                req.language, count=req.prompts_per_style,
            )
            for s in req.styles
        }
        prompts: dict[str, list[EnrollmentPrompt]] = {
            s: [
                EnrollmentPrompt(
                    text=mp.text,
                    style=s,
                    phonemes=mp.phonemes,
                    unique_phoneme_count=mp.unique_phoneme_count,
                    nasal_count=0,
                    affricate_count=0,
                )
                for mp in ml_list
            ]
            for s, ml_list in ml_prompts.items()
        }
    else:
        prompts = {
            s: _generator.select_prompts(s, count=req.prompts_per_style)
            for s in req.styles
        }

    session = EnrollmentSession(
        session_id=str(uuid.uuid4())[:8],
        identity_id=req.identity_id,
        styles=req.styles,
        started_at=datetime.datetime.now(tz=datetime.UTC).isoformat(),
        status=SessionStatus.IN_PROGRESS,
        prompts_per_style=req.prompts_per_style,
        prompts=prompts,
        language=req.language,
    )
    store.save(session)
    return _session_to_response(session)


@router.get("/sessions/{session_id}", response_model=EnrollSessionResponse)
async def get_session(
    session_id: str,
    store: SessionStore = Depends(_get_session_store),
) -> EnrollSessionResponse:
    session = _load_session(session_id, store)
    return _session_to_response(session)


@router.post(
    "/sessions/{session_id}/samples",
    response_model=UploadSampleResponse,
)
async def upload_sample(
    session_id: str,
    file: UploadFile,
    store: SessionStore = Depends(_get_session_store),
    vox: VoxID = Depends(get_voxid),
) -> UploadSampleResponse:
    session = _load_session(session_id, store)

    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=409,
            detail=f"Session is {session.status.value}, not in_progress",
        )

    prompt = session.current_prompt()
    style = session.current_style()
    if prompt is None or style is None:
        raise HTTPException(
            status_code=409,
            detail="No more prompts remaining in session",
        )

    contents = await file.read()
    audio_data, sr = sf.read(io.BytesIO(contents))
    audio_arr = np.asarray(audio_data, dtype=np.float64)

    report = _gate.validate(audio_arr, sr)

    qr = QualityReportResponse(
        passed=report.passed,
        snr_db=report.snr_db,
        rms_dbfs=report.rms_dbfs,
        peak_dbfs=report.peak_dbfs,
        speech_ratio=report.speech_ratio,
        net_speech_duration_s=report.net_speech_duration_s,
        total_duration_s=report.total_duration_s,
        sample_rate=report.sample_rate,
        warnings=report.warnings,
        rejection_reasons=report.rejection_reasons,
    )

    if report.passed:
        processed, proc_sr = _preprocessor.process(audio_arr, sr)
        audio_dir = (
            vox._store._root / "enrollment_sessions"
            / session.session_id / "samples"
        )
        audio_path = audio_dir / f"{style}_{session.current_prompt_index}.wav"
        save_recording(processed, proc_sr, audio_path)

        sample = EnrollmentSample(
            prompt_index=session.current_prompt_index,
            prompt_text=prompt.text,
            style_id=style,
            attempt=1,
            audio_path=str(audio_path),
            duration_s=report.total_duration_s,
            quality_report=report,
            accepted=True,
            rejection_reason=None,
        )
        session.accept_sample(sample)
        session.advance()
    else:
        session.reject_sample(
            session.current_prompt_index,
            "; ".join(report.rejection_reasons),
        )
        session.advance()

    store.save(session)

    next_prompt = session.current_prompt()
    return UploadSampleResponse(
        accepted=report.passed,
        quality_report=qr,
        next_prompt=(
            _prompt_to_response(next_prompt)
            if next_prompt is not None
            else None
        ),
    )


@router.post(
    "/sessions/{session_id}/complete",
    response_model=CompleteSessionResponse,
)
async def complete_session(
    session_id: str,
    store: SessionStore = Depends(_get_session_store),
    vox: VoxID = Depends(get_voxid),
) -> CompleteSessionResponse:
    session = _load_session(session_id, store)

    if session.status != SessionStatus.IN_PROGRESS:
        raise HTTPException(
            status_code=409,
            detail=f"Session is {session.status.value}, not in_progress",
        )

    registered: list[str] = []
    for style in session.styles:
        best = session.best_sample_for_style(style)
        if best is None or best.audio_path is None:
            continue
        vox.add_style(
            identity_id=session.identity_id,
            id=style,
            label=style.replace("_", " ").title(),
            description=f"Enrolled {style} style",
            ref_audio=best.audio_path,
            ref_text=best.prompt_text,
        )
        registered.append(style)

    session.complete()
    store.save(session)

    return CompleteSessionResponse(
        session_id=session.session_id,
        styles_registered=registered,
    )


@router.delete("/sessions/{session_id}", status_code=204)
async def delete_session(
    session_id: str,
    store: SessionStore = Depends(_get_session_store),
) -> None:
    try:
        store.delete(session_id)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found",
        ) from exc


@router.get("/prompts", response_model=EnrollPromptsResponse)
async def get_prompts(
    style: str = Query(..., description="Style ID"),
    count: int = Query(5, description="Number of prompts"),
) -> EnrollPromptsResponse:
    try:
        prompts = _generator.select_prompts(style, count=count)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return EnrollPromptsResponse(
        style=style,
        prompts=[_prompt_to_response(p) for p in prompts],
    )


@router.get("/prompts/next", response_model=EnrollPromptResponse | None)
async def get_next_prompt(
    session_id: str = Query(..., description="Session ID"),
    store: SessionStore = Depends(_get_session_store),
) -> EnrollPromptResponse | None:
    session = _load_session(session_id, store)
    style = session.current_style()
    if style is None:
        return None

    tracker = session.phoneme_trackers.get(style)
    if tracker is None:
        prompt = session.current_prompt()
        return _prompt_to_response(prompt) if prompt is not None else None

    next_prompt = _generator.select_next_adaptive(
        style, tracker,
        exclude_texts=[s.prompt_text for s in session.samples],
    )
    if next_prompt is not None:
        return _prompt_to_response(next_prompt)

    prompt = session.current_prompt()
    return _prompt_to_response(prompt) if prompt is not None else None


# ── Consent endpoints ──────────────────────────────────────────────────


def _get_consent_manager(vox: VoxID = Depends(get_voxid)) -> ConsentManager:
    return ConsentManager(vox._store._root)


@router.get("/consent/statement", response_model=ConsentStatementResponse)
async def get_consent_statement(
    identity_id: str = Query(..., description="Identity ID"),
    vox: VoxID = Depends(get_voxid),
) -> ConsentStatementResponse:
    if identity_id not in vox.list_identities():
        raise HTTPException(status_code=404, detail=f"Identity '{identity_id}' not found")
    identity = vox._store.get_identity(identity_id)
    mgr = ConsentManager(vox._store._root)
    statement = mgr.generate_statement(identity.name)
    return ConsentStatementResponse(
        identity_id=identity_id,
        identity_name=identity.name,
        statement=statement,
    )


@router.get("/consent/{identity_id}/status", response_model=ConsentStatusResponse)
async def get_consent_status(
    identity_id: str,
    consent_mgr: ConsentManager = Depends(_get_consent_manager),
) -> ConsentStatusResponse:
    return ConsentStatusResponse(
        identity_id=identity_id,
        has_consent=consent_mgr.verify_consent_exists(identity_id),
    )


@router.post("/consent/{identity_id}", response_model=ConsentRecordResponse)
async def upload_consent(
    identity_id: str,
    file: UploadFile,
    vox: VoxID = Depends(get_voxid),
) -> ConsentRecordResponse:
    if identity_id not in vox.list_identities():
        raise HTTPException(status_code=404, detail=f"Identity '{identity_id}' not found")

    contents = await file.read()
    audio_data, sr = sf.read(io.BytesIO(contents))
    audio_arr = np.asarray(audio_data, dtype=np.float64)

    report = _gate.validate(audio_arr, sr)
    if not report.passed:
        raise HTTPException(
            status_code=422,
            detail={
                "message": "Consent recording failed quality check",
                "rejection_reasons": report.rejection_reasons,
                "snr_db": report.snr_db,
                "rms_dbfs": report.rms_dbfs,
                "peak_dbfs": report.peak_dbfs,
                "speech_ratio": report.speech_ratio,
                "total_duration_s": report.total_duration_s,
            },
        )

    mgr = ConsentManager(vox._store._root)
    record = mgr.record_consent(
        identity_id=identity_id,
        audio=audio_arr,
        sr=sr,
        scope="text-to-speech generation",
    )
    return ConsentRecordResponse(
        identity_id=identity_id,
        timestamp=record.timestamp,
        scope=record.scope,
        jurisdiction=record.jurisdiction,
        document_hash=record.document_hash,
    )


# ── Sample audio streaming ─────────────────────────────────────────────


@router.get("/sessions/{session_id}/samples/{sample_index}/audio")
async def get_sample_audio(
    session_id: str,
    sample_index: int,
    store: SessionStore = Depends(_get_session_store),
) -> FileResponse:
    session = _load_session(session_id, store)

    accepted = [s for s in session.samples if s.accepted]
    if sample_index < 0 or sample_index >= len(accepted):
        raise HTTPException(status_code=404, detail="Sample not found")

    sample = accepted[sample_index]
    if sample.audio_path is None or not Path(sample.audio_path).exists():
        raise HTTPException(status_code=404, detail="Sample audio file not found")

    return FileResponse(sample.audio_path, media_type="audio/wav")
