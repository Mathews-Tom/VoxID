from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING

import numpy as np

from .consent import ConsentManager
from .phoneme_tracker import (
    ALL_PHONEMES,
    PHONEME_WEIGHTS,
    PhonemeTracker,
    load_cmudict,
    text_to_phonemes,
)
from .preprocessor import AudioPreprocessor
from .quality_gate import QualityConfig, QualityGate, QualityReport, estimate_snr
from .recorder import (
    AudioRecorder,
    RecordingMetrics,
    detect_speech_energy,
    save_recording,
)
from .script_generator import EnrollmentPrompt, ScriptGenerator
from .session import (
    EnrollmentSample,
    EnrollmentSession,
    SessionStatus,
    SessionStore,
)

if TYPE_CHECKING:
    from voxid.core import VoxID
    from voxid.models import Style


class EnrollmentPipeline:
    """High-level orchestrator for the enrollment workflow.

    Wraps ScriptGenerator, QualityGate, AudioPreprocessor, and
    SessionStore into a unified API used by CLI, REST, and
    programmatic callers.
    """

    def __init__(self, voxid: VoxID) -> None:
        self._voxid = voxid
        self._generator = ScriptGenerator()
        self._gate = QualityGate()
        self._preprocessor = AudioPreprocessor()
        self._session_store = SessionStore(voxid._store._root)

    def create_session(
        self,
        identity_id: str,
        styles: list[str],
        prompts_per_style: int = 5,
    ) -> EnrollmentSession:
        """Create a new enrollment session with generated prompts."""
        if identity_id not in self._voxid.list_identities():
            raise ValueError(
                f"Identity '{identity_id}' not found"
            )

        prompts: dict[str, list[EnrollmentPrompt]] = {
            s: self._generator.select_prompts(s, count=prompts_per_style)
            for s in styles
        }
        session = EnrollmentSession(
            session_id=str(uuid.uuid4())[:8],
            identity_id=identity_id,
            styles=styles,
            started_at=datetime.datetime.now(
                tz=datetime.UTC,
            ).isoformat(),
            status=SessionStatus.IN_PROGRESS,
            prompts_per_style=prompts_per_style,
            prompts=prompts,
        )
        self._session_store.save(session)
        return session

    def record_sample(
        self,
        session: EnrollmentSession,
        audio: np.ndarray,
        sr: int,
    ) -> tuple[EnrollmentSample, QualityReport]:
        """Validate and process an audio sample for the current prompt."""
        prompt = session.current_prompt()
        style = session.current_style()
        if prompt is None or style is None:
            raise RuntimeError("No more prompts remaining in session")

        report = self._gate.validate(audio, sr)

        if report.passed:
            processed, proc_sr = self._preprocessor.process(audio, sr)
            audio_dir = (
                self._voxid._store._root / "enrollment_sessions"
                / session.session_id / "samples"
            )
            audio_path = (
                audio_dir
                / f"{style}_{session.current_prompt_index}.wav"
            )
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
            sample = EnrollmentSample(
                prompt_index=session.current_prompt_index,
                prompt_text=prompt.text,
                style_id=style,
                attempt=1,
                audio_path=None,
                duration_s=report.total_duration_s,
                quality_report=report,
                accepted=False,
                rejection_reason="; ".join(report.rejection_reasons),
            )
            session.reject_sample(
                session.current_prompt_index,
                "; ".join(report.rejection_reasons),
            )
            session.advance()

        self._session_store.save(session)
        return sample, report

    def finalize(
        self, session: EnrollmentSession,
    ) -> list[Style]:
        """Select best samples per style and register as VoxID styles."""
        registered: list[Style] = []

        for style_id in session.styles:
            best = session.best_sample_for_style(style_id)
            if best is None or best.audio_path is None:
                continue

            s = self._voxid.add_style(
                identity_id=session.identity_id,
                id=style_id,
                label=style_id.replace("_", " ").title(),
                description=f"Enrolled {style_id} style",
                ref_audio=best.audio_path,
                ref_text=best.prompt_text,
            )
            registered.append(s)

        session.complete()
        self._session_store.save(session)
        return registered


__all__ = [
    "ALL_PHONEMES",
    "AudioPreprocessor",
    "AudioRecorder",
    "ConsentManager",
    "EnrollmentPipeline",
    "EnrollmentPrompt",
    "EnrollmentSample",
    "EnrollmentSession",
    "PHONEME_WEIGHTS",
    "PhonemeTracker",
    "QualityConfig",
    "QualityGate",
    "QualityReport",
    "RecordingMetrics",
    "ScriptGenerator",
    "SessionStatus",
    "SessionStore",
    "detect_speech_energy",
    "estimate_snr",
    "load_cmudict",
    "save_recording",
    "text_to_phonemes",
]
