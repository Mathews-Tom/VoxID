from __future__ import annotations

import datetime
import uuid
from typing import TYPE_CHECKING, Literal

import numpy as np

from .consent import ConsentManager
from .health import EnrollmentHealthReport, check_enrollment_health
from .multilingual import (
    LanguageConfig,
    MultilingualScriptGenerator,
    UniversalPhonemeTracker,
    get_language_config,
    list_languages,
)
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
from .vad import VADBackend, detect_speech, detect_speech_silero, detect_speech_webrtc

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
        self._ml_generator = MultilingualScriptGenerator()
        self._gate = QualityGate()
        self._preprocessor = AudioPreprocessor()
        self._session_store = SessionStore(voxid._store._root)

    def create_session(
        self,
        identity_id: str,
        styles: list[str],
        prompts_per_style: int = 5,
        language: str | None = None,
    ) -> EnrollmentSession:
        """Create a new enrollment session with generated prompts.

        When language is None or "en", uses the English ARPAbet pipeline.
        For other languages, uses the multilingual IPA pipeline.
        """
        if identity_id not in self._voxid.list_identities():
            raise ValueError(
                f"Identity '{identity_id}' not found"
            )

        is_multilingual = language is not None and language != "en"

        if is_multilingual:
            ml_prompts = {
                s: self._ml_generator.select_prompts(
                    language, count=prompts_per_style,
                )
                for s in styles
            }
            # Convert MultilingualPrompt → EnrollmentPrompt for session compat
            prompts: dict[str, list[EnrollmentPrompt]] = {}
            for s, ml_list in ml_prompts.items():
                prompts[s] = [
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
        else:
            prompts = {
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
            language=language,
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

    def fuse_samples(
        self,
        samples: list[EnrollmentSample],
        strategy: Literal["best", "concatenate", "average"] = "best",
    ) -> tuple[np.ndarray, int, str]:
        """Fuse multiple enrollment samples into a single reference.

        Returns (audio, sample_rate, ref_text).

        Strategies:
        - best: select highest-SNR sample
        - concatenate: join all with 500ms silence gaps
        - average: same as concatenate (stores individuals for
          future embedding averaging when adapters support it)
        """
        import soundfile as sf_mod

        accepted = [s for s in samples if s.accepted and s.audio_path]
        if not accepted:
            raise ValueError("No accepted samples to fuse")

        if strategy == "best":
            best = max(
                accepted,
                key=lambda s: (
                    s.quality_report.snr_db
                    if s.quality_report is not None
                    else 0.0
                ),
            )
            assert best.audio_path is not None
            audio, sr = sf_mod.read(best.audio_path)
            return (
                np.asarray(audio, dtype=np.float64),
                int(sr),
                best.prompt_text,
            )

        # concatenate and average share the same audio logic
        target_sr: int | None = None
        chunks: list[np.ndarray] = []
        texts: list[str] = []

        for sample in accepted:
            assert sample.audio_path is not None
            audio, sr = sf_mod.read(sample.audio_path)
            arr = np.asarray(audio, dtype=np.float64)
            if target_sr is None:
                target_sr = int(sr)
            elif int(sr) != target_sr:
                n_out = int(len(arr) * target_sr / sr)
                arr = np.interp(
                    np.linspace(0, len(arr) - 1, n_out),
                    np.arange(len(arr)),
                    arr,
                )
            chunks.append(arr)
            texts.append(sample.prompt_text)

        assert target_sr is not None
        silence = np.zeros(int(target_sr * 0.5), dtype=np.float64)

        fused_parts: list[np.ndarray] = []
        for i, chunk in enumerate(chunks):
            if i > 0:
                fused_parts.append(silence)
            fused_parts.append(chunk)

        fused_audio = np.concatenate(fused_parts)
        ref_text = " ".join(texts)
        return fused_audio, target_sr, ref_text


__all__ = [
    "ALL_PHONEMES",
    "AudioPreprocessor",
    "AudioRecorder",
    "ConsentManager",
    "EnrollmentHealthReport",
    "EnrollmentPipeline",
    "EnrollmentPrompt",
    "EnrollmentSample",
    "EnrollmentSession",
    "LanguageConfig",
    "MultilingualScriptGenerator",
    "PHONEME_WEIGHTS",
    "PhonemeTracker",
    "QualityConfig",
    "QualityGate",
    "QualityReport",
    "RecordingMetrics",
    "ScriptGenerator",
    "SessionStatus",
    "SessionStore",
    "UniversalPhonemeTracker",
    "VADBackend",
    "check_enrollment_health",
    "detect_speech",
    "detect_speech_energy",
    "detect_speech_silero",
    "detect_speech_webrtc",
    "estimate_snr",
    "get_language_config",
    "list_languages",
    "load_cmudict",
    "save_recording",
    "text_to_phonemes",
]
