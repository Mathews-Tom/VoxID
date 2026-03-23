from __future__ import annotations

from pathlib import Path

import pytest

from voxid.enrollment.phoneme_tracker import PhonemeTracker
from voxid.enrollment.quality_gate import QualityReport
from voxid.enrollment.script_generator import EnrollmentPrompt
from voxid.enrollment.session import (
    EnrollmentSample,
    EnrollmentSession,
    SessionStatus,
    SessionStore,
)


def _make_prompt(text: str, style: str) -> EnrollmentPrompt:
    return EnrollmentPrompt(
        text=text,
        style=style,
        phonemes=["HH", "AH", "L", "OW"],
        unique_phoneme_count=4,
        nasal_count=0,
        affricate_count=0,
    )


def _make_quality_report(
    snr: float = 35.0, passed: bool = True,
) -> QualityReport:
    return QualityReport(
        passed=passed,
        snr_db=snr,
        rms_dbfs=-20.0,
        peak_dbfs=-4.0,
        speech_ratio=0.8,
        net_speech_duration_s=4.0,
        total_duration_s=5.0,
        sample_rate=24000,
        warnings=[],
        rejection_reasons=[],
    )


def _make_sample(
    prompt_index: int = 0,
    style_id: str = "conversational",
    accepted: bool = True,
    snr: float = 35.0,
) -> EnrollmentSample:
    return EnrollmentSample(
        prompt_index=prompt_index,
        prompt_text="Hello world test",
        style_id=style_id,
        attempt=1,
        audio_path="/tmp/test.wav",
        duration_s=5.0,
        quality_report=_make_quality_report(snr=snr, passed=accepted),
        accepted=accepted,
        rejection_reason=None if accepted else "Low SNR",
    )


def _make_session(
    styles: list[str] | None = None,
    prompts_per_style: int = 2,
) -> EnrollmentSession:
    styles = styles or ["conversational", "technical"]
    prompts: dict[str, list[EnrollmentPrompt]] = {}
    for style in styles:
        prompts[style] = [
            _make_prompt(f"Prompt {i} for {style}", style)
            for i in range(prompts_per_style)
        ]
    return EnrollmentSession(
        session_id="test-session-001",
        identity_id="alice",
        styles=styles,
        started_at="2026-03-23T10:00:00Z",
        status=SessionStatus.IN_PROGRESS,
        prompts_per_style=prompts_per_style,
        prompts=prompts,
    )


# --- SessionStatus ---


class TestSessionStatus:
    def test_enum_values(self) -> None:
        assert SessionStatus.IN_PROGRESS.value == "in_progress"
        assert SessionStatus.COMPLETE.value == "complete"
        assert SessionStatus.ABANDONED.value == "abandoned"

    def test_session_default_status_is_in_progress(self) -> None:
        session = _make_session()
        assert session.status == SessionStatus.IN_PROGRESS


# --- EnrollmentSample ---


class TestEnrollmentSample:
    def test_roundtrip_serialization(self) -> None:
        # Arrange
        sample = _make_sample()

        # Act
        data = sample.to_dict()
        restored = EnrollmentSample.from_dict(data)

        # Assert
        assert restored.prompt_index == sample.prompt_index
        assert restored.prompt_text == sample.prompt_text
        assert restored.style_id == sample.style_id
        assert restored.accepted == sample.accepted
        assert restored.audio_path == sample.audio_path
        assert restored.duration_s == sample.duration_s
        assert restored.quality_report is not None
        assert restored.quality_report.snr_db == 35.0

    def test_roundtrip_with_none_quality_report(self) -> None:
        sample = EnrollmentSample(
            prompt_index=0,
            prompt_text="test",
            style_id="phonetic",
            attempt=1,
            audio_path=None,
            duration_s=0.0,
            quality_report=None,
            accepted=False,
            rejection_reason="skipped",
        )
        data = sample.to_dict()
        restored = EnrollmentSample.from_dict(data)
        assert restored.quality_report is None
        assert restored.rejection_reason == "skipped"


# --- EnrollmentSession serialization ---


class TestEnrollmentSessionSerialization:
    def test_roundtrip_serialization(self) -> None:
        # Arrange
        session = _make_session()
        sample = _make_sample(style_id="conversational")
        session.accept_sample(sample)
        session.advance()

        # Act
        data = session.to_dict()
        restored = EnrollmentSession.from_dict(data)

        # Assert
        assert restored.session_id == session.session_id
        assert restored.identity_id == session.identity_id
        assert restored.styles == session.styles
        assert restored.status == session.status
        assert restored.current_style_index == session.current_style_index
        assert restored.current_prompt_index == session.current_prompt_index
        assert len(restored.samples) == 1
        assert restored.samples[0].accepted is True
        assert "conversational" in restored.phoneme_trackers
        assert len(restored.prompts["conversational"]) == 2


# --- State machine ---


class TestSessionStateMachine:
    def test_current_prompt_returns_first_initially(self) -> None:
        session = _make_session()
        prompt = session.current_prompt()
        assert prompt is not None
        assert prompt.text == "Prompt 0 for conversational"

    def test_current_style_returns_first_initially(self) -> None:
        session = _make_session()
        assert session.current_style() == "conversational"

    def test_accept_sample_advances_prompt(self) -> None:
        session = _make_session()
        sample = _make_sample(prompt_index=0, style_id="conversational")
        session.accept_sample(sample)
        session.advance()
        prompt = session.current_prompt()
        assert prompt is not None
        assert prompt.text == "Prompt 1 for conversational"

    def test_accept_sample_updates_phoneme_tracker(self) -> None:
        session = _make_session()
        sample = _make_sample(style_id="conversational")
        session.accept_sample(sample)
        assert "conversational" in session.phoneme_trackers
        report = session.phoneme_trackers["conversational"].coverage_report()
        assert any(v > 0 for v in report.values())

    def test_reject_sample_records_reason(self) -> None:
        session = _make_session()
        session.reject_sample(0, "Too noisy")
        assert len(session.samples) == 1
        assert session.samples[0].accepted is False
        assert session.samples[0].rejection_reason == "Too noisy"

    def test_skip_prompt_advances_without_recording(self) -> None:
        session = _make_session()
        session.skip_prompt()
        prompt = session.current_prompt()
        assert prompt is not None
        assert prompt.text == "Prompt 1 for conversational"

    def test_advance_across_style_boundary(self) -> None:
        session = _make_session(prompts_per_style=1)
        # First style has 1 prompt, advance should move to second style
        session.advance()
        assert session.current_style() == "technical"
        assert session.current_prompt_index == 0

    def test_advance_returns_false_when_all_done(self) -> None:
        session = _make_session(prompts_per_style=1)
        assert session.advance() is True  # moves to technical
        assert session.advance() is False  # past all styles

    def test_complete_sets_status(self) -> None:
        session = _make_session()
        session.complete()
        assert session.status == SessionStatus.COMPLETE

    def test_abandon_sets_status(self) -> None:
        session = _make_session()
        session.abandon()
        assert session.status == SessionStatus.ABANDONED

    def test_accept_after_complete_raises(self) -> None:
        session = _make_session()
        session.complete()
        with pytest.raises(RuntimeError, match="not in_progress"):
            session.accept_sample(_make_sample())

    def test_accept_after_abandon_raises(self) -> None:
        session = _make_session()
        session.abandon()
        with pytest.raises(RuntimeError, match="not in_progress"):
            session.accept_sample(_make_sample())

    def test_advance_after_complete_raises(self) -> None:
        session = _make_session()
        session.complete()
        with pytest.raises(RuntimeError, match="not in_progress"):
            session.advance()

    def test_accepted_samples_for_style_filters_correctly(self) -> None:
        session = _make_session()
        session.accept_sample(_make_sample(style_id="conversational"))
        session.accept_sample(
            _make_sample(style_id="conversational", accepted=False),
        )
        session.accept_sample(_make_sample(style_id="technical"))

        conv = session.accepted_samples_for_style("conversational")
        assert len(conv) == 1
        assert conv[0].style_id == "conversational"

    def test_best_sample_for_style_returns_highest_snr(self) -> None:
        session = _make_session()
        session.accept_sample(
            _make_sample(style_id="conversational", snr=30.0),
        )
        session.accept_sample(
            _make_sample(style_id="conversational", snr=45.0),
        )
        session.accept_sample(
            _make_sample(style_id="conversational", snr=38.0),
        )

        best = session.best_sample_for_style("conversational")
        assert best is not None
        assert best.quality_report is not None
        assert best.quality_report.snr_db == 45.0

    def test_best_sample_for_style_returns_none_when_empty(self) -> None:
        session = _make_session()
        assert session.best_sample_for_style("conversational") is None

    def test_progress_summary_structure(self) -> None:
        session = _make_session(prompts_per_style=3)
        session.accept_sample(_make_sample(style_id="conversational"))
        session.reject_sample(1, "noise")

        summary = session.progress_summary()
        assert "conversational" in summary
        assert "technical" in summary
        assert summary["conversational"]["accepted"] == 1
        assert summary["conversational"]["rejected"] == 1
        assert summary["conversational"]["total_prompts"] == 3
        assert summary["conversational"]["coverage_percent"] >= 0.0
        assert summary["technical"]["accepted"] == 0

    def test_full_session_flow_3_styles_5_prompts(self) -> None:
        styles = ["conversational", "technical", "narration"]
        session = _make_session(styles=styles, prompts_per_style=2)

        for style in styles:
            assert session.current_style() == style
            for i in range(2):
                prompt = session.current_prompt()
                assert prompt is not None
                sample = _make_sample(
                    prompt_index=i, style_id=style, snr=35.0 + i,
                )
                session.accept_sample(sample)
                session.advance()

        # All done
        assert session.current_style() is None
        assert session.current_prompt() is None
        session.complete()
        assert session.status == SessionStatus.COMPLETE
        assert len(session.samples) == 6

    def test_current_style_none_when_exhausted(self) -> None:
        session = _make_session(prompts_per_style=0)
        # No prompts means immediately at end
        assert session.current_prompt() is None

    def test_reject_sample_when_no_current_style(self) -> None:
        session = _make_session(
            styles=["conv"], prompts_per_style=1,
        )
        session.advance()  # past the only style
        # reject_sample should not crash even with no current style
        session.reject_sample(0, "late rejection")


# --- SessionStore ---


class TestSessionStore:
    @pytest.fixture
    def store(self, tmp_path: Path) -> SessionStore:
        return SessionStore(tmp_path)

    def test_save_creates_file(
        self, store: SessionStore, tmp_path: Path,
    ) -> None:
        session = _make_session()
        path = store.save(session)
        assert path.exists()
        assert path.suffix == ".json"

    def test_load_restores_session(self, store: SessionStore) -> None:
        session = _make_session()
        session.accept_sample(_make_sample(style_id="conversational"))
        store.save(session)

        restored = store.load("test-session-001")
        assert restored.session_id == session.session_id
        assert len(restored.samples) == 1

    def test_save_load_roundtrip(self, store: SessionStore) -> None:
        session = _make_session()
        session.accept_sample(
            _make_sample(style_id="conversational", snr=42.0),
        )
        session.advance()
        session.advance()  # move to technical style
        store.save(session)

        restored = store.load("test-session-001")
        assert restored.current_style_index == session.current_style_index
        assert restored.current_prompt_index == session.current_prompt_index
        assert restored.samples[0].quality_report is not None
        assert restored.samples[0].quality_report.snr_db == 42.0
        assert (
            restored.progress_summary()
            == session.progress_summary()
        )

    def test_list_sessions_empty(self, store: SessionStore) -> None:
        assert store.list_sessions() == []

    def test_list_sessions_finds_saved(
        self, store: SessionStore,
    ) -> None:
        store.save(_make_session())
        sessions = store.list_sessions()
        assert "test-session-001" in sessions

    def test_list_sessions_filters_by_identity(
        self, store: SessionStore,
    ) -> None:
        s1 = _make_session()
        s1.session_id = "session-alice"
        s1.identity_id = "alice"
        store.save(s1)

        s2 = _make_session()
        s2.session_id = "session-bob"
        s2.identity_id = "bob"
        store.save(s2)

        alice_sessions = store.list_sessions(identity_id="alice")
        assert "session-alice" in alice_sessions
        assert "session-bob" not in alice_sessions

    def test_delete_removes_file(self, store: SessionStore) -> None:
        store.save(_make_session())
        store.delete("test-session-001")
        assert store.list_sessions() == []

    def test_load_nonexistent_raises(
        self, store: SessionStore,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            store.load("nonexistent")

    def test_delete_nonexistent_raises(
        self, store: SessionStore,
    ) -> None:
        with pytest.raises(FileNotFoundError, match="not found"):
            store.delete("nonexistent")
