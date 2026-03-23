from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import voxid.adapters.stub  # noqa: F401
from voxid.config import VoxIDConfig
from voxid.core import VoxID
from voxid.enrollment import (
    EnrollmentPipeline,
    SessionStatus,
    SessionStore,
)


def _make_good_audio(sr: int = 24000, duration_s: float = 5.0) -> np.ndarray:
    """Generate audio that passes all quality gates."""
    rng = np.random.default_rng(42)
    noise_floor = (0.001 * rng.standard_normal(int(sr * 0.5))).astype(np.float64)
    t = np.linspace(0, duration_s - 0.5, int(sr * (duration_s - 0.5)), endpoint=False)
    signal = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float64)
    return np.concatenate([noise_floor, signal])


def _make_bad_audio(sr: int = 24000) -> np.ndarray:
    """Generate audio that fails quality gates (too short)."""
    return np.zeros(int(sr * 0.5), dtype=np.float64)


@pytest.fixture
def vox(tmp_path: Path) -> VoxID:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    v = VoxID(config=config)
    v.create_identity(id="alice", name="Alice")
    return v


@pytest.fixture
def pipeline(vox: VoxID) -> EnrollmentPipeline:
    return EnrollmentPipeline(vox)


# --- EnrollmentPipeline ---


class TestPipelineCreateSession:
    def test_creates_session(self, pipeline: EnrollmentPipeline) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=2,
        )
        assert session.identity_id == "alice"
        assert session.status == SessionStatus.IN_PROGRESS
        assert len(session.prompts["phonetic"]) == 2

    def test_invalid_identity_raises(
        self, pipeline: EnrollmentPipeline,
    ) -> None:
        with pytest.raises(ValueError, match="not found"):
            pipeline.create_session("nonexistent", ["phonetic"])


class TestPipelineRecordSample:
    def test_good_audio_accepted(
        self, pipeline: EnrollmentPipeline,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=2,
        )
        audio = _make_good_audio()
        sample, report = pipeline.record_sample(session, audio, 24000)
        assert sample.accepted is True
        assert report.passed is True
        assert sample.audio_path is not None

    def test_bad_audio_rejected(
        self, pipeline: EnrollmentPipeline,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=2,
        )
        audio = _make_bad_audio()
        sample, report = pipeline.record_sample(session, audio, 24000)
        assert sample.accepted is False
        assert report.passed is False
        assert sample.rejection_reason is not None

    def test_advances_prompt_after_accept(
        self, pipeline: EnrollmentPipeline,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=2,
        )
        first_text = session.current_prompt()
        assert first_text is not None

        pipeline.record_sample(session, _make_good_audio(), 24000)
        second_text = session.current_prompt()
        assert second_text is not None
        assert second_text.text != first_text.text


class TestPipelineFinalize:
    def test_registers_styles(
        self, pipeline: EnrollmentPipeline, vox: VoxID,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=1,
        )
        pipeline.record_sample(session, _make_good_audio(), 24000)
        styles = pipeline.finalize(session)
        assert len(styles) == 1
        assert styles[0].id == "phonetic"
        assert "phonetic" in vox.list_styles("alice")

    def test_skips_styles_without_samples(
        self, pipeline: EnrollmentPipeline, vox: VoxID,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic", "conversational"],
            prompts_per_style=1,
        )
        # Only record for phonetic, skip conversational
        pipeline.record_sample(session, _make_good_audio(), 24000)
        # Advance past conversational (reject with bad audio)
        pipeline.record_sample(session, _make_bad_audio(), 24000)

        styles = pipeline.finalize(session)
        registered_ids = [s.id for s in styles]
        assert "phonetic" in registered_ids
        assert "conversational" not in registered_ids

    def test_sets_status_complete(
        self, pipeline: EnrollmentPipeline,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=1,
        )
        pipeline.record_sample(session, _make_good_audio(), 24000)
        pipeline.finalize(session)
        assert session.status == SessionStatus.COMPLETE

    def test_ref_text_matches_prompt(
        self, pipeline: EnrollmentPipeline, vox: VoxID,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=1,
        )
        prompt_text = session.current_prompt()
        assert prompt_text is not None

        pipeline.record_sample(session, _make_good_audio(), 24000)
        styles = pipeline.finalize(session)
        assert styles[0].ref_text == prompt_text.text

    def test_full_flow_two_styles(
        self, pipeline: EnrollmentPipeline, vox: VoxID,
    ) -> None:
        session = pipeline.create_session(
            "alice", ["phonetic", "conversational"],
            prompts_per_style=2,
        )
        # Record all prompts
        for _ in range(4):
            if session.current_prompt() is None:
                break
            pipeline.record_sample(session, _make_good_audio(), 24000)

        styles = pipeline.finalize(session)
        assert len(styles) == 2
        assert {s.id for s in styles} == {"phonetic", "conversational"}


# --- VoxID.enroll() ---


class TestVoxIDEnroll:
    def test_creates_session(self, vox: VoxID) -> None:
        session = vox.enroll("alice", ["phonetic"], prompts_per_style=2)
        assert session.identity_id == "alice"
        assert session.status == SessionStatus.IN_PROGRESS

    def test_invalid_identity_raises(self, vox: VoxID) -> None:
        with pytest.raises(ValueError, match="not found"):
            vox.enroll("nonexistent", ["phonetic"])


# --- End-to-End ---


class TestEndToEnd:
    def test_enroll_then_generate(
        self, vox: VoxID, tmp_path: Path,
    ) -> None:
        """Full flow: enroll → finalize → generate audio."""
        pipeline = EnrollmentPipeline(vox)
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=1,
        )
        pipeline.record_sample(session, _make_good_audio(), 24000)
        styles = pipeline.finalize(session)
        assert len(styles) == 1

        # Generate with the enrolled style
        audio_path, sr = vox.generate(
            text="Hello world",
            identity_id="alice",
            style="phonetic",
        )
        assert Path(audio_path).exists()
        assert sr > 0

    def test_session_resume_after_interrupt(
        self, vox: VoxID, tmp_path: Path,
    ) -> None:
        """Resume a session after saving and reloading."""
        pipeline = EnrollmentPipeline(vox)
        session = pipeline.create_session(
            "alice", ["phonetic"], prompts_per_style=3,
        )

        # Record 1 sample then "interrupt"
        pipeline.record_sample(session, _make_good_audio(), 24000)
        session_id = session.session_id

        # Reload from disk
        store = SessionStore(vox._store._root)
        restored = store.load(session_id)

        assert restored.session_id == session_id
        assert restored.current_prompt_index == session.current_prompt_index
        assert len(restored.samples) == 1

        # Continue with fresh pipeline
        pipeline2 = EnrollmentPipeline(vox)
        pipeline2.record_sample(restored, _make_good_audio(), 24000)
        pipeline2.record_sample(restored, _make_good_audio(), 24000)

        styles = pipeline2.finalize(restored)
        assert len(styles) == 1
        assert restored.status == SessionStatus.COMPLETE
