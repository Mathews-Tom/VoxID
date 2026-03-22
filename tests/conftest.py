from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from voxid.models import ConsentRecord, Identity, Style
from voxid.store import VoicePromptStore


@pytest.fixture
def consent_record() -> ConsentRecord:
    return ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="US",
        transferable=False,
        document_hash="abc123",
    )


@pytest.fixture
def sample_identity(consent_record: ConsentRecord) -> Identity:
    return Identity(
        id="tom",
        name="Tom",
        description="Test identity",
        default_style="conversational",
        created_at="2026-03-22T00:00:00Z",
        metadata={"locale": "en-US"},
        consent_record=consent_record,
    )


@pytest.fixture
def sample_style() -> Style:
    return Style(
        id="conversational",
        identity_id="tom",
        label="Conversational",
        description="Relaxed, warm, peer-to-peer",
        default_engine="qwen3-tts",
        ref_audio_path="/tmp/test_ref.wav",
        ref_text="So last week I tried making dosa from scratch",
        language="en-US",
        metadata={"energy_level": "medium"},
    )


@pytest.fixture
def store(tmp_path: Path) -> VoicePromptStore:
    return VoicePromptStore(tmp_path / "voxid_test")


@pytest.fixture
def ref_audio_file(tmp_path: Path) -> Path:
    """Create a minimal valid WAV file for testing."""
    audio = np.zeros(24000, dtype=np.float32)  # 1 second of silence at 24kHz
    path = tmp_path / "ref_audio.wav"
    sf.write(str(path), audio, 24000)
    return path
