from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

import voxid.adapters.stub  # noqa: F401
from voxid.api.app import create_app
from voxid.api.deps import get_voxid
from voxid.config import VoxIDConfig
from voxid.core import VoxID


def _make_wav_bytes(
    sr: int = 24000,
    duration_s: float = 5.0,
    amplitude: float = 0.3,
) -> bytes:
    """Generate WAV file bytes with noise floor + sine (passes quality gate)."""
    rng = np.random.default_rng(42)
    noise_floor = (0.001 * rng.standard_normal(int(sr * 0.5))).astype(np.float32)
    t = np.linspace(0, duration_s - 0.5, int(sr * (duration_s - 0.5)), endpoint=False)
    signal = (amplitude * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    audio = np.concatenate([noise_floor, signal])
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


def _make_bad_wav_bytes(sr: int = 24000) -> bytes:
    """Generate a very short WAV that fails duration gate."""
    audio = np.zeros(int(sr * 0.5), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV", subtype="PCM_16")
    buf.seek(0)
    return buf.read()


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    vox = VoxID(config=config)
    app = create_app()
    app.dependency_overrides[get_voxid] = lambda: vox
    return TestClient(app)


@pytest.fixture
def seeded_client(client: TestClient) -> TestClient:
    """Client with a pre-created identity."""
    client.post(
        "/identities",
        json={
            "id": "alice",
            "name": "Alice",
            "default_style": "conversational",
        },
    )
    return client


def _create_session(
    client: TestClient,
    styles: list[str] | None = None,
    prompts: int = 2,
) -> dict[str, object]:
    resp = client.post(
        "/enroll/sessions",
        json={
            "identity_id": "alice",
            "styles": styles or ["phonetic"],
            "prompts_per_style": prompts,
        },
    )
    assert resp.status_code == 201
    return resp.json()


class TestCreateSession:
    def test_returns_201(self, seeded_client: TestClient) -> None:
        resp = seeded_client.post(
            "/enroll/sessions",
            json={
                "identity_id": "alice",
                "styles": ["phonetic"],
                "prompts_per_style": 3,
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["identity_id"] == "alice"
        assert data["status"] == "in_progress"
        assert "session_id" in data

    def test_generates_prompts(self, seeded_client: TestClient) -> None:
        data = _create_session(seeded_client, prompts=3)
        assert data["current_prompt"] is not None
        assert data["current_style"] == "phonetic"

    def test_identity_not_found_returns_404(
        self, client: TestClient,
    ) -> None:
        resp = client.post(
            "/enroll/sessions",
            json={
                "identity_id": "nonexistent",
                "styles": ["phonetic"],
            },
        )
        assert resp.status_code == 404


class TestGetSession:
    def test_returns_status(self, seeded_client: TestClient) -> None:
        session = _create_session(seeded_client)
        sid = session["session_id"]
        resp = seeded_client.get(f"/enroll/sessions/{sid}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "in_progress"

    def test_not_found_returns_404(
        self, seeded_client: TestClient,
    ) -> None:
        resp = seeded_client.get("/enroll/sessions/nonexistent")
        assert resp.status_code == 404


class TestUploadSample:
    def test_accepts_wav(self, seeded_client: TestClient) -> None:
        session = _create_session(seeded_client)
        sid = session["session_id"]
        wav = _make_wav_bytes()
        resp = seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert data["quality_report"]["passed"] is True

    def test_validates_quality(self, seeded_client: TestClient) -> None:
        session = _create_session(seeded_client)
        sid = session["session_id"]
        wav = _make_wav_bytes()
        resp = seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        qr = resp.json()["quality_report"]
        assert "snr_db" in qr
        assert "rms_dbfs" in qr
        assert "speech_ratio" in qr

    def test_rejects_bad_audio(self, seeded_client: TestClient) -> None:
        session = _create_session(seeded_client)
        sid = session["session_id"]
        bad_wav = _make_bad_wav_bytes()
        resp = seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("bad.wav", bad_wav, "audio/wav")},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is False
        assert len(data["quality_report"]["rejection_reasons"]) > 0

    def test_returns_next_prompt(
        self, seeded_client: TestClient,
    ) -> None:
        session = _create_session(seeded_client, prompts=3)
        sid = session["session_id"]
        wav = _make_wav_bytes()
        resp = seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        data = resp.json()
        # With 3 prompts, after accepting 1 there should be a next
        assert data["next_prompt"] is not None

    def test_upload_to_completed_session_returns_409(
        self, seeded_client: TestClient,
    ) -> None:
        session = _create_session(seeded_client, prompts=1)
        sid = session["session_id"]
        # Upload one sample to exhaust prompts
        wav = _make_wav_bytes()
        seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        # Complete the session
        seeded_client.post(f"/enroll/sessions/{sid}/complete")
        # Try to upload again
        resp = seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        assert resp.status_code == 409


class TestCompleteSession:
    def test_registers_styles(self, seeded_client: TestClient) -> None:
        session = _create_session(seeded_client, prompts=1)
        sid = session["session_id"]
        wav = _make_wav_bytes()
        seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        resp = seeded_client.post(f"/enroll/sessions/{sid}/complete")
        assert resp.status_code == 200
        data = resp.json()
        assert "phonetic" in data["styles_registered"]

    def test_complete_already_completed_returns_409(
        self, seeded_client: TestClient,
    ) -> None:
        session = _create_session(seeded_client, prompts=1)
        sid = session["session_id"]
        seeded_client.post(f"/enroll/sessions/{sid}/complete")
        resp = seeded_client.post(f"/enroll/sessions/{sid}/complete")
        assert resp.status_code == 409


class TestDeleteSession:
    def test_removes_data(self, seeded_client: TestClient) -> None:
        session = _create_session(seeded_client)
        sid = session["session_id"]
        resp = seeded_client.delete(f"/enroll/sessions/{sid}")
        assert resp.status_code == 204
        # Verify gone
        resp = seeded_client.get(f"/enroll/sessions/{sid}")
        assert resp.status_code == 404

    def test_not_found_returns_404(
        self, seeded_client: TestClient,
    ) -> None:
        resp = seeded_client.delete("/enroll/sessions/nonexistent")
        assert resp.status_code == 404


class TestGetPrompts:
    def test_returns_list(self, seeded_client: TestClient) -> None:
        resp = seeded_client.get(
            "/enroll/prompts", params={"style": "phonetic", "count": 3},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["style"] == "phonetic"
        assert len(data["prompts"]) == 3
        assert "text" in data["prompts"][0]

    def test_invalid_style_returns_error(
        self, seeded_client: TestClient,
    ) -> None:
        resp = seeded_client.get(
            "/enroll/prompts", params={"style": "nonexistent"},
        )
        assert resp.status_code in (422, 500)


class TestGetNextPrompt:
    def test_adapts_to_coverage(
        self, seeded_client: TestClient,
    ) -> None:
        session = _create_session(seeded_client, prompts=3)
        sid = session["session_id"]
        # Upload a sample to change coverage
        wav = _make_wav_bytes()
        seeded_client.post(
            f"/enroll/sessions/{sid}/samples",
            files={"file": ("sample.wav", wav, "audio/wav")},
        )
        resp = seeded_client.get(
            "/enroll/prompts/next",
            params={"session_id": sid},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data is not None
        assert "text" in data

    def test_session_not_found_returns_404(
        self, seeded_client: TestClient,
    ) -> None:
        resp = seeded_client.get(
            "/enroll/prompts/next",
            params={"session_id": "nonexistent"},
        )
        assert resp.status_code == 404
