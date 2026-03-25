from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
from fastapi import HTTPException
from fastapi.testclient import TestClient

import voxid.adapters.stub  # noqa: F401
from voxid.api.app import create_app
from voxid.api.deps import get_voxid
from voxid.config import VoxIDConfig
from voxid.core import VoxID


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def limited_client(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, ref_audio: Path
) -> TestClient:
    monkeypatch.setenv("VOXID_RATE_LIMIT", "3")
    monkeypatch.setenv("VOXID_RATE_WINDOW", "60")

    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    vox = VoxID(config=config)
    app = create_app()
    app.dependency_overrides[get_voxid] = lambda: vox
    client = TestClient(app)

    # Seed identity and styles so generate requests succeed
    client.post(
        "/api/identities",
        json={"id": "tom", "name": "Tom", "default_style": "conversational"},
    )
    for sid in ["conversational", "technical"]:
        client.post(
            "/api/identities/tom/styles",
            json={
                "id": sid,
                "label": sid.title(),
                "description": f"{sid} style",
                "ref_audio_path": str(ref_audio),
                "ref_text": f"Reference for {sid}",
                "engine": "stub",
            },
        )

    return client


# ── Rate limit tests ──────────────────────────────────────────────────────────


def test_rate_limit_allows_under_threshold(limited_client: TestClient) -> None:
    statuses = []
    for i in range(3):
        response = limited_client.post(
            "/api/generate",
            json={"text": f"Request number {i}.", "identity_id": "tom"},
        )
        statuses.append(response.status_code)

    assert all(s == 200 for s in statuses)


def test_rate_limit_returns_429_over_threshold(limited_client: TestClient) -> None:
    for i in range(3):
        limited_client.post(
            "/api/generate",
            json={"text": f"Request number {i}.", "identity_id": "tom"},
        )

    # RateLimitMiddleware raises HTTPException(429) inside BaseHTTPMiddleware;
    # TestClient propagates it as a Python exception.
    with pytest.raises(HTTPException) as exc_info:
        limited_client.post(
            "/api/generate",
            json={"text": "One request too many.", "identity_id": "tom"},
        )

    assert exc_info.value.status_code == 429


def test_rate_limit_non_generate_not_limited(limited_client: TestClient) -> None:
    statuses = [limited_client.get("/api/health").status_code for _ in range(100)]

    assert all(s == 200 for s in statuses)


def test_rate_limit_retry_after_header(limited_client: TestClient) -> None:
    for i in range(3):
        limited_client.post(
            "/api/generate",
            json={"text": f"Request number {i}.", "identity_id": "tom"},
        )

    # HTTPException raised in middleware carries headers; verify Retry-After.
    with pytest.raises(HTTPException) as exc_info:
        limited_client.post(
            "/api/generate",
            json={"text": "Exceeds limit.", "identity_id": "tom"},
        )

    assert exc_info.value.status_code == 429
    headers = exc_info.value.headers or {}
    assert "Retry-After" in headers
    retry_after = int(headers["Retry-After"])
    assert retry_after >= 1
