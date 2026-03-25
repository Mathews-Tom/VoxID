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
def auth_client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("VOXID_API_KEY", "test-secret-key")
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    vox = VoxID(config=config)
    app = create_app()
    app.dependency_overrides[get_voxid] = lambda: vox
    return TestClient(app)


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


# ── Auth tests ────────────────────────────────────────────────────────────────


def test_auth_valid_key_returns_200(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/api/identities",
        json={"id": "alice", "name": "Alice", "default_style": "conversational"},
        headers={"X-API-Key": "test-secret-key"},
    )

    assert response.status_code == 201


def test_auth_missing_key_returns_401(auth_client: TestClient) -> None:
    # APIKeyMiddleware raises HTTPException(401) directly inside
    # BaseHTTPMiddleware.dispatch; with raise_server_exceptions=True (default),
    # TestClient propagates it as a Python exception.
    with pytest.raises(HTTPException) as exc_info:
        auth_client.post(
            "/api/identities",
            json={"id": "alice", "name": "Alice", "default_style": "conversational"},
        )

    assert exc_info.value.status_code == 401


def test_auth_wrong_key_returns_401(auth_client: TestClient) -> None:
    with pytest.raises(HTTPException) as exc_info:
        auth_client.post(
            "/api/identities",
            json={"id": "alice", "name": "Alice", "default_style": "conversational"},
            headers={"X-API-Key": "wrong-key"},
        )

    assert exc_info.value.status_code == 401


def test_auth_key_in_query_param(auth_client: TestClient) -> None:
    response = auth_client.post(
        "/api/identities?api_key=test-secret-key",
        json={"id": "bob", "name": "Bob", "default_style": "conversational"},
    )

    assert response.status_code == 201


def test_auth_health_exempt(auth_client: TestClient) -> None:
    response = auth_client.get("/api/health")

    assert response.status_code == 200


def test_auth_docs_exempt(auth_client: TestClient) -> None:
    response = auth_client.get("/docs")

    assert response.status_code == 200
