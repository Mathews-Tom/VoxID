from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
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
def ref_audio(tmp_path: Path) -> Path:
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def seeded_client(client: TestClient, ref_audio: Path) -> TestClient:
    client.post(
        "/identities",
        json={
            "id": "tom",
            "name": "Tom",
            "default_style": "conversational",
        },
    )
    for sid in ["conversational", "technical", "narration", "emphatic"]:
        client.post(
            "/identities/tom/styles",
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


# ── Health ────────────────────────────────────────────────────────────────────


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "version" in body
    assert body["version"]


# ── Identity CRUD ─────────────────────────────────────────────────────────────


def test_create_identity_returns_201(client: TestClient) -> None:
    response = client.post(
        "/identities",
        json={"id": "alice", "name": "Alice", "default_style": "conversational"},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "alice"
    assert body["name"] == "Alice"
    assert body["default_style"] == "conversational"
    assert "created_at" in body


def test_create_identity_duplicate_returns_409(client: TestClient) -> None:
    payload = {"id": "bob", "name": "Bob", "default_style": "conversational"}
    client.post("/identities", json=payload)

    response = client.post("/identities", json=payload)

    assert response.status_code == 409


def test_list_identities_empty(client: TestClient) -> None:
    response = client.get("/identities")

    assert response.status_code == 200
    assert response.json()["identities"] == []


def test_list_identities_after_create(client: TestClient) -> None:
    client.post(
        "/identities",
        json={"id": "carol", "name": "Carol", "default_style": "conversational"},
    )
    client.post(
        "/identities",
        json={"id": "dave", "name": "Dave", "default_style": "conversational"},
    )

    response = client.get("/identities")

    assert response.status_code == 200
    identities = response.json()["identities"]
    assert "carol" in identities
    assert "dave" in identities


def test_get_identity_returns_200(client: TestClient) -> None:
    client.post(
        "/identities",
        json={"id": "tom", "name": "Tom", "default_style": "conversational"},
    )

    response = client.get("/identities/tom")

    assert response.status_code == 200
    body = response.json()
    assert body["id"] == "tom"
    assert body["name"] == "Tom"


def test_get_identity_not_found_returns_404(client: TestClient) -> None:
    response = client.get("/identities/missing")

    assert response.status_code == 404


def test_delete_identity_returns_204(client: TestClient) -> None:
    client.post(
        "/identities",
        json={"id": "eve", "name": "Eve", "default_style": "conversational"},
    )

    response = client.delete("/identities/eve")

    assert response.status_code == 204


def test_delete_identity_not_found_returns_404(client: TestClient) -> None:
    response = client.delete("/identities/missing")

    assert response.status_code == 404


# ── Style CRUD ────────────────────────────────────────────────────────────────


def test_add_style_returns_201(client: TestClient, ref_audio: Path) -> None:
    client.post(
        "/identities",
        json={"id": "tom", "name": "Tom", "default_style": "conversational"},
    )

    response = client.post(
        "/identities/tom/styles",
        json={
            "id": "conversational",
            "label": "Conversational",
            "description": "Relaxed, warm style",
            "ref_audio_path": str(ref_audio),
            "ref_text": "Hello, how are you?",
            "engine": "stub",
        },
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == "conversational"
    assert body["identity_id"] == "tom"


def test_add_style_identity_not_found_returns_404(
    client: TestClient, ref_audio: Path
) -> None:
    response = client.post(
        "/identities/nonexistent/styles",
        json={
            "id": "conversational",
            "label": "Conversational",
            "description": "A style",
            "ref_audio_path": str(ref_audio),
            "ref_text": "Hello there.",
            "engine": "stub",
        },
    )

    assert response.status_code == 404


def test_list_styles_returns_200(
    client: TestClient, ref_audio: Path
) -> None:
    client.post(
        "/identities",
        json={"id": "tom", "name": "Tom", "default_style": "conversational"},
    )
    client.post(
        "/identities/tom/styles",
        json={
            "id": "conversational",
            "label": "Conversational",
            "description": "Relaxed style",
            "ref_audio_path": str(ref_audio),
            "ref_text": "Hello.",
            "engine": "stub",
        },
    )

    response = client.get("/identities/tom/styles")

    assert response.status_code == 200
    body = response.json()
    assert "styles" in body
    assert "conversational" in body["styles"]


# ── Generate ──────────────────────────────────────────────────────────────────


def test_generate_returns_200_with_audio_path(
    seeded_client: TestClient,
) -> None:
    response = seeded_client.post(
        "/generate",
        json={"text": "Hello world.", "identity_id": "tom"},
    )

    assert response.status_code == 200
    body = response.json()
    assert "audio_path" in body
    assert body["audio_path"]
    assert body["identity_id"] == "tom"
    assert "sample_rate" in body


def test_generate_with_explicit_style(seeded_client: TestClient) -> None:
    response = seeded_client.post(
        "/generate",
        json={
            "text": "Deploy the new version.",
            "identity_id": "tom",
            "style": "technical",
        },
    )

    assert response.status_code == 200
    assert response.json()["style_used"] == "technical"


def test_generate_identity_not_found_returns_404(client: TestClient) -> None:
    response = client.post(
        "/generate",
        json={"text": "Hello.", "identity_id": "nobody"},
    )

    assert response.status_code == 404


def test_generate_segments_returns_200(seeded_client: TestClient) -> None:
    response = seeded_client.post(
        "/generate/segments",
        json={
            "text": "First sentence. Second sentence. Third sentence.",
            "identity_id": "tom",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "segments" in body
    assert len(body["segments"]) > 0


def test_generate_segments_stitched_path(seeded_client: TestClient) -> None:
    response = seeded_client.post(
        "/generate/segments",
        json={
            "text": "First sentence. Second sentence.",
            "identity_id": "tom",
            "stitch": True,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["stitched_path"] is not None


def test_generate_manifest_returns_200(seeded_client: TestClient) -> None:
    response = seeded_client.post(
        "/generate/manifest",
        json={
            "identity_id": "tom",
            "engine": "stub",
            "scenes": [
                {
                    "scene_id": "s1",
                    "text": "Welcome to the show.",
                    "style": "conversational",
                },
                {
                    "scene_id": "s2",
                    "text": "Today we discuss the topic.",
                    "style": "technical",
                },
            ],
            "metadata": {},
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "scenes" in body
    assert len(body["scenes"]) == 2
    assert "total_duration_ms" in body


def test_generate_stream_returns_sse(seeded_client: TestClient) -> None:
    response = seeded_client.post(
        "/generate/stream",
        json={
            "text": "Hello world. This is a test.",
            "identity_id": "tom",
        },
    )

    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]


# ── Route ─────────────────────────────────────────────────────────────────────


def test_route_returns_style_and_confidence(seeded_client: TestClient) -> None:
    response = seeded_client.post(
        "/route",
        json={
            "text": "Let me walk you through the deployment steps.",
            "identity_id": "tom",
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "style" in body
    assert "confidence" in body
    assert "tier" in body
    assert "scores" in body
    assert isinstance(body["confidence"], float)
    assert isinstance(body["scores"], dict)


# ── Concurrent ────────────────────────────────────────────────────────────────


def test_concurrent_generate_no_deadlock(
    tmp_path: Path, ref_audio: Path
) -> None:
    # Build a fresh, isolated VoxID instance for this test so no SQLite
    # connection is shared with any other fixture or test.
    config = VoxIDConfig(
        store_path=tmp_path / "concurrent_voxid",
        default_engine="stub",
    )
    vox = VoxID(config=config)
    app = create_app()
    app.dependency_overrides[get_voxid] = lambda: vox
    fresh_client = TestClient(app, raise_server_exceptions=False)

    fresh_client.post(
        "/identities",
        json={"id": "tom", "name": "Tom", "default_style": "conversational"},
    )
    for sid in ["conversational", "technical"]:
        fresh_client.post(
            "/identities/tom/styles",
            json={
                "id": sid,
                "label": sid.title(),
                "description": f"{sid} style",
                "ref_audio_path": str(ref_audio),
                "ref_text": f"Reference for {sid}",
                "engine": "stub",
            },
        )

    def make_request(i: int) -> int:
        response = fresh_client.post(
            "/generate",
            json={
                "text": f"Concurrent test sentence number {i}.",
                "identity_id": "tom",
            },
        )
        return response.status_code

    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = [executor.submit(make_request, i) for i in range(5)]
        results = [f.result() for f in as_completed(futures)]

    assert all(status == 200 for status in results)
