from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import voxid.adapters.stub  # noqa: F401 — registers StubAdapter
from voxid.config import VoxIDConfig
from voxid.core import VoxID


@pytest.fixture
def vox(tmp_path: Path) -> VoxID:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    return VoxID(config=config)


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    """Create a minimal WAV for testing."""
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def identity_with_styles(vox: VoxID, ref_audio: Path) -> str:
    """Create an identity with 4 styles for routing tests."""
    vox.create_identity(id="tom", name="Tom")
    for style_id in ["conversational", "technical", "narration", "emphatic"]:
        vox.add_style(
            identity_id="tom",
            id=style_id,
            label=style_id.title(),
            description=f"{style_id} style",
            ref_audio=ref_audio,
            ref_text=f"Reference text for {style_id}",
            engine="stub",
        )
    return "tom"


# ── Identity tests ────────────────────────────────────────────────────────────


def test_create_identity_returns_identity(vox: VoxID) -> None:
    identity = vox.create_identity(id="alice", name="Alice")

    assert identity.id == "alice"
    assert identity.name == "Alice"
    assert identity.default_style == "conversational"
    assert identity.consent_record is not None


def test_create_identity_persists_to_store(vox: VoxID) -> None:
    vox.create_identity(id="bob", name="Bob")

    assert "bob" in vox.list_identities()


def test_list_identities_empty(vox: VoxID) -> None:
    assert vox.list_identities() == []


def test_list_identities_after_create(vox: VoxID) -> None:
    vox.create_identity(id="alice", name="Alice")
    vox.create_identity(id="bob", name="Bob")

    ids = vox.list_identities()
    assert "alice" in ids
    assert "bob" in ids
    assert len(ids) == 2


# ── Style tests ───────────────────────────────────────────────────────────────


def test_add_style_returns_style(vox: VoxID, ref_audio: Path) -> None:
    vox.create_identity(id="carol", name="Carol")

    style = vox.add_style(
        identity_id="carol",
        id="conversational",
        label="Conversational",
        description="Relaxed style",
        ref_audio=ref_audio,
        ref_text="Hello, how are you?",
        engine="stub",
    )

    assert style.id == "conversational"
    assert style.identity_id == "carol"
    assert style.default_engine == "stub"
    assert style.label == "Conversational"


def test_add_style_builds_prompt_cache(vox: VoxID, ref_audio: Path) -> None:
    vox.create_identity(id="dave", name="Dave")
    vox.add_style(
        identity_id="dave",
        id="conversational",
        label="Conversational",
        description="Relaxed style",
        ref_audio=ref_audio,
        ref_text="Hello world",
        engine="stub",
    )

    prompt_path = vox._store.get_prompt_path("dave", "conversational", "stub")
    assert prompt_path is not None
    assert prompt_path.exists()
    assert prompt_path.name == "stub.safetensors"


def test_list_styles_after_add(vox: VoxID, ref_audio: Path) -> None:
    vox.create_identity(id="eve", name="Eve")
    vox.add_style(
        identity_id="eve",
        id="conversational",
        label="Conversational",
        description="Relaxed",
        ref_audio=ref_audio,
        ref_text="Text one",
        engine="stub",
    )
    vox.add_style(
        identity_id="eve",
        id="technical",
        label="Technical",
        description="Precise",
        ref_audio=ref_audio,
        ref_text="Text two",
        engine="stub",
    )

    styles = vox.list_styles("eve")
    assert "conversational" in styles
    assert "technical" in styles
    assert len(styles) == 2


# ── Generate tests ────────────────────────────────────────────────────────────


def test_generate_explicit_style_produces_wav(
    vox: VoxID, ref_audio: Path
) -> None:
    vox.create_identity(id="frank", name="Frank")
    vox.add_style(
        identity_id="frank",
        id="conversational",
        label="Conversational",
        description="Relaxed",
        ref_audio=ref_audio,
        ref_text="Hello there",
        engine="stub",
    )

    output_path, sr = vox.generate(
        text="Hello world",
        identity_id="frank",
        style="conversational",
    )

    assert output_path.exists()
    assert output_path.suffix == ".wav"
    assert sr == 24000


def test_generate_auto_routes_when_style_none(
    identity_with_styles: str, vox: VoxID
) -> None:
    output_path, sr = vox.generate(
        text="Let me explain the technical architecture",
        identity_id=identity_with_styles,
        style=None,
    )

    assert output_path.exists()
    assert sr > 0


def test_generate_output_is_valid_wav(vox: VoxID, ref_audio: Path) -> None:
    vox.create_identity(id="grace", name="Grace")
    vox.add_style(
        identity_id="grace",
        id="conversational",
        label="Conversational",
        description="Relaxed",
        ref_audio=ref_audio,
        ref_text="Test reference",
        engine="stub",
    )

    output_path, _ = vox.generate(
        text="Testing output",
        identity_id="grace",
        style="conversational",
    )

    waveform, sample_rate = sf.read(str(output_path))
    assert sample_rate > 0
    assert len(waveform) > 0


# ── Prompt cache tests ────────────────────────────────────────────────────────


def test_ensure_prompt_creates_safetensors(vox: VoxID, ref_audio: Path) -> None:
    vox.create_identity(id="henry", name="Henry")
    vox.add_style(
        identity_id="henry",
        id="conversational",
        label="Conversational",
        description="Relaxed",
        ref_audio=ref_audio,
        ref_text="Test reference",
        engine="stub",
    )

    prompt_path = vox._store.get_prompt_path("henry", "conversational", "stub")
    assert prompt_path is not None
    assert prompt_path.suffix == ".safetensors"
    assert prompt_path.exists()


def test_prompt_cache_miss_triggers_rebuild(vox: VoxID, ref_audio: Path) -> None:
    vox.create_identity(id="ivan", name="Ivan")
    vox.add_style(
        identity_id="ivan",
        id="conversational",
        label="Conversational",
        description="Relaxed",
        ref_audio=ref_audio,
        ref_text="Test reference",
        engine="stub",
    )

    # Delete the cached prompt to simulate a cache miss
    prompt_path = vox._store.get_prompt_path("ivan", "conversational", "stub")
    assert prompt_path is not None
    prompt_path.unlink()
    assert not prompt_path.exists()

    # generate should trigger rebuild
    output_path, sr = vox.generate(
        text="Hello again",
        identity_id="ivan",
        style="conversational",
    )

    assert output_path.exists()
    rebuilt = vox._store.get_prompt_path("ivan", "conversational", "stub")
    assert rebuilt is not None
    assert rebuilt.exists()


# ── Route tests ───────────────────────────────────────────────────────────────


def test_route_returns_decision_with_scores(
    identity_with_styles: str, vox: VoxID
) -> None:
    result = vox.route(
        text="Hello, how are you doing today?",
        identity_id=identity_with_styles,
    )

    assert "style" in result
    assert "confidence" in result
    assert "tier" in result
    assert "scores" in result
    assert isinstance(result["confidence"], float)
    assert isinstance(result["scores"], dict)


def test_route_technical_text_returns_technical(
    identity_with_styles: str, vox: VoxID
) -> None:
    result = vox.route(
        text=(
            "The API endpoint uses OAuth2 with PKCE flow. "
            "The JWT access token expires after 15 minutes."
        ),
        identity_id=identity_with_styles,
    )

    assert result["style"] == "technical"
    assert result["confidence"] > 0.0


def test_route_conversational_text(
    identity_with_styles: str, vox: VoxID
) -> None:
    result = vox.route(
        text="Hey! So last week I was thinking about what we discussed.",
        identity_id=identity_with_styles,
    )

    assert result["style"] == "conversational"
    assert result["confidence"] > 0.0
