from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import voxid.adapters.stub  # noqa: F401
from voxid.config import VoxIDConfig
from voxid.core import VoxID
from voxid.plugins.voicebox.models import VoiceBoxProfile
from voxid.plugins.voicebox.sync import ProfileSync


@pytest.fixture
def vox(tmp_path: Path) -> VoxID:
    config = VoxIDConfig(
        store_path=tmp_path / "voxid",
        default_engine="stub",
    )
    return VoxID(config=config)


@pytest.fixture
def ref_audio(tmp_path: Path) -> Path:
    audio = np.zeros(24000, dtype=np.float32)
    path = tmp_path / "ref.wav"
    sf.write(str(path), audio, 24000)
    return path


@pytest.fixture
def seeded_vox(vox: VoxID, ref_audio: Path) -> VoxID:
    vox.create_identity(id="tom", name="Tom")
    for sid in ["conversational", "technical", "narration", "emphatic"]:
        vox.add_style(
            identity_id="tom",
            id=sid,
            label=sid.title(),
            description=f"{sid} style",
            ref_audio=ref_audio,
            ref_text=f"Ref for {sid}",
            engine="stub",
        )
    return vox


@pytest.fixture
def sync(vox: VoxID) -> ProfileSync:
    return ProfileSync(voxid=vox)


@pytest.fixture
def seeded_sync(seeded_vox: VoxID) -> ProfileSync:
    return ProfileSync(voxid=seeded_vox)


# ── import_from_voicebox ──────────────────────────────────────────────────────


def test_import_voicebox_profile_creates_identity(
    sync: ProfileSync,
    ref_audio: Path,
    vox: VoxID,
) -> None:
    profile = VoiceBoxProfile(
        name="Alice",
        audio_files=[str(ref_audio)],
    )

    sync.import_from_voicebox(profile)

    assert "alice" in vox.list_identities()


def test_import_voicebox_profile_creates_styles(
    sync: ProfileSync,
    ref_audio: Path,
    vox: VoxID,
    tmp_path: Path,
) -> None:
    second_audio = tmp_path / "ref2.wav"
    sf.write(str(second_audio), np.zeros(24000, dtype=np.float32), 24000)

    profile = VoiceBoxProfile(
        name="Bob",
        audio_files=[str(ref_audio), str(second_audio)],
    )

    identity = sync.import_from_voicebox(profile)

    styles = vox.list_styles(identity.id)
    assert len(styles) == 2


def test_import_voicebox_profile_metadata_preserved(
    sync: ProfileSync,
    ref_audio: Path,
    vox: VoxID,
) -> None:
    profile = VoiceBoxProfile(
        name="Carol",
        audio_files=[str(ref_audio)],
        tags=["narrator", "calm"],
    )

    identity = sync.import_from_voicebox(profile)

    stored = vox._store.get_identity(identity.id)
    assert stored.metadata["source"] == "voicebox"
    assert stored.metadata["tags"] == ["narrator", "calm"]


# ── export_to_voicebox ────────────────────────────────────────────────────────


def test_export_to_voicebox_profile(seeded_sync: ProfileSync) -> None:
    profile = seeded_sync.export_to_voicebox("tom")

    assert profile.name == "Tom"
    assert len(profile.audio_files) == 4


def test_export_to_voicebox_metadata(seeded_sync: ProfileSync) -> None:
    profile = seeded_sync.export_to_voicebox("tom")

    assert profile.metadata["voxid_identity_id"] == "tom"


# ── roundtrip ─────────────────────────────────────────────────────────────────


def test_roundtrip_import_export(
    sync: ProfileSync,
    ref_audio: Path,
    tmp_path: Path,
) -> None:
    second_audio = tmp_path / "ref2.wav"
    sf.write(str(second_audio), np.zeros(24000, dtype=np.float32), 24000)

    original_profile = VoiceBoxProfile(
        name="Diana",
        audio_files=[str(ref_audio), str(second_audio)],
    )

    identity = sync.import_from_voicebox(original_profile)
    exported = sync.export_to_voicebox(identity.id)

    assert exported.name == "Diana"
    assert len(exported.audio_files) == len(original_profile.audio_files)


def test_export_to_json_creates_file(
    seeded_sync: ProfileSync,
    tmp_path: Path,
) -> None:
    output_path = tmp_path / "exports" / "tom.json"

    seeded_sync.export_to_json("tom", output_path)

    assert output_path.exists()
    data = json.loads(output_path.read_text(encoding="utf-8"))
    assert data["name"] == "Tom"
    assert isinstance(data["audio_files"], list)


def test_import_from_json_roundtrip(
    seeded_sync: ProfileSync,
    tmp_path: Path,
    ref_audio: Path,
    vox: VoxID,
) -> None:
    # Export from the seeded vox instance
    json_path = tmp_path / "tom.json"
    seeded_sync.export_to_json("tom", json_path)

    # Import into a fresh vox instance
    fresh_config = VoxIDConfig(
        store_path=tmp_path / "voxid_fresh",
        default_engine="stub",
    )
    fresh_vox = VoxID(config=fresh_config)
    fresh_sync = ProfileSync(voxid=fresh_vox)

    identity = fresh_sync.import_from_json(json_path)

    assert identity.name == "Tom"
    assert identity.id in fresh_vox.list_identities()


# ── _slugify ─────────────────────────────────────────────────────────────────


def test_slugify_simple() -> None:
    result = ProfileSync._slugify("Tom Smith")

    assert result == "tom-smith"


def test_slugify_special_chars() -> None:
    result = ProfileSync._slugify("Tom's Voice!")

    assert result == "toms-voice"
