from __future__ import annotations

import json
import zipfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf  # type: ignore[import-untyped]

import voxid.adapters.stub  # noqa: F401
from voxid.archive import ArchiveExporter, ArchiveImporter
from voxid.models import ConsentRecord, Identity, Style
from voxid.store import VoicePromptStore


@pytest.fixture
def store_with_identity(tmp_path: Path) -> VoicePromptStore:
    """Store with an identity and 2 styles."""
    store = VoicePromptStore(tmp_path / "voxid")

    consent = ConsentRecord(
        timestamp="2026-03-22T00:00:00Z",
        scope="personal",
        jurisdiction="US",
        transferable=True,
        document_hash="abc",
    )
    identity = Identity(
        id="tom",
        name="Tom",
        description="Test",
        default_style="conversational",
        created_at="2026-03-22T00:00:00Z",
        metadata={},
        consent_record=consent,
    )
    store.create_identity(identity)

    audio = np.zeros(24000, dtype=np.float32)
    ref = tmp_path / "ref.wav"
    sf.write(str(ref), audio, 24000)

    for sid in ["conversational", "technical"]:
        style = Style(
            id=sid,
            identity_id="tom",
            label=sid.title(),
            description=f"{sid} style",
            default_engine="stub",
            ref_audio_path=str(ref),
            ref_text=f"Ref for {sid}",
            language="en-US",
            metadata={},
        )
        store.add_style(style, ref)

    return store


def test_export_creates_archive_file(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    assert out.exists()


def test_export_archive_is_valid_zip(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    assert zipfile.is_zipfile(out)


def test_export_contains_manifest(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    with zipfile.ZipFile(out, "r") as zf:
        assert "manifest.json" in zf.namelist()


def test_export_contains_identity_toml(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    with zipfile.ZipFile(out, "r") as zf:
        assert "identity.toml" in zf.namelist()


def test_export_contains_consent_json(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    with zipfile.ZipFile(out, "r") as zf:
        assert "consent.json" in zf.namelist()


def test_export_contains_style_files(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    with zipfile.ZipFile(out, "r") as zf:
        names = zf.namelist()
        assert "styles/conversational/style.toml" in names
        assert any(
            n.startswith("styles/conversational/ref_audio") for n in names
        )
        assert "styles/conversational/ref_text.txt" in names


def test_export_no_engine_prompts(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"

    exporter.export("tom", out)

    with zipfile.ZipFile(out, "r") as zf:
        safetensors = [n for n in zf.namelist() if n.endswith(".safetensors")]
        assert safetensors == []


def test_export_with_hmac(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    key = b"secret-signing-key"

    exporter.export("tom", out, signing_key=key)

    with zipfile.ZipFile(out, "r") as zf:
        manifest = json.loads(zf.read("manifest.json"))
        assert "hmac" in manifest


def test_import_creates_identity(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    exporter.export("tom", out)

    clean_store = VoicePromptStore(tmp_path / "import_store")
    importer = ArchiveImporter(clean_store)
    importer.import_archive(out)

    assert "tom" in clean_store.list_identities()


def test_import_creates_styles(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    exporter.export("tom", out)

    clean_store = VoicePromptStore(tmp_path / "import_store")
    importer = ArchiveImporter(clean_store)
    importer.import_archive(out)

    styles = clean_store.list_styles("tom")
    assert len(styles) == 2
    assert "conversational" in styles
    assert "technical" in styles


def test_import_roundtrip_identity_fields(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    exporter.export("tom", out)

    clean_store = VoicePromptStore(tmp_path / "import_store")
    importer = ArchiveImporter(clean_store)
    importer.import_archive(out)

    original = store_with_identity.get_identity("tom")
    imported = clean_store.get_identity("tom")

    assert imported.id == original.id
    assert imported.name == original.name
    assert imported.default_style == original.default_style
    assert imported.created_at == original.created_at


def test_import_hmac_verification_passes(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    key = b"shared-secret"
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    exporter.export("tom", out, signing_key=key)

    clean_store = VoicePromptStore(tmp_path / "import_store")
    importer = ArchiveImporter(clean_store)

    identity = importer.import_archive(out, signing_key=key)
    assert identity.id == "tom"


def test_import_hmac_verification_fails(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    key_a = b"key-alpha"
    key_b = b"key-beta"
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    exporter.export("tom", out, signing_key=key_a)

    clean_store = VoicePromptStore(tmp_path / "import_store")
    importer = ArchiveImporter(clean_store)

    with pytest.raises(ValueError, match="HMAC verification failed"):
        importer.import_archive(out, signing_key=key_b)


def test_import_tampered_archive_fails(
    store_with_identity: VoicePromptStore, tmp_path: Path
) -> None:
    exporter = ArchiveExporter(store_with_identity)
    out = tmp_path / "tom.voxid"
    exporter.export("tom", out)

    # Tamper: corrupt the hash of identity.toml in the manifest
    tampered = tmp_path / "tom_tampered.voxid"
    with zipfile.ZipFile(out, "r") as zf_in, zipfile.ZipFile(
        tampered, "w", zipfile.ZIP_DEFLATED
    ) as zf_out:
        for item in zf_in.infolist():
            data = zf_in.read(item.filename)
            if item.filename == "manifest.json":
                manifest = json.loads(data)
                manifest["file_hashes"]["identity.toml"] = "deadbeef" * 8
                data = json.dumps(manifest, indent=2).encode()
            zf_out.writestr(item, data)

    clean_store = VoicePromptStore(tmp_path / "import_store")
    importer = ArchiveImporter(clean_store)

    with pytest.raises(ValueError, match="Hash mismatch"):
        importer.import_archive(tampered)
