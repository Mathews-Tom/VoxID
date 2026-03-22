from __future__ import annotations

from pathlib import Path

import pytest

from voxid.versioning import EmbeddingVersion, VersionTracker


@pytest.fixture
def style_dir(tmp_path: Path) -> Path:
    d = tmp_path / "style_dir"
    d.mkdir()
    return d


def test_add_version_creates_file(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")

    assert (style_dir / "versions.json").exists()


def test_add_version_returns_embedding_version(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir)

    result = tracker.add_version(
        model_id="stub",
        embedding_path="v1.safetensors",
    )

    assert isinstance(result, EmbeddingVersion)
    assert result.model_id == "stub"
    assert result.embedding_path == "v1.safetensors"
    assert result.version == 1


def test_list_versions_empty(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir)

    assert tracker.list_versions() == []


def test_list_versions_after_add(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=10)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v2.safetensors")

    assert len(tracker.list_versions()) == 2


def test_list_versions_ordered_oldest_first(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=10)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v2.safetensors")

    versions = tracker.list_versions()
    assert versions[0].version == 1
    assert versions[1].version == 2


def test_get_latest_returns_newest(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=10)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v2.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v3.safetensors")

    latest = tracker.get_latest()
    assert latest is not None
    assert latest.version == 3
    assert latest.embedding_path == "v3.safetensors"


def test_get_latest_empty_returns_none(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir)

    assert tracker.get_latest() is None


def test_get_version_by_number(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=10)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v2.safetensors")

    v1 = tracker.get_version(1)
    assert v1 is not None
    assert v1.version == 1
    assert v1.embedding_path == "v1.safetensors"


def test_get_version_not_found(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=10)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")

    assert tracker.get_version(999) is None


def test_max_versions_trims_oldest(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=2)

    tracker.add_version(model_id="stub", embedding_path="v1.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v2.safetensors")
    tracker.add_version(model_id="stub", embedding_path="v3.safetensors")

    versions = tracker.list_versions()
    assert len(versions) == 2
    version_numbers = [v.version for v in versions]
    assert 1 not in version_numbers
    assert 2 in version_numbers
    assert 3 in version_numbers


def test_similarity_to_previous_stored(style_dir: Path) -> None:
    tracker = VersionTracker(style_dir, max_versions=10)

    tracker.add_version(
        model_id="stub",
        embedding_path="v1.safetensors",
        similarity_to_previous=0.92,
    )

    latest = tracker.get_latest()
    assert latest is not None
    assert latest.similarity_to_previous == pytest.approx(0.92)


def test_version_persistence(style_dir: Path) -> None:
    tracker_a = VersionTracker(style_dir, max_versions=10)
    tracker_a.add_version(model_id="stub", embedding_path="v1.safetensors")
    tracker_a.add_version(model_id="stub", embedding_path="v2.safetensors")

    tracker_b = VersionTracker(style_dir, max_versions=10)
    versions = tracker_b.list_versions()

    assert len(versions) == 2
    assert versions[0].version == 1
    assert versions[1].version == 2
