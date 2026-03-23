from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxid.security.drift import DriftReport, check_drift, cosine_similarity
from voxid.serialization import save_prompt
from voxid.versioning import VersionTracker


@pytest.fixture
def style_dir_with_versions(tmp_path: Path) -> Path:
    style_dir = tmp_path / "style"
    style_dir.mkdir()
    tracker = VersionTracker(style_dir)

    rng = np.random.default_rng(42)

    # Version 1: baseline embedding
    emb1 = rng.standard_normal(192).astype(np.float32)
    path1 = style_dir / "v1.safetensors"
    save_prompt({"ref_spk_embedding": emb1}, path1)
    tracker.add_version("model-v1", str(path1))

    # Version 2: slightly different
    emb2 = emb1 + rng.standard_normal(192).astype(np.float32) * 0.1
    path2 = style_dir / "v2.safetensors"
    save_prompt({"ref_spk_embedding": emb2}, path2)
    tracker.add_version("model-v1", str(path2), similarity_to_previous=0.95)

    return style_dir


@pytest.fixture
def style_dir_dissimilar(tmp_path: Path) -> Path:
    style_dir = tmp_path / "style_dissimilar"
    style_dir.mkdir()
    tracker = VersionTracker(style_dir)

    rng = np.random.default_rng(0)

    emb1 = rng.standard_normal(192).astype(np.float32)
    path1 = style_dir / "v1.safetensors"
    save_prompt({"ref_spk_embedding": emb1}, path1)
    tracker.add_version("model-v1", str(path1))

    # Opposite direction → very low cosine similarity
    emb2 = -emb1 + rng.standard_normal(192).astype(np.float32) * 0.05
    path2 = style_dir / "v2.safetensors"
    save_prompt({"ref_spk_embedding": emb2}, path2)
    tracker.add_version("model-v1", str(path2), similarity_to_previous=0.1)

    return style_dir


def test_cosine_similarity_identical_vectors() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, v) == pytest.approx(1.0, abs=1e-6)


def test_cosine_similarity_orthogonal_vectors() -> None:
    a = np.array([1.0, 0.0], dtype=np.float32)
    b = np.array([0.0, 1.0], dtype=np.float32)
    assert cosine_similarity(a, b) == pytest.approx(0.0, abs=1e-6)


def test_cosine_similarity_opposite_vectors() -> None:
    v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(v, -v) == pytest.approx(-1.0, abs=1e-6)


def test_cosine_similarity_zero_vector() -> None:
    a = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([1.0, 2.0, 3.0], dtype=np.float32)
    assert cosine_similarity(a, b) == 0.0


def test_check_drift_single_version_no_drift(tmp_path: Path) -> None:
    style_dir = tmp_path / "single"
    style_dir.mkdir()
    tracker = VersionTracker(style_dir)

    emb = np.ones(192, dtype=np.float32)
    path = style_dir / "v1.safetensors"
    save_prompt({"ref_spk_embedding": emb}, path)
    tracker.add_version("model-v1", str(path))

    report = check_drift(style_dir, "id-1", "style-1")

    assert isinstance(report, DriftReport)
    assert report.current_similarity == pytest.approx(1.0)
    assert report.below_threshold is False


def test_check_drift_two_identical_embeddings(tmp_path: Path) -> None:
    style_dir = tmp_path / "identical"
    style_dir.mkdir()
    tracker = VersionTracker(style_dir)

    emb = np.ones(192, dtype=np.float32)
    path1 = style_dir / "v1.safetensors"
    path2 = style_dir / "v2.safetensors"
    save_prompt({"ref_spk_embedding": emb}, path1)
    save_prompt({"ref_spk_embedding": emb}, path2)
    tracker.add_version("model-v1", str(path1))
    tracker.add_version("model-v1", str(path2), similarity_to_previous=1.0)

    report = check_drift(style_dir, "id-1", "style-1")

    assert report.current_similarity == pytest.approx(1.0, abs=1e-5)
    assert report.below_threshold is False


def test_check_drift_dissimilar_embeddings_flags(
    style_dir_dissimilar: Path,
) -> None:
    report = check_drift(
        style_dir_dissimilar,
        "id-dissimilar",
        "style-dissimilar",
        threshold=0.75,
    )
    assert report.below_threshold is True
    assert len(report.recommendation) > 0


def test_check_drift_threshold_configurable(tmp_path: Path) -> None:
    # Build two embeddings with cosine similarity ~0.95 so threshold=0.99 flags them
    style_dir = tmp_path / "strict"
    style_dir.mkdir()
    tracker = VersionTracker(style_dir)

    rng = np.random.default_rng(7)
    emb1 = rng.standard_normal(192).astype(np.float32)
    # Add noise large enough to push similarity below 0.99
    emb2 = emb1 + rng.standard_normal(192).astype(np.float32) * 0.5
    path1 = style_dir / "v1.safetensors"
    path2 = style_dir / "v2.safetensors"
    save_prompt({"ref_spk_embedding": emb1}, path1)
    save_prompt({"ref_spk_embedding": emb2}, path2)
    tracker.add_version("model-v1", str(path1))
    tracker.add_version("model-v1", str(path2), similarity_to_previous=0.95)

    report = check_drift(style_dir, "id-strict", "style-strict", threshold=0.99)
    assert report.threshold == 0.99
    assert report.below_threshold is True


def test_check_drift_missing_files_graceful(tmp_path: Path) -> None:
    style_dir = tmp_path / "missing"
    style_dir.mkdir()
    tracker = VersionTracker(style_dir)

    tracker.add_version("model-v1", str(style_dir / "ghost1.safetensors"))
    tracker.add_version("model-v1", str(style_dir / "ghost2.safetensors"))

    # Must not raise
    report = check_drift(style_dir, "id-ghost", "style-ghost")
    assert isinstance(report, DriftReport)
