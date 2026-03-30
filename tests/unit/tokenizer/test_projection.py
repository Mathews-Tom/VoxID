from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from voxid.tokenizer.projection import EngineProjector


@pytest.fixture
def projections_dir(tmp_path: Path) -> Path:
    d = tmp_path / "projections"
    d.mkdir()
    return d


@pytest.fixture
def projector(projections_dir: Path) -> EngineProjector:
    return EngineProjector(projections_dir)


class TestEngineProjectorFit:
    def test_fit_identity_projection(self, projector: EngineProjector) -> None:
        """Fitting with Y ≈ X should produce near-identity projection."""
        rng = np.random.default_rng(42)
        n, dim = 20, 64
        unified = rng.standard_normal((n, dim)).astype(np.float32)
        engine = unified.copy()

        projector.fit("test-engine", unified, engine)
        projected = projector.project("test-engine", unified[0])

        similarity = float(
            np.dot(projected, engine[0])
            / (np.linalg.norm(projected) * np.linalg.norm(engine[0]))
        )
        assert similarity > 0.99

    def test_fit_dimension_change(self, projector: EngineProjector) -> None:
        """Projection can map between different dimensions."""
        rng = np.random.default_rng(42)
        n = 30
        unified = rng.standard_normal((n, 64)).astype(np.float32)
        engine = rng.standard_normal((n, 128)).astype(np.float32)

        projector.fit("wide-engine", unified, engine)
        projected = projector.project("wide-engine", unified[0])

        assert projected.shape == (128,)
        assert projected.dtype == np.float32

    def test_fit_sample_mismatch_raises(self, projector: EngineProjector) -> None:
        unified = np.ones((5, 64), dtype=np.float32)
        engine = np.ones((3, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="Sample count mismatch"):
            projector.fit("bad", unified, engine)

    def test_fit_too_few_samples_raises(self, projector: EngineProjector) -> None:
        unified = np.ones((1, 64), dtype=np.float32)
        engine = np.ones((1, 64), dtype=np.float32)

        with pytest.raises(ValueError, match="at least 2 samples"):
            projector.fit("tiny", unified, engine)


class TestEngineProjectorPersistence:
    def test_save_and_load(self, projector: EngineProjector) -> None:
        rng = np.random.default_rng(42)
        n, dim = 20, 64
        unified = rng.standard_normal((n, dim)).astype(np.float32)
        engine = unified.copy()

        projector.fit("persist-engine", unified, engine)
        projector.save("persist-engine")

        # Create a fresh projector pointing to the same dir
        fresh = EngineProjector(projector._projections_dir)
        projected = fresh.project("persist-engine", unified[0])

        similarity = float(
            np.dot(projected, engine[0])
            / (np.linalg.norm(projected) * np.linalg.norm(engine[0]))
        )
        assert similarity > 0.99

    def test_save_unfitted_raises(self, projector: EngineProjector) -> None:
        with pytest.raises(RuntimeError, match="No projection fitted"):
            projector.save("nonexistent")

    def test_load_missing_raises(self, projector: EngineProjector) -> None:
        with pytest.raises(FileNotFoundError, match="No projection matrix"):
            projector.project("ghost-engine", np.ones(64, dtype=np.float32))

    def test_has_projection(self, projector: EngineProjector) -> None:
        rng = np.random.default_rng(42)
        unified = rng.standard_normal((5, 64)).astype(np.float32)
        engine = rng.standard_normal((5, 64)).astype(np.float32)

        assert not projector.has_projection("test-engine")

        projector.fit("test-engine", unified, engine)
        assert projector.has_projection("test-engine")

        projector.save("test-engine")
        fresh = EngineProjector(projector._projections_dir)
        assert fresh.has_projection("test-engine")


class TestRoundtripSimilarity:
    def test_perfect_roundtrip(self, projector: EngineProjector) -> None:
        rng = np.random.default_rng(42)
        n, dim = 50, 64
        unified = rng.standard_normal((n, dim)).astype(np.float32)
        engine = unified.copy()

        projector.fit("rt-engine", unified, engine)

        # Roundtrip similarity on training data should be very high
        sim = projector.roundtrip_similarity("rt-engine", unified[0], engine[0])
        assert sim > 0.99

    def test_roundtrip_with_noise(self, projector: EngineProjector) -> None:
        rng = np.random.default_rng(42)
        n, dim = 50, 64
        unified = rng.standard_normal((n, dim)).astype(np.float32)
        noise = rng.standard_normal((n, dim)).astype(np.float32) * 0.1
        engine = unified + noise

        projector.fit("noisy", unified, engine)

        sim = projector.roundtrip_similarity("noisy", unified[0], engine[0])
        assert sim > 0.9

    def test_roundtrip_zero_actual_embedding(self, projector: EngineProjector) -> None:
        rng = np.random.default_rng(42)
        n, dim = 10, 64
        unified = rng.standard_normal((n, dim)).astype(np.float32)
        engine = rng.standard_normal((n, dim)).astype(np.float32)

        projector.fit("zero-test", unified, engine)

        # Zero actual embedding → zero norm → similarity = 0.0
        sim = projector.roundtrip_similarity(
            "zero-test",
            unified[0],
            np.zeros(dim, dtype=np.float32),
        )
        assert sim == 0.0
