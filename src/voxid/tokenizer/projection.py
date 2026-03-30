from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from voxid.serialization import load_prompt, save_prompt


class EngineProjector:
    """Projects unified speaker embeddings to engine-specific embeddings.

    Uses per-engine linear least-squares projection matrices stored
    as SafeTensors in the projections directory. Each engine has a
    matrix W and bias b such that:

        engine_embedding = unified_embedding @ W + b

    Train with paired (unified, engine) embedding examples via
    ``fit()``, then persist with ``save()`` / ``load()``.
    """

    def __init__(self, projections_dir: Path) -> None:
        self._projections_dir = projections_dir
        self._matrices: dict[str, tuple[NDArray[np.float32], NDArray[np.float32]]] = {}

    def _matrix_path(self, engine: str) -> Path:
        return self._projections_dir / f"{engine}.safetensors"

    def fit(
        self,
        engine: str,
        unified_embeddings: NDArray[np.float32],
        engine_embeddings: NDArray[np.float32],
    ) -> None:
        """Fit a linear projection from unified to engine embeddings.

        Uses ordinary least-squares: min ||X @ W + b - Y||^2
        where X = unified_embeddings, Y = engine_embeddings.

        Args:
            engine: Engine name (e.g. "qwen3-tts").
            unified_embeddings: Shape (N, unified_dim).
            engine_embeddings: Shape (N, engine_dim).
        """
        if unified_embeddings.shape[0] != engine_embeddings.shape[0]:
            raise ValueError(
                f"Sample count mismatch: {unified_embeddings.shape[0]} "
                f"vs {engine_embeddings.shape[0]}"
            )
        if unified_embeddings.shape[0] < 2:
            raise ValueError("Need at least 2 samples to fit projection")

        n = unified_embeddings.shape[0]
        # Augment with bias column
        x_aug = np.column_stack([unified_embeddings, np.ones(n)])

        # Least-squares solution: (X^T X)^-1 X^T Y
        solution, _, _, _ = np.linalg.lstsq(x_aug, engine_embeddings, rcond=None)

        # Split into W and b
        w = solution[:-1].astype(np.float32)
        b = solution[-1].astype(np.float32)

        self._matrices[engine] = (w, b)

    def project(
        self,
        engine: str,
        unified_embedding: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Project a unified embedding to an engine-specific one.

        Loads the projection matrix on first use if not already cached.
        """
        if engine not in self._matrices:
            self._load_matrix(engine)

        w, b = self._matrices[engine]

        if unified_embedding.ndim == 1:
            return (unified_embedding @ w + b).astype(np.float32)
        return (unified_embedding @ w + b).astype(np.float32)

    def _load_matrix(self, engine: str) -> None:
        """Load projection matrix from disk."""
        path = self._matrix_path(engine)
        if not path.exists():
            raise FileNotFoundError(
                f"No projection matrix for engine {engine!r} at {path}"
            )
        tensors, _ = load_prompt(path)
        self._matrices[engine] = (
            tensors["weight"].astype(np.float32),
            tensors["bias"].astype(np.float32),
        )

    def save(self, engine: str) -> Path:
        """Save a fitted projection matrix to disk."""
        if engine not in self._matrices:
            raise RuntimeError(
                f"No projection fitted for engine {engine!r}. Call fit() first."
            )
        w, b = self._matrices[engine]
        path = self._matrix_path(engine)
        save_prompt(
            {"weight": w, "bias": b},
            path,
            metadata={"engine": engine},
        )
        return path

    def load(self, engine: str) -> None:
        """Explicitly load a projection matrix from disk."""
        self._load_matrix(engine)

    def has_projection(self, engine: str) -> bool:
        """Check if a projection exists (in memory or on disk)."""
        if engine in self._matrices:
            return True
        return self._matrix_path(engine).exists()

    def roundtrip_similarity(
        self,
        engine: str,
        unified_embedding: NDArray[np.float32],
        engine_embedding: NDArray[np.float32],
    ) -> float:
        """Compute cosine similarity between projected and actual engine embedding."""
        projected = self.project(engine, unified_embedding)
        norm_p = float(np.linalg.norm(projected))
        norm_e = float(np.linalg.norm(engine_embedding))
        if norm_p == 0.0 or norm_e == 0.0:
            return 0.0
        return float(np.dot(projected, engine_embedding) / (norm_p * norm_e))
