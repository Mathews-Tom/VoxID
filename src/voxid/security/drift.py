from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from voxid.serialization import load_prompt
from voxid.versioning import VersionTracker


@dataclass(frozen=True)
class DriftReport:
    identity_id: str
    style_id: str
    current_similarity: float  # cosine sim to baseline
    rolling_average: float  # rolling avg over recent versions
    below_threshold: bool
    threshold: float
    recommendation: str  # "" or "re-enrollment recommended"
    version_count: int


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = float(np.linalg.norm(a))
    norm_b = float(np.linalg.norm(b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def check_drift(
    style_dir: Path,
    identity_id: str,
    style_id: str,
    threshold: float = 0.75,
) -> DriftReport:
    """Check voice drift for a style by comparing embeddings.

    Reads the version history and computes cosine similarity
    between the latest embedding and the baseline (version 1).

    Returns DriftReport with recommendation if below threshold.
    """
    tracker = VersionTracker(style_dir)
    versions = tracker.list_versions()

    if len(versions) < 2:
        return DriftReport(
            identity_id=identity_id,
            style_id=style_id,
            current_similarity=1.0,
            rolling_average=1.0,
            below_threshold=False,
            threshold=threshold,
            recommendation="",
            version_count=len(versions),
        )

    # Load baseline and latest embeddings
    baseline = versions[0]
    latest = versions[-1]

    try:
        baseline_tensors, _ = load_prompt(
            Path(baseline.embedding_path),
        )
        latest_tensors, _ = load_prompt(
            Path(latest.embedding_path),
        )
    except (FileNotFoundError, KeyError):
        return DriftReport(
            identity_id=identity_id,
            style_id=style_id,
            current_similarity=1.0,
            rolling_average=1.0,
            below_threshold=False,
            threshold=threshold,
            recommendation="embedding files not found",
            version_count=len(versions),
        )

    # Use ref_spk_embedding if available, else first tensor
    baseline_emb = baseline_tensors.get(
        "ref_spk_embedding",
        next(iter(baseline_tensors.values())),
    )
    latest_emb = latest_tensors.get(
        "ref_spk_embedding",
        next(iter(latest_tensors.values())),
    )

    similarity = cosine_similarity(
        baseline_emb.flatten(),
        latest_emb.flatten(),
    )

    # Rolling average from version similarities
    sims = [
        v.similarity_to_previous
        for v in versions
        if v.similarity_to_previous is not None
    ]
    rolling = sum(sims) / len(sims) if sims else similarity

    below = similarity < threshold
    recommendation = (
        f"Re-enrollment recommended for "
        f"{identity_id}:{style_id} — "
        f"similarity {similarity:.3f} < {threshold}"
        if below
        else ""
    )

    return DriftReport(
        identity_id=identity_id,
        style_id=style_id,
        current_similarity=similarity,
        rolling_average=rolling,
        below_threshold=below,
        threshold=threshold,
        recommendation=recommendation,
        version_count=len(versions),
    )
