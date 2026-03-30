from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
import tomli


class DispatchStrategy(Enum):
    ROUND_ROBIN = "round_robin"
    LEAST_LOADED = "least_loaded"


@dataclass(frozen=True)
class WorkerConfig:
    """Configuration for a single TTS GPU worker."""

    engine: str
    device: str  # e.g. "cuda:0", "cuda:1"
    max_batch_size: int = 4
    max_queue_depth: int = 16


@dataclass(frozen=True)
class ServingConfig:
    """Top-level serving configuration for multi-GPU dispatch."""

    workers: list[WorkerConfig]
    dispatch_strategy: DispatchStrategy = DispatchStrategy.ROUND_ROBIN
    health_check_interval_s: float = 30.0


@dataclass(frozen=True)
class GenerationRequest:
    """A request dispatched to a TTSWorker."""

    request_id: str
    text: str
    prompt_path: Path
    engine: str
    language: str = "en"
    context_params: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class GenerationResult:
    """Result returned by a TTSWorker."""

    request_id: str
    waveform: np.ndarray[Any, np.dtype[np.floating[Any]]]
    sample_rate: int


def load_serving_config(config_path: Path) -> ServingConfig:
    """Load a ServingConfig from a TOML file.

    Expected format::

        dispatch_strategy = "round_robin"  # or "least_loaded"
        health_check_interval_s = 30.0

        [[workers]]
        engine = "qwen3-tts"
        device = "cuda:0"
        max_batch_size = 4
        max_queue_depth = 16

        [[workers]]
        engine = "fish-speech"
        device = "cuda:1"
    """
    with config_path.open("rb") as f:
        raw = tomli.load(f)

    workers: list[WorkerConfig] = []
    for w in raw.get("workers", []):
        workers.append(
            WorkerConfig(
                engine=w["engine"],
                device=w["device"],
                max_batch_size=int(w.get("max_batch_size", 4)),
                max_queue_depth=int(w.get("max_queue_depth", 16)),
            )
        )

    if not workers:
        msg = f"serving config at {config_path} defines no workers"
        raise ValueError(msg)

    strategy_str = raw.get("dispatch_strategy", "round_robin")
    strategy = DispatchStrategy(strategy_str)

    return ServingConfig(
        workers=workers,
        dispatch_strategy=strategy,
        health_check_interval_s=float(
            raw.get("health_check_interval_s", 30.0)
        ),
    )
