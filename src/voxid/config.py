from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import tomli


def default_store_path() -> Path:
    path = Path.home() / ".voxid"
    path.mkdir(parents=True, exist_ok=True)
    return path


@dataclass
class VoxIDConfig:
    store_path: Path = field(default_factory=default_store_path)
    default_engine: str = "qwen3-tts"
    router_confidence_threshold: float = 0.8
    cache_ttl_seconds: int = 3600
    max_embedding_versions: int = 3


def load_config(config_path: Path | None = None) -> VoxIDConfig:
    if config_path is None:
        config_path = Path.home() / ".voxid" / "config.toml"

    if not config_path.exists():
        return VoxIDConfig()

    with config_path.open("rb") as f:
        raw = tomli.load(f)

    store_path = (
        Path(raw["store_path"])
        if "store_path" in raw
        else default_store_path()
    )

    return VoxIDConfig(
        store_path=store_path,
        default_engine=raw.get("default_engine", "qwen3-tts"),
        router_confidence_threshold=float(raw.get("router_confidence_threshold", 0.8)),
        cache_ttl_seconds=int(raw.get("cache_ttl_seconds", 3600)),
        max_embedding_versions=int(raw.get("max_embedding_versions", 3)),
    )
