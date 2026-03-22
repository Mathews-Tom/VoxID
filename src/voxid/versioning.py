from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import cast


@dataclass(frozen=True)
class EmbeddingVersion:
    version: int
    model_id: str
    timestamp: float  # Unix timestamp
    embedding_path: str
    similarity_to_previous: float | None = None


class VersionTracker:
    """Track embedding version history per style.

    Stores version metadata in a versions.json file
    alongside the style's embedding files. Retains
    the last N versions (configurable, default 3).
    """

    def __init__(
        self,
        style_dir: Path,
        max_versions: int = 3,
    ) -> None:
        self._style_dir = style_dir
        self._max_versions = max_versions
        self._versions_file = style_dir / "versions.json"

    def _load_versions(self) -> list[dict[str, object]]:
        if not self._versions_file.exists():
            return []
        return json.loads(  # type: ignore[no-any-return]
            self._versions_file.read_text(encoding="utf-8"),
        )

    def _save_versions(
        self,
        versions: list[dict[str, object]],
    ) -> None:
        self._versions_file.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        self._versions_file.write_text(
            json.dumps(versions, indent=2),
            encoding="utf-8",
        )

    def add_version(
        self,
        model_id: str,
        embedding_path: str,
        similarity_to_previous: float | None = None,
    ) -> EmbeddingVersion:
        """Record a new embedding version.

        Trims older versions beyond max_versions.
        """
        versions = self._load_versions()
        next_version = (
            max(
                (cast(int, v["version"]) for v in versions), default=0
            )
            + 1
        )

        record: dict[str, object] = {
            "version": next_version,
            "model_id": model_id,
            "timestamp": time.time(),
            "embedding_path": embedding_path,
            "similarity_to_previous": similarity_to_previous,
        }
        versions.append(record)

        # Trim to max_versions (keep newest)
        if len(versions) > self._max_versions:
            versions = versions[-self._max_versions :]

        self._save_versions(versions)

        return EmbeddingVersion(
            version=next_version,
            model_id=model_id,
            timestamp=cast(float, record["timestamp"]),
            embedding_path=embedding_path,
            similarity_to_previous=similarity_to_previous,
        )

    def list_versions(self) -> list[EmbeddingVersion]:
        """Return all stored versions, oldest first."""
        versions = self._load_versions()
        return [
            EmbeddingVersion(
                version=cast(int, v["version"]),
                model_id=str(v["model_id"]),
                timestamp=cast(float, v["timestamp"]),
                embedding_path=str(v["embedding_path"]),
                similarity_to_previous=(
                    cast(float, v["similarity_to_previous"])
                    if v.get("similarity_to_previous") is not None
                    else None
                ),
            )
            for v in versions
        ]

    def get_latest(self) -> EmbeddingVersion | None:
        """Return the most recent version, or None."""
        versions = self.list_versions()
        return versions[-1] if versions else None

    def get_version(self, version: int) -> EmbeddingVersion | None:
        """Return a specific version by number."""
        for v in self.list_versions():
            if v.version == version:
                return v
        return None
