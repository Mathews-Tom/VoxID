from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class CachedDecision:
    style: str
    confidence: float
    tier: str
    scores_json: str  # JSON-encoded dict[str, float]
    cached_at: float  # Unix timestamp


class RouterCache:
    """SQLite-backed LRU cache for routing decisions."""

    def __init__(
        self,
        db_path: Path,
        ttl_seconds: int = 3600,
        max_entries: int = 10000,
    ) -> None:
        self._db_path = db_path
        self._ttl = ttl_seconds
        self._max_entries = max_entries
        self._conn: sqlite3.Connection | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _ensure_db(self) -> sqlite3.Connection:
        if self._conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("""
                CREATE TABLE IF NOT EXISTS decisions (
                    text_hash TEXT PRIMARY KEY,
                    style TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    tier TEXT NOT NULL,
                    scores_json TEXT NOT NULL,
                    cached_at REAL NOT NULL,
                    last_accessed REAL NOT NULL
                )
            """)
            conn.commit()
            self._conn = conn
        return self._conn

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()

    def _evict_if_needed(self, conn: sqlite3.Connection) -> None:
        row = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        count: int = row[0] if row else 0
        if count >= self._max_entries:
            evict_count = max(1, self._max_entries // 10)
            conn.execute(
                """
                DELETE FROM decisions WHERE text_hash IN (
                    SELECT text_hash FROM decisions
                    ORDER BY last_accessed ASC
                    LIMIT ?
                )
                """,
                (evict_count,),
            )
            conn.commit()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get(self, text: str) -> CachedDecision | None:
        """Look up a cached decision. Returns None on miss or expired entry."""
        conn = self._ensure_db()
        text_hash = self._hash(text)
        row = conn.execute(
            "SELECT style, confidence, tier, scores_json, cached_at "
            "FROM decisions WHERE text_hash = ?",
            (text_hash,),
        ).fetchone()

        if row is None:
            return None

        style, confidence, tier, scores_json, cached_at = row
        now = time.time()

        if now - cached_at > self._ttl:
            conn.execute(
                "DELETE FROM decisions WHERE text_hash = ?", (text_hash,)
            )
            conn.commit()
            return None

        conn.execute(
            "UPDATE decisions SET last_accessed = ? WHERE text_hash = ?",
            (now, text_hash),
        )
        conn.commit()

        return CachedDecision(
            style=style,
            confidence=confidence,
            tier=tier,
            scores_json=scores_json,
            cached_at=cached_at,
        )

    def put(
        self,
        text: str,
        style: str,
        confidence: float,
        tier: str,
        scores: dict[str, float],
    ) -> None:
        """Cache a routing decision. Evicts oldest entries when over max."""
        conn = self._ensure_db()
        self._evict_if_needed(conn)

        text_hash = self._hash(text)
        now = time.time()
        scores_json = json.dumps(scores)

        conn.execute(
            """
            INSERT OR REPLACE INTO decisions
                (text_hash, style, confidence, tier,
                 scores_json, cached_at, last_accessed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (text_hash, style, confidence, tier, scores_json, now, now),
        )
        conn.commit()

    def invalidate(self, text: str | None = None) -> None:
        """Invalidate one entry by text, or all entries if text is None."""
        conn = self._ensure_db()
        if text is None:
            conn.execute("DELETE FROM decisions")
        else:
            conn.execute(
                "DELETE FROM decisions WHERE text_hash = ?",
                (self._hash(text),),
            )
        conn.commit()

    def stats(self) -> dict[str, int]:
        """Return cache statistics."""
        conn = self._ensure_db()
        now = time.time()

        total_row = conn.execute("SELECT COUNT(*) FROM decisions").fetchone()
        total: int = total_row[0] if total_row else 0

        expired_row = conn.execute(
            "SELECT COUNT(*) FROM decisions WHERE ? - cached_at > ?",
            (now, self._ttl),
        ).fetchone()
        expired: int = expired_row[0] if expired_row else 0

        return {
            "total": total,
            "expired": expired,
            "active": total - expired,
        }

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
