from __future__ import annotations

import json

import pytest

from voxid.router.cache import CachedDecision, RouterCache

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_scores() -> dict[str, float]:
    return {
        "conversational": 0.6,
        "technical": 0.3,
        "narration": 0.05,
        "emphatic": 0.05,
    }


# ---------------------------------------------------------------------------
# RouterCache tests
# ---------------------------------------------------------------------------


class TestRouterCache:
    def test_cache_put_and_get_roundtrip(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        cache = RouterCache(db_path=tmp_path / "cache.db")
        text = "The model returns logits for each token."
        scores = _make_scores()
        # Act
        cache.put(text, "technical", 0.91, "rule-based", scores)
        result = cache.get(text)
        # Assert
        assert result is not None
        assert isinstance(result, CachedDecision)
        assert result.style == "technical"
        assert result.confidence == pytest.approx(0.91)
        assert result.tier == "rule-based"
        assert json.loads(result.scores_json) == scores
        assert result.cached_at > 0.0

    def test_cache_miss_returns_none(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        cache = RouterCache(db_path=tmp_path / "cache.db")
        # Act
        result = cache.get("text that was never stored")
        # Assert
        assert result is None

    def test_cache_expired_entry_returns_none(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange — ttl_seconds=0 causes immediate expiry
        cache = RouterCache(db_path=tmp_path / "cache.db", ttl_seconds=0)
        text = "Some text to cache."
        cache.put(text, "conversational", 0.85, "centroid", _make_scores())
        # Act
        result = cache.get(text)
        # Assert
        assert result is None

    def test_cache_invalidate_single_entry(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        cache = RouterCache(db_path=tmp_path / "cache.db")
        text_a = "First text to cache."
        text_b = "Second text to cache."
        cache.put(text_a, "technical", 0.88, "rule-based", _make_scores())
        cache.put(text_b, "conversational", 0.72, "centroid", _make_scores())
        # Act
        cache.invalidate(text_a)
        # Assert
        assert cache.get(text_a) is None
        assert cache.get(text_b) is not None

    def test_cache_invalidate_all(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        cache = RouterCache(db_path=tmp_path / "cache.db")
        cache.put("text one", "technical", 0.9, "rule-based", _make_scores())
        cache.put("text two", "narration", 0.75, "centroid", _make_scores())
        # Act
        cache.invalidate()
        # Assert
        assert cache.get("text one") is None
        assert cache.get("text two") is None

    def test_cache_stats_counts_entries(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        cache = RouterCache(db_path=tmp_path / "cache.db")
        cache.put("entry one", "technical", 0.9, "rule-based", _make_scores())
        cache.put("entry two", "conversational", 0.8, "centroid", _make_scores())
        cache.put("entry three", "narration", 0.7, "centroid", _make_scores())
        # Act
        stats = cache.stats()
        # Assert
        assert stats["total"] == 3

    def test_cache_eviction_when_over_max(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        cache = RouterCache(db_path=tmp_path / "cache.db", max_entries=5)
        # Act
        for i in range(10):
            cache.put(
                f"unique text entry number {i}",
                "technical",
                0.9,
                "rule-based",
                _make_scores(),
            )
        stats = cache.stats()
        # Assert — eviction brings count back at or below max
        assert stats["total"] <= 5

    def test_cache_close_and_reopen(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        db_path = tmp_path / "cache.db"
        cache = RouterCache(db_path=db_path)
        text = "Persistent text entry."
        cache.put(text, "narration", 0.78, "centroid", _make_scores())
        # Act
        cache.close()
        reopened = RouterCache(db_path=db_path)
        result = reopened.get(text)
        # Assert
        assert result is not None
        assert result.style == "narration"
        assert result.confidence == pytest.approx(0.78)
