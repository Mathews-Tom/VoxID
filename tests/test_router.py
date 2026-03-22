from __future__ import annotations

import pytest

from voxid.router import RouteDecision, StyleRouter

_ALL_STYLES = ["conversational", "technical", "narration", "emphatic"]


class TestStyleRouter:
    def test_router_routes_technical_text(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = "The API endpoint returns a 429 when rate limited."
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert result.style == "technical"

    def test_router_routes_conversational_text(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = (
            "So I've been thinking about this and honestly"
            " it's kind of interesting."
        )
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert result.style == "conversational"

    def test_router_routes_emphatic_text(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = "Stop everything! This is absolutely incredible!"
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert result.style == "emphatic"

    def test_router_routes_narration_text(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = (
            "The history of computing is filled with ideas that arrived before"
            " the world was ready for them, concepts that lingered in obscurity"
            " before finally finding their moment."
        )
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert result.style == "narration"

    def test_router_returns_route_decision(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = (
            "The inference pipeline batches requests"
            " to maximize GPU throughput."
        )
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert isinstance(result, RouteDecision)
        assert isinstance(result.style, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.tier, str)
        assert isinstance(result.scores, dict)

    def test_router_single_available_style_returns_it(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        # Act
        result = router.route("Any text at all.", ["technical"])
        # Assert
        assert result.style == "technical"
        assert result.confidence == pytest.approx(1.0)

    def test_router_caches_result(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = "The API endpoint returns a 429 when rate limited."
        router.route(text, _ALL_STYLES)
        # Act — second call should come from cache
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert result.tier == "cache"

    def test_router_different_texts_different_styles(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        technical = (
            "The model's inference latency is measured in milliseconds"
            " per batch."
        )
        conversational = (
            "Honestly I think it's kind of wild how fast this thing runs."
        )
        # Act
        r1 = router.route(technical, _ALL_STYLES)
        r2 = router.route(conversational, _ALL_STYLES)
        # Assert
        assert r1.style != r2.style

    def test_router_scores_dict_has_all_styles(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = (
            "The transformer model processes tokens in parallel"
            " across attention heads."
        )
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        for style in _ALL_STYLES:
            assert style in result.scores

    def test_router_confidence_between_zero_and_one(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = (
            "The embedding model returns a dense vector for each input token."
        )
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert 0.0 <= result.confidence <= 1.0

    def test_router_invalidate_cache_clears(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange
        router = StyleRouter(cache_dir=tmp_path / "cache")
        text = "The API endpoint returns a 429 when rate limited."
        router.route(text, _ALL_STYLES)  # prime the cache
        router.invalidate_cache(text)
        # Act — should re-classify, not return from cache
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert result.tier != "cache"

    def test_router_with_custom_threshold(
        self, tmp_path: pytest.TempPathFactory
    ) -> None:
        # Arrange — threshold so high rule-based rarely satisfies it
        router = StyleRouter(
            cache_dir=tmp_path / "cache",
            confidence_threshold=0.99,
        )
        text = "So this happened and honestly it was kind of unexpected."
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert — centroid tier used because rule-based confidence < 0.99
        assert result.tier in ("centroid", "rule-based")

    def test_router_without_cache(self) -> None:
        # Arrange
        router = StyleRouter(cache_dir=None)
        text = "The API endpoint returns a 429 when rate limited."
        # Act
        result = router.route(text, _ALL_STYLES)
        # Assert
        assert isinstance(result, RouteDecision)
        assert result.style in _ALL_STYLES
