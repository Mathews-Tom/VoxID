from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from voxid.router.cache import RouterCache
from voxid.router.classifiers import (
    CentroidClassifier,
    RuleBasedClassifier,
    get_training_data,
)


@dataclass(frozen=True)
class RouteDecision:
    style: str
    confidence: float
    tier: str  # "rule-based" | "centroid" | "cache" | "default"
    scores: dict[str, float]


class StyleRouter:
    """Orchestrates style classification via rule-based and centroid classifiers."""

    def __init__(
        self,
        cache_dir: Path | None = None,
        confidence_threshold: float = 0.8,
        cache_ttl: int = 3600,
    ) -> None:
        self._threshold = confidence_threshold
        self._rule_classifier = RuleBasedClassifier()
        self._centroid_classifier = CentroidClassifier()
        self._centroid_classifier.fit(get_training_data())

        self._cache: RouterCache | None = None
        if cache_dir is not None:
            self._cache = RouterCache(
                db_path=cache_dir / "router_cache.db",
                ttl_seconds=cache_ttl,
            )

    def route(
        self,
        text: str,
        available_styles: list[str],
        default_style: str = "conversational",
    ) -> RouteDecision:
        """Route text to the most appropriate style.

        Order of operations:
        1. If only one style available, return it immediately (tier="default").
        2. Check cache; on hit, return cached decision (tier="cache").
        3. Run RuleBasedClassifier; if confidence >= threshold, cache and return.
        4. Run CentroidClassifier; pick higher-confidence result between the two.
        5. Cache and return final decision.
        """
        if not available_styles:
            available_styles = [default_style]

        if len(available_styles) == 1:
            return RouteDecision(
                style=available_styles[0],
                confidence=1.0,
                tier="default",
                scores={available_styles[0]: 1.0},
            )

        # Cache lookup
        if self._cache is not None:
            cached = self._cache.get(text)
            if cached is not None:
                scores: dict[str, float] = json.loads(cached.scores_json)
                return RouteDecision(
                    style=cached.style,
                    confidence=cached.confidence,
                    tier="cache",
                    scores=scores,
                )

        # Rule-based classification
        rule_result = self._rule_classifier.classify(text, available_styles)

        if rule_result.confidence >= self._threshold:
            if self._cache is not None:
                self._cache.put(
                    text,
                    rule_result.style,
                    rule_result.confidence,
                    "rule-based",
                    rule_result.scores,
                )
            return RouteDecision(
                style=rule_result.style,
                confidence=rule_result.confidence,
                tier="rule-based",
                scores=rule_result.scores,
            )

        # Centroid fallback
        centroid_result = self._centroid_classifier.classify(
            text, available_styles
        )

        if centroid_result.confidence >= rule_result.confidence:
            final_style = centroid_result.style
            final_confidence = centroid_result.confidence
            final_scores = centroid_result.scores
            tier = "centroid"
        else:
            final_style = rule_result.style
            final_confidence = rule_result.confidence
            final_scores = rule_result.scores
            tier = "rule-based"

        if self._cache is not None:
            self._cache.put(
                text,
                final_style,
                final_confidence,
                tier,
                final_scores,
            )

        return RouteDecision(
            style=final_style,
            confidence=final_confidence,
            tier=tier,
            scores=final_scores,
        )

    def invalidate_cache(self, text: str | None = None) -> None:
        """Clear one cache entry by text, or all entries if text is None."""
        if self._cache is not None:
            self._cache.invalidate(text)

    def close(self) -> None:
        """Release cache resources."""
        if self._cache is not None:
            self._cache.close()
