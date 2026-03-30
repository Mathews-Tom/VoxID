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
from voxid.router.semantic_classifier import SemanticStyleClassifier


@dataclass(frozen=True)
class RouteDecision:
    style: str
    confidence: float
    tier: str  # "rule-based" | "semantic" | "centroid" | "cache" | "default"
    scores: dict[str, float]


class StyleRouter:
    """Orchestrates style classification via tiered classifiers.

    Routing tiers:
        Tier 1 (Rule-based, < 1ms)  → confidence ≥ 0.9 → use
                                     → confidence < 0.9 → Tier 1.5
        Tier 1.5 (Semantic, < 10ms) → confidence ≥ 0.8 → use
                                     → confidence < 0.8 → Tier 2
        Tier 2 (Centroid, ~15ms)    → always returns a decision

    Tier 1.5 is active only when a fitted SemanticStyleClassifier is provided.
    """

    def __init__(
        self,
        cache_dir: Path | None = None,
        confidence_threshold: float = 0.9,
        semantic_threshold: float = 0.8,
        cache_ttl: int = 3600,
        semantic_classifier: SemanticStyleClassifier | None = None,
    ) -> None:
        self._threshold = confidence_threshold
        self._semantic_threshold = semantic_threshold
        self._rule_classifier = RuleBasedClassifier()
        self._centroid_classifier = CentroidClassifier()
        self._centroid_classifier.fit(get_training_data())

        self._semantic_classifier: SemanticStyleClassifier | None = None
        if semantic_classifier is not None and semantic_classifier.is_fitted:
            self._semantic_classifier = semantic_classifier

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
        context_texts: list[str] | None = None,
    ) -> RouteDecision:
        """Route text to the most appropriate style.

        Tier cascade:
        1. Single style → return immediately (tier="default").
        2. Cache hit → return cached decision (tier="cache").
        3. Tier 1: RuleBasedClassifier → confidence ≥ 0.9 → use.
        4. Tier 1.5: SemanticStyleClassifier → confidence ≥ 0.8 → use.
        5. Tier 2: CentroidClassifier → pick best between remaining.
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

        # Tier 1: Rule-based classification
        rule_result = self._rule_classifier.classify(text, available_styles)

        if rule_result.confidence >= self._threshold:
            return self._cache_and_return(
                text,
                rule_result.style,
                rule_result.confidence,
                "rule-based",
                rule_result.scores,
            )

        # Tier 1.5: Semantic classification (when available)
        if self._semantic_classifier is not None:
            if context_texts is not None:
                sem_result = self._semantic_classifier.classify_with_context(
                    text, context_texts, available_styles,
                )
            else:
                sem_result = self._semantic_classifier.classify(
                    text, available_styles,
                )

            if sem_result.confidence >= self._semantic_threshold:
                return self._cache_and_return(
                    text,
                    sem_result.style,
                    sem_result.confidence,
                    "semantic",
                    sem_result.scores,
                )

        # Tier 2: Centroid classification
        centroid_result = self._centroid_classifier.classify(
            text, available_styles,
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

        return self._cache_and_return(
            text, final_style, final_confidence, tier, final_scores,
        )

    def _cache_and_return(
        self,
        text: str,
        style: str,
        confidence: float,
        tier: str,
        scores_dict: dict[str, float],
    ) -> RouteDecision:
        """Cache a result and return it as a RouteDecision."""
        if self._cache is not None:
            self._cache.put(text, style, confidence, tier, scores_dict)
        return RouteDecision(
            style=style,
            confidence=confidence,
            tier=tier,
            scores=scores_dict,
        )

    def invalidate_cache(self, text: str | None = None) -> None:
        """Clear one cache entry by text, or all entries if text is None."""
        if self._cache is not None:
            self._cache.invalidate(text)

    def close(self) -> None:
        """Release cache resources."""
        if self._cache is not None:
            self._cache.close()
