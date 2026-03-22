from __future__ import annotations

from dataclasses import dataclass

from voxid.router import RouteDecision


@dataclass(frozen=True)
class SmoothedDecision:
    style: str
    confidence: float
    original_style: str
    original_confidence: float
    was_smoothed: bool


class StyleSmoother:
    """Smooth style transitions across consecutive segments.

    Rules:
    1. Minimum segment length before style switch: 2 sentences.
       If a segment has < 2 sentences, it inherits the previous
       segment's style.
    2. Confidence delta threshold: a style switch only occurs if
       the new style's confidence exceeds the previous by at
       least `switch_threshold`.
    3. Convex interpolation: when a switch is marginal,
       blend confidence scores using α=0.7 (previous weight).
    """

    def __init__(
        self,
        min_sentences_for_switch: int = 2,
        switch_threshold: float = 0.15,
        alpha: float = 0.7,
    ) -> None:
        self._min_sentences = min_sentences_for_switch
        self._switch_threshold = switch_threshold
        self._alpha = alpha

    def smooth(
        self,
        decisions: list[RouteDecision],
        sentence_counts: list[int],
    ) -> list[SmoothedDecision]:
        """Smooth a sequence of routing decisions.

        Args:
            decisions: per-segment RouteDecision from the router
            sentence_counts: number of sentences in each segment
                (from TextSegment.sentence_count)

        Returns:
            SmoothedDecision per segment, with was_smoothed=True
            if the style was changed by smoothing.
        """
        if not decisions:
            return []

        if len(decisions) != len(sentence_counts):
            raise ValueError(
                f"decisions length {len(decisions)} != "
                f"sentence_counts length {len(sentence_counts)}"
            )

        result: list[SmoothedDecision] = []

        for i, decision in enumerate(decisions):
            original_style = decision.style
            original_confidence = decision.confidence

            if i == 0:
                result.append(
                    SmoothedDecision(
                        style=original_style,
                        confidence=original_confidence,
                        original_style=original_style,
                        original_confidence=original_confidence,
                        was_smoothed=False,
                    )
                )
                continue

            prev = result[i - 1]
            prev_style = prev.style
            prev_confidence = prev.confidence

            if original_style == prev_style:
                result.append(
                    SmoothedDecision(
                        style=original_style,
                        confidence=original_confidence,
                        original_style=original_style,
                        original_confidence=original_confidence,
                        was_smoothed=False,
                    )
                )
                continue

            # Different style — apply smoothing rules

            # Rule 1: short segments inherit previous style
            if sentence_counts[i] < self._min_sentences:
                blended = (
                    self._alpha * prev_confidence
                    + (1 - self._alpha) * original_confidence
                )
                result.append(
                    SmoothedDecision(
                        style=prev_style,
                        confidence=blended,
                        original_style=original_style,
                        original_confidence=original_confidence,
                        was_smoothed=True,
                    )
                )
                continue

            # Rule 2: confidence delta must exceed threshold
            delta = original_confidence - prev_confidence
            if delta < self._switch_threshold:
                blended = (
                    self._alpha * prev_confidence
                    + (1 - self._alpha) * original_confidence
                )
                result.append(
                    SmoothedDecision(
                        style=prev_style,
                        confidence=blended,
                        original_style=original_style,
                        original_confidence=original_confidence,
                        was_smoothed=True,
                    )
                )
                continue

            # Sufficient confidence for switch
            result.append(
                SmoothedDecision(
                    style=original_style,
                    confidence=original_confidence,
                    original_style=original_style,
                    original_confidence=original_confidence,
                    was_smoothed=False,
                )
            )

        return result
