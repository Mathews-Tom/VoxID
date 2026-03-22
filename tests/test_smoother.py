from __future__ import annotations

import pytest

from voxid.router import RouteDecision
from voxid.segments.smoother import StyleSmoother


def _decision(style: str, confidence: float = 0.8) -> RouteDecision:
    return RouteDecision(
        style=style,
        confidence=confidence,
        tier="test",
        scores={style: confidence},
    )


def test_smooth_single_segment_no_change():
    # Arrange
    smoother = StyleSmoother()
    decisions = [_decision("technical", 0.9)]
    sentence_counts = [3]

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert
    assert len(result) == 1
    assert result[0].style == "technical"
    assert result[0].was_smoothed is False


def test_smooth_same_style_no_change():
    # Arrange
    smoother = StyleSmoother()
    decisions = [_decision("technical")] * 3
    sentence_counts = [2, 2, 2]

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert
    assert all(d.was_smoothed is False for d in result)
    assert all(d.style == "technical" for d in result)


def test_smooth_short_segment_inherits_previous():
    # Arrange — segment index 1 has only 1 sentence (below min_sentences=2)
    smoother = StyleSmoother(min_sentences_for_switch=2)
    decisions = [
        _decision("technical", 0.9),
        _decision("conversational", 0.95),  # different style, high confidence
        _decision("technical", 0.9),
    ]
    sentence_counts = [3, 1, 3]  # segment 1 is short

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert — short segment inherits previous ("technical")
    assert result[1].style == "technical"
    assert result[1].was_smoothed is True
    assert result[1].original_style == "conversational"


def test_smooth_long_segment_allows_switch():
    # Arrange — segment with 3 sentences and high confidence delta
    smoother = StyleSmoother(min_sentences_for_switch=2, switch_threshold=0.15)
    decisions = [
        _decision("technical", 0.5),
        _decision("conversational", 0.9),  # delta=0.4 > 0.15
    ]
    sentence_counts = [3, 3]

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert — switch allowed
    assert result[1].style == "conversational"
    assert result[1].was_smoothed is False


def test_smooth_low_delta_prevents_switch():
    # Arrange — delta = 0.05 < switch_threshold=0.15
    smoother = StyleSmoother(switch_threshold=0.15)
    decisions = [
        _decision("technical", 0.80),
        _decision("conversational", 0.85),  # delta=0.05
    ]
    sentence_counts = [3, 3]

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert — switch suppressed
    assert result[1].style == "technical"
    assert result[1].was_smoothed is True


def test_smooth_high_delta_allows_switch():
    # Arrange — delta = 0.30 > switch_threshold=0.15
    smoother = StyleSmoother(switch_threshold=0.15)
    decisions = [
        _decision("technical", 0.55),
        _decision("conversational", 0.85),  # delta=0.30
    ]
    sentence_counts = [3, 3]

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert — switch allowed
    assert result[1].style == "conversational"
    assert result[1].was_smoothed is False


def test_smooth_preserves_original_values():
    # Arrange
    smoother = StyleSmoother(switch_threshold=0.15)
    decisions = [
        _decision("technical", 0.80),
        _decision("conversational", 0.82),  # delta=0.02, will be smoothed
    ]
    sentence_counts = [3, 3]

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert
    smoothed = result[1]
    assert smoothed.original_style == "conversational"
    assert smoothed.original_confidence == pytest.approx(0.82)
    assert smoothed.was_smoothed is True


def test_smooth_custom_alpha():
    # Arrange — alpha=0.5 vs default 0.7 should give different blended confidence
    decisions = [
        _decision("technical", 0.80),
        _decision("conversational", 0.82),  # below threshold, will be blended
    ]
    sentence_counts = [3, 3]

    smoother_default = StyleSmoother(alpha=0.7, switch_threshold=0.15)
    smoother_custom = StyleSmoother(alpha=0.5, switch_threshold=0.15)

    # Act
    result_default = smoother_default.smooth(decisions, sentence_counts)
    result_custom = smoother_custom.smooth(decisions, sentence_counts)

    # Assert — blended confidences should differ
    assert result_default[1].confidence != pytest.approx(result_custom[1].confidence)
    # Verify numeric blend: alpha * prev + (1-alpha) * curr
    expected_default = 0.7 * 0.80 + 0.3 * 0.82
    expected_custom = 0.5 * 0.80 + 0.5 * 0.82
    assert result_default[1].confidence == pytest.approx(expected_default, rel=1e-5)
    assert result_custom[1].confidence == pytest.approx(expected_custom, rel=1e-5)


def test_smooth_no_switches_within_two_sentence_window():
    # Arrange — alternating single-sentence segments with different styles.
    # Short segments (< min_sentences=2) that differ from the previous
    # effective style are suppressed (was_smoothed=True). When a short
    # segment's routed style already matches the effective previous style
    # it is NOT marked smoothed — the style-equality fast path fires first.
    smoother = StyleSmoother(min_sentences_for_switch=2, switch_threshold=0.15)
    decisions = [
        _decision("technical", 0.9),      # 0: anchor
        _decision("conversational", 0.95), # 1: different + short → smoothed
        _decision("conversational", 0.95), # 2: diff from effective → smoothed
        _decision("conversational", 0.95), # 3: same as above → smoothed
    ]
    sentence_counts = [1, 1, 1, 1]  # all short

    # Act
    result = smoother.smooth(decisions, sentence_counts)

    # Assert
    assert result[0].style == "technical"
    assert result[0].was_smoothed is False
    # All subsequent segments differ from effective style and are short → smoothed
    for i in range(1, len(result)):
        assert result[i].was_smoothed is True
        assert result[i].style == "technical"
