from __future__ import annotations

from collections import Counter

from voxid.router.classifiers import (
    CentroidClassifier,
    ClassificationResult,
    RuleBasedClassifier,
    get_training_data,
)

_ALL_STYLES = ["conversational", "technical", "narration", "emphatic"]

# ---------------------------------------------------------------------------
# RuleBasedClassifier
# ---------------------------------------------------------------------------


class TestRuleBasedClassifier:
    def test_rule_classifier_technical_text_returns_technical(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        text = (
            "The embedding model uses a 768-dimensional dense vector space"
            " for semantic retrieval."
        )
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert result.style == "technical"

    def test_rule_classifier_conversational_text_returns_conversational(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        text = "Honestly, I think the whole thing took way longer than I expected."
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert result.style == "conversational"

    def test_rule_classifier_emphatic_text_returns_emphatic(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        text = "This changes everything! The numbers are absolutely unreal!"
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert result.style == "emphatic"

    def test_rule_classifier_narration_text_returns_narration(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        text = (
            "There is a certain clarity that comes from working on a problem"
            " long enough, a patience that only develops through sustained effort."
        )
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert result.style == "narration"

    def test_rule_classifier_empty_text_returns_first_available(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        # Act
        result = clf.classify("", _ALL_STYLES)
        # Assert
        assert result.style == _ALL_STYLES[0]
        assert result.confidence <= 0.6

    def test_rule_classifier_short_text_low_confidence(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        # Act
        result = clf.classify("Hello", _ALL_STYLES)
        # Assert
        assert result.confidence < 0.8

    def test_rule_classifier_filters_unavailable_styles(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        text = (
            "The embedding model uses a 768-dimensional dense vector space"
            " for semantic retrieval and inference."
        )
        available = ["conversational", "narration"]
        # Act
        result = clf.classify(text, available)
        # Assert
        assert result.style != "technical"
        assert result.style in available

    def test_rule_classifier_returns_classification_result(self) -> None:
        # Arrange
        clf = RuleBasedClassifier()
        text = "The API endpoint returns a 429 when the rate limiter triggers."
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert isinstance(result, ClassificationResult)
        assert isinstance(result.style, str)
        assert isinstance(result.confidence, float)
        assert isinstance(result.scores, dict)
        assert result.style in result.scores


# ---------------------------------------------------------------------------
# CentroidClassifier
# ---------------------------------------------------------------------------


class TestCentroidClassifier:
    def test_centroid_classifier_fit_and_classify(self) -> None:
        # Arrange
        clf = CentroidClassifier()
        clf.fit(get_training_data())
        text = (
            "The embedding model produces a 768-dimensional vector"
            " for each token in the sequence."
        )
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert result.style == "technical"

    def test_centroid_classifier_not_fitted_raises(self) -> None:
        # Arrange
        clf = CentroidClassifier()
        # Act / Assert
        import pytest

        with pytest.raises(RuntimeError):
            clf.classify("Some text", ["conversational", "technical"])

    def test_centroid_classifier_conversational_text(self) -> None:
        # Arrange
        clf = CentroidClassifier()
        clf.fit(get_training_data())
        text = "So I've been working on this thing and honestly it's pretty cool."
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        assert result.style == "conversational"

    def test_centroid_classifier_scores_sum_approximately_one(self) -> None:
        # Arrange
        clf = CentroidClassifier()
        clf.fit(get_training_data())
        text = "The inference pipeline batches requests to maximize GPU throughput."
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 1e-6

    def test_centroid_classifier_filters_available_styles(self) -> None:
        # Arrange
        clf = CentroidClassifier()
        clf.fit(get_training_data())
        text = (
            "The embedding model uses a 768-dimensional dense vector space"
            " for semantic retrieval."
        )
        available = ["conversational", "narration"]
        # Act
        result = clf.classify(text, available)
        # Assert
        assert result.style in available
        assert "technical" not in result.scores

    def test_centroid_classifier_all_styles_scored(self) -> None:
        # Arrange
        clf = CentroidClassifier()
        clf.fit(get_training_data())
        text = "The API returns JSON with a confidence score for each label."
        # Act
        result = clf.classify(text, _ALL_STYLES)
        # Assert
        for style in _ALL_STYLES:
            assert style in result.scores


# ---------------------------------------------------------------------------
# Training data
# ---------------------------------------------------------------------------


class TestTrainingData:
    def test_training_data_has_all_four_styles(self) -> None:
        # Arrange / Act
        data = get_training_data()
        labels = {label for _, label in data}
        # Assert
        assert {
            "conversational",
            "technical",
            "narration",
            "emphatic",
        }.issubset(labels)

    def test_training_data_minimum_per_class(self) -> None:
        # Arrange / Act
        data = get_training_data()
        counts = Counter(label for _, label in data)
        # Assert
        for style in ("conversational", "technical", "narration", "emphatic"):
            assert counts[style] >= 50, (
                f"{style} has only {counts[style]} examples"
            )

    def test_training_data_no_empty_texts(self) -> None:
        # Arrange / Act
        data = get_training_data()
        # Assert
        for text, label in data:
            assert text.strip() != "", f"Empty text found with label {label!r}"
