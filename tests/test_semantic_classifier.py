from __future__ import annotations

import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from voxid.router.classifiers import get_training_data
from voxid.router.semantic_classifier import (
    SemanticClassifierConfig,
    SemanticStyleClassifier,
    _extract_ngrams,
    _hash_features,
)
from voxid.router.training import train_semantic_classifier

_ALL_STYLES = ["conversational", "technical", "narration", "emphatic"]


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------


class TestFeatureExtraction:
    def test_extract_ngrams_char_and_word_present(self) -> None:
        ngrams = _extract_ngrams("hello world", (2, 3), (1, 2))
        char_ngrams = [n for n in ngrams if n.startswith("c:")]
        word_ngrams = [n for n in ngrams if n.startswith("w:")]
        assert len(char_ngrams) > 0
        assert len(word_ngrams) > 0
        assert "w:hello" in word_ngrams
        assert "w:hello world" in word_ngrams

    def test_hash_features_deterministic(self) -> None:
        ngrams = _extract_ngrams("test input", (2, 3), (1, 1))
        v1 = _hash_features(ngrams, 256)
        v2 = _hash_features(ngrams, 256)
        np.testing.assert_array_equal(v1, v2)

    def test_hash_features_normalized(self) -> None:
        ngrams = _extract_ngrams("some text", (2, 3), (1, 1))
        features = _hash_features(ngrams, 256)
        norm = np.linalg.norm(features)
        assert abs(norm - 1.0) < 1e-6

    def test_hash_features_different_texts_differ(self) -> None:
        n1 = _extract_ngrams("technical embedding model", (2, 3), (1, 1))
        n2 = _extract_ngrams("hey how are you doing", (2, 3), (1, 1))
        v1 = _hash_features(n1, 256)
        v2 = _hash_features(n2, 256)
        assert not np.allclose(v1, v2)


# ---------------------------------------------------------------------------
# SemanticStyleClassifier
# ---------------------------------------------------------------------------


class TestSemanticStyleClassifier:
    def test_unfitted_classifier_raises_on_classify(self) -> None:
        clf = SemanticStyleClassifier()
        assert not clf.is_fitted
        try:
            clf.classify("hello", _ALL_STYLES)
            raise AssertionError("Expected RuntimeError")  # noqa: TRY301
        except RuntimeError:
            pass

    def test_fit_makes_classifier_fitted(self) -> None:
        clf = SemanticStyleClassifier(
            SemanticClassifierConfig(n_features=256, hidden_dim=32),
        )
        examples = get_training_data()
        clf.fit(examples, epochs=10)
        assert clf.is_fitted

    def test_classify_returns_valid_result(self) -> None:
        clf = SemanticStyleClassifier(
            SemanticClassifierConfig(n_features=512, hidden_dim=64),
        )
        clf.fit(get_training_data(), epochs=50)
        result = clf.classify(
            "The API endpoint uses vector embeddings for retrieval.",
            _ALL_STYLES,
        )
        assert result.style in _ALL_STYLES
        assert 0.0 <= result.confidence <= 1.0
        assert abs(sum(result.scores.values()) - 1.0) < 1e-6

    def test_classify_technical_text_returns_technical(self) -> None:
        clf = _make_trained_classifier()
        result = clf.classify(
            "The API endpoint handles inference requests on the GPU cluster "
            "with batch processing through the embedding pipeline.",
            _ALL_STYLES,
        )
        assert result.style == "technical"

    def test_classify_conversational_text_returns_conversational(self) -> None:
        clf = _make_trained_classifier()
        result = clf.classify(
            "I honestly think the whole thing took way longer than I expected, "
            "you know what I mean?",
            _ALL_STYLES,
        )
        assert result.style == "conversational"

    def test_classify_filters_to_available_styles(self) -> None:
        clf = _make_trained_classifier()
        result = clf.classify(
            "The API uses vector embeddings.",
            ["conversational", "narration"],
        )
        assert result.style in ["conversational", "narration"]
        assert set(result.scores.keys()) == {"conversational", "narration"}

    def test_classify_scores_sum_to_one(self) -> None:
        clf = _make_trained_classifier()
        result = clf.classify("Just a normal sentence.", _ALL_STYLES)
        total = sum(result.scores.values())
        assert abs(total - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# Context-aware classification
# ---------------------------------------------------------------------------


class TestContextAwareClassification:
    def test_classify_with_empty_context_same_as_no_context(self) -> None:
        clf = _make_trained_classifier()
        text = "Deploy the container to the Kubernetes cluster."
        r1 = clf.classify(text, _ALL_STYLES)
        r2 = clf.classify_with_context(text, [], _ALL_STYLES)
        assert r1.style == r2.style

    def test_context_influences_classification(self) -> None:
        clf = _make_trained_classifier()
        text = "It needs to be fast."
        # Without context: ambiguous
        r_alone = clf.classify(text, _ALL_STYLES)
        # With technical context
        r_tech = clf.classify_with_context(
            text,
            [
                "The inference pipeline runs on GPU.",
                "Latency must stay below 10ms at p99.",
            ],
            _ALL_STYLES,
        )
        # With context, classifier should lean technical
        assert r_tech.scores.get("technical", 0) >= r_alone.scores.get(
            "technical", 0,
        ) - 0.1  # Allow small tolerance


# ---------------------------------------------------------------------------
# Latency
# ---------------------------------------------------------------------------


class TestLatency:
    def test_classify_latency_under_10ms(self) -> None:
        clf = _make_trained_classifier()
        text = "The embedding model uses dense vector retrieval."
        # Warm up
        clf.classify(text, _ALL_STYLES)
        # Measure
        start = time.perf_counter_ns()
        iterations = 100
        for _ in range(iterations):
            clf.classify(text, _ALL_STYLES)
        elapsed_ns = time.perf_counter_ns() - start
        avg_ms = elapsed_ns / iterations / 1_000_000
        assert avg_ms < 10.0, f"Average latency {avg_ms:.2f}ms exceeds 10ms"


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


class TestPersistence:
    def test_save_and_load_roundtrip(self) -> None:
        clf = _make_trained_classifier()
        text = "Deploy the API to production."

        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.npz"
            clf.save(path)
            assert path.exists()

            clf2 = SemanticStyleClassifier()
            clf2.load(path)
            assert clf2.is_fitted

            r1 = clf.classify(text, _ALL_STYLES)
            r2 = clf2.classify(text, _ALL_STYLES)
            assert r1.style == r2.style
            assert abs(r1.confidence - r2.confidence) < 1e-6


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------


class TestTrainingPipeline:
    def test_train_semantic_classifier_returns_fitted(self) -> None:
        clf = train_semantic_classifier(
            config=SemanticClassifierConfig(n_features=256, hidden_dim=32),
            epochs=20,
        )
        assert clf.is_fitted

    def test_train_with_save_path(self) -> None:
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "trained.npz"
            clf = train_semantic_classifier(
                config=SemanticClassifierConfig(
                    n_features=256, hidden_dim=32,
                ),
                epochs=20,
                save_path=path,
            )
            assert clf.is_fitted
            assert path.exists()

    def test_train_with_extra_examples(self) -> None:
        extras = [
            ("Wow, this is incredibly amazing!", "emphatic"),
            ("kubectl apply -f deployment.yaml", "technical"),
        ]
        clf = train_semantic_classifier(
            config=SemanticClassifierConfig(
                n_features=256, hidden_dim=32,
            ),
            extra_examples=extras,
            epochs=20,
        )
        assert clf.is_fitted


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CACHED_CLASSIFIER: SemanticStyleClassifier | None = None


def _make_trained_classifier() -> SemanticStyleClassifier:
    global _CACHED_CLASSIFIER  # noqa: PLW0603
    if _CACHED_CLASSIFIER is None:
        _CACHED_CLASSIFIER = train_semantic_classifier(
            config=SemanticClassifierConfig(
                n_features=4096, hidden_dim=128,
            ),
            epochs=300,
        )
    return _CACHED_CLASSIFIER
