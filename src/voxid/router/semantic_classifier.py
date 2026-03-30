from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from voxid.router.classifiers import ClassificationResult


@dataclass(frozen=True)
class SemanticClassifierConfig:
    """Configuration for the semantic style classifier."""

    n_features: int = 4096
    hidden_dim: int = 128
    context_window: int = 2
    context_decay: float = 0.5
    confidence_threshold: float = 0.8
    char_ngram_range: tuple[int, int] = (2, 4)
    word_ngram_range: tuple[int, int] = (1, 2)


@dataclass
class _CalibrationParams:
    """Temperature scaling for calibrated multi-class confidence."""

    temperature: float = 1.0


@dataclass
class _MLPWeights:
    """Stored MLP weights for inference."""

    w1: NDArray[np.float64]  # (n_features, hidden_dim)
    b1: NDArray[np.float64]  # (hidden_dim,)
    w2: NDArray[np.float64]  # (hidden_dim, n_classes)
    b2: NDArray[np.float64]  # (n_classes,)
    style_labels: list[str] = field(default_factory=list)
    calibration: _CalibrationParams = field(
        default_factory=_CalibrationParams,
    )


def _extract_ngrams(
    text: str,
    char_range: tuple[int, int],
    word_range: tuple[int, int],
) -> list[str]:
    """Extract character and word n-grams from text."""
    ngrams: list[str] = []
    lower = text.lower()

    for n in range(char_range[0], char_range[1] + 1):
        for i in range(len(lower) - n + 1):
            ngrams.append(f"c:{lower[i:i + n]}")

    words = lower.split()
    for n in range(word_range[0], word_range[1] + 1):
        for i in range(len(words) - n + 1):
            ngrams.append(f"w:{' '.join(words[i:i + n])}")

    return ngrams


def _hash_features(ngrams: list[str], n_features: int) -> NDArray[np.float64]:
    """Hash n-grams into a fixed-size feature vector using the hashing trick."""
    features = np.zeros(n_features, dtype=np.float64)
    for ngram in ngrams:
        h = int(hashlib.md5(ngram.encode(), usedforsecurity=False).hexdigest(), 16)  # noqa: S324
        idx = h % n_features
        sign = 1.0 if (h // n_features) % 2 == 0 else -1.0
        features[idx] += sign
    norm = np.linalg.norm(features)
    if norm > 0:
        features /= norm
    return features


def _relu(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.maximum(0, x)


def _softmax(logits: NDArray[np.float64]) -> NDArray[np.float64]:
    shifted = logits - logits.max()
    exp_vals = np.exp(shifted)
    result: NDArray[np.float64] = exp_vals / exp_vals.sum()
    return result


class SemanticStyleClassifier:
    """Tier 1.5 semantic classifier: hashed n-gram features + MLP.

    Uses character and word n-grams hashed into a fixed-size vector,
    fed through a two-layer MLP with ReLU activation. Supports
    contextual classification using ±2 neighboring segment embeddings.
    Platt scaling calibrates raw logits into reliable confidence scores.
    """

    def __init__(self, config: SemanticClassifierConfig | None = None) -> None:
        self._config = config or SemanticClassifierConfig()
        self._weights: _MLPWeights | None = None

    @property
    def is_fitted(self) -> bool:
        return self._weights is not None

    def _embed(self, text: str) -> NDArray[np.float64]:
        """Convert text to a fixed-size feature vector."""
        ngrams = _extract_ngrams(
            text,
            self._config.char_ngram_range,
            self._config.word_ngram_range,
        )
        return _hash_features(ngrams, self._config.n_features)

    def _forward(self, features: NDArray[np.float64]) -> NDArray[np.float64]:
        """Run MLP forward pass, returning raw logits."""
        if self._weights is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        h = _relu(features @ self._weights.w1 + self._weights.b1)
        return h @ self._weights.w2 + self._weights.b2

    def _calibrated_softmax(
        self,
        logits: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        """Apply temperature-scaled softmax for calibrated confidence."""
        if self._weights is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        temp = self._weights.calibration.temperature
        return _softmax(logits / temp)

    def classify(
        self,
        text: str,
        available_styles: list[str],
    ) -> ClassificationResult:
        """Classify text into a style using semantic features."""
        if self._weights is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        features = self._embed(text)
        logits = self._forward(features)
        probs = self._calibrated_softmax(logits)
        return self._build_result(probs, available_styles)

    def classify_with_context(
        self,
        text: str,
        context_texts: list[str],
        available_styles: list[str],
    ) -> ClassificationResult:
        """Classify text using ±N neighboring segments as context.

        Context embeddings are averaged with exponential distance decay
        and added to the target embedding before MLP forward pass.
        """
        if self._weights is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        target_features = self._embed(text)

        if context_texts:
            context_sum = np.zeros_like(target_features)
            weight_sum = 0.0
            for i, ctx in enumerate(context_texts):
                distance = i + 1
                weight = self._config.context_decay ** distance
                context_sum += weight * self._embed(ctx)
                weight_sum += weight
            if weight_sum > 0:
                context_avg = context_sum / weight_sum
                # Blend: 80% target, 20% context
                target_features = 0.8 * target_features + 0.2 * context_avg
                norm = np.linalg.norm(target_features)
                if norm > 0:
                    target_features /= norm

        logits = self._forward(target_features)
        probs = self._calibrated_softmax(logits)
        return self._build_result(probs, available_styles)

    def _build_result(
        self,
        probs: NDArray[np.float64],
        available_styles: list[str],
    ) -> ClassificationResult:
        """Build ClassificationResult, filtering to available styles."""
        if self._weights is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")

        all_scores: dict[str, float] = {}
        for i, label in enumerate(self._weights.style_labels):
            all_scores[label] = float(probs[i])

        # Filter to available styles and renormalize
        filtered: dict[str, float] = {
            s: all_scores.get(s, 0.0) for s in available_styles
        }
        total = sum(filtered.values())
        if total > 0:
            filtered = {s: v / total for s, v in filtered.items()}
        else:
            uniform = 1.0 / len(available_styles)
            filtered = {s: uniform for s in available_styles}

        best_style = max(filtered, key=filtered.get)  # type: ignore[arg-type]
        return ClassificationResult(
            style=best_style,
            confidence=filtered[best_style],
            scores=filtered,
        )

    def fit(
        self,
        examples: list[tuple[str, str]],
        epochs: int = 200,
        learning_rate: float = 0.01,
        seed: int = 42,
    ) -> None:
        """Train the MLP on labeled (text, style) examples.

        Uses mini-batch SGD with cross-entropy loss. After training,
        fits Platt scaling parameters on the training logits.
        """
        rng = np.random.default_rng(seed)
        style_set = sorted({s for _, s in examples})
        style_to_idx = {s: i for i, s in enumerate(style_set)}
        n_classes = len(style_set)
        n_feat = self._config.n_features
        hidden = self._config.hidden_dim

        # Build feature matrix
        x_all = np.array([self._embed(text) for text, _ in examples])
        y_all = np.array([style_to_idx[s] for _, s in examples])

        # Xavier initialization
        w1 = rng.standard_normal((n_feat, hidden)) * math.sqrt(2.0 / n_feat)
        b1 = np.zeros(hidden)
        w2 = rng.standard_normal((hidden, n_classes)) * math.sqrt(2.0 / hidden)
        b2 = np.zeros(n_classes)

        batch_size = min(32, len(examples))

        for _epoch in range(epochs):
            indices = rng.permutation(len(examples))
            for start in range(0, len(examples), batch_size):
                batch_idx = indices[start : start + batch_size]
                x_batch = x_all[batch_idx]
                y_batch = y_all[batch_idx]

                # Forward
                h = _relu(x_batch @ w1 + b1)
                logits = h @ w2 + b2
                probs = np.array([_softmax(row) for row in logits])

                # Cross-entropy gradient
                grad_logits = probs.copy()
                for i, yi in enumerate(y_batch):
                    grad_logits[i, yi] -= 1.0
                grad_logits /= len(y_batch)

                # Backward
                grad_w2 = h.T @ grad_logits
                grad_b2 = grad_logits.sum(axis=0)
                grad_h = grad_logits @ w2.T
                grad_h[h <= 0] = 0.0  # ReLU derivative
                grad_w1 = x_batch.T @ grad_h
                grad_b1 = grad_h.sum(axis=0)

                # Update
                w1 -= learning_rate * grad_w1
                b1 -= learning_rate * grad_b1
                w2 -= learning_rate * grad_w2
                b2 -= learning_rate * grad_b2

        # Fit temperature scaling on training data
        calibration = self._fit_temperature(
            x_all, y_all, w1, b1, w2, b2,
        )

        self._weights = _MLPWeights(
            w1=w1,
            b1=b1,
            w2=w2,
            b2=b2,
            style_labels=style_set,
            calibration=calibration,
        )

    def _fit_temperature(
        self,
        x: NDArray[np.float64],
        y: NDArray[np.int64],
        w1: NDArray[np.float64],
        b1: NDArray[np.float64],
        w2: NDArray[np.float64],
        b2: NDArray[np.float64],
    ) -> _CalibrationParams:
        """Fit temperature parameter to minimize NLL on training logits."""
        h = _relu(x @ w1 + b1)
        logits = h @ w2 + b2

        best_t = 1.0
        best_nll = float("inf")

        for t_candidate in np.arange(0.1, 5.0, 0.1):
            nll = 0.0
            for i in range(len(y)):
                probs = _softmax(logits[i] / t_candidate)
                p = max(probs[y[i]], 1e-15)
                nll -= math.log(p)
            if nll < best_nll:
                best_nll = nll
                best_t = float(t_candidate)

        return _CalibrationParams(temperature=best_t)

    def save(self, path: Path) -> None:
        """Save trained weights and config to a .npz file."""
        if self._weights is None:
            raise RuntimeError("Classifier not fitted. Call fit() first.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            w1=self._weights.w1,
            b1=self._weights.b1,
            w2=self._weights.w2,
            b2=self._weights.b2,
            style_labels=np.array(self._weights.style_labels),
            temperature=np.array(
                [self._weights.calibration.temperature],
            ),
            n_features=np.array([self._config.n_features]),
            hidden_dim=np.array([self._config.hidden_dim]),
        )

    def load(self, path: Path) -> None:
        """Load trained weights from a .npz file."""
        data = np.load(str(path), allow_pickle=False)
        n_features = int(data["n_features"][0])
        hidden_dim = int(data["hidden_dim"][0])
        self._config = SemanticClassifierConfig(
            n_features=n_features,
            hidden_dim=hidden_dim,
        )
        self._weights = _MLPWeights(
            w1=data["w1"],
            b1=data["b1"],
            w2=data["w2"],
            b2=data["b2"],
            style_labels=list(data["style_labels"]),
            calibration=_CalibrationParams(
                temperature=float(data["temperature"][0]),
            ),
        )
