from __future__ import annotations

from pathlib import Path

from voxid.router.classifiers import get_training_data
from voxid.router.semantic_classifier import (
    SemanticClassifierConfig,
    SemanticStyleClassifier,
)


def train_semantic_classifier(
    config: SemanticClassifierConfig | None = None,
    extra_examples: list[tuple[str, str]] | None = None,
    epochs: int = 200,
    learning_rate: float = 0.01,
    save_path: Path | None = None,
) -> SemanticStyleClassifier:
    """Train a SemanticStyleClassifier on the built-in training corpus.

    Args:
        config: Classifier configuration. Uses defaults if None.
        extra_examples: Additional (text, style) pairs to augment training.
        epochs: Number of training epochs.
        learning_rate: SGD learning rate.
        save_path: If provided, save trained weights to this .npz path.

    Returns:
        A fitted SemanticStyleClassifier ready for inference.
    """
    examples = get_training_data()
    if extra_examples:
        examples = [*examples, *extra_examples]

    classifier = SemanticStyleClassifier(config)
    classifier.fit(
        examples,
        epochs=epochs,
        learning_rate=learning_rate,
    )

    if save_path is not None:
        classifier.save(save_path)

    return classifier
