from __future__ import annotations

import re
from dataclasses import dataclass

import pytest

from voxid.segments.segmenter import TextSegmenter


@dataclass
class LabeledDocument:
    text: str
    expected_segment_count: int
    expected_boundary_types: list[str]  # per segment, in order


# ---------------------------------------------------------------------------
# Labeled corpus — 20 documents with known structure
# ---------------------------------------------------------------------------

# Group A: Simple 2-paragraph texts (docs 0-4)
_A0 = LabeledDocument(
    text=(
        "The sun rose over the mountains. "
        "It cast a golden light across the valley. "
        "Birds began to sing as morning arrived.\n\n"
        "In the village below, people woke to the sound of church bells. "
        "Farmers headed to their fields. The day had begun."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_A1 = LabeledDocument(
    text=(
        "Machine learning models require large amounts of training data. "
        "Without sufficient data, models tend to overfit.\n\n"
        "Regularisation techniques help combat overfitting. "
        "Dropout and weight decay are common strategies."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_A2 = LabeledDocument(
    text=(
        "She opened the letter and read it twice. "
        "Her hands trembled slightly. "
        "The words were both unexpected and welcome.\n\n"
        "She set the letter down and looked out the window. "
        "A smile slowly formed on her face."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_A3 = LabeledDocument(
    text=(
        "Water boils at one hundred degrees Celsius at sea level. "
        "At higher altitudes the boiling point is lower.\n\n"
        "This affects cooking times significantly. "
        "Recipes often need adjustment in mountainous regions."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_A4 = LabeledDocument(
    text=(
        "The project kicked off in January. "
        "The team was small but motivated. "
        "Everyone understood the deadline.\n\n"
        "By March the first milestone was reached. "
        "The client was satisfied with the progress."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

# Group B: Multi-paragraph blog post style (docs 5-9)
_B0 = LabeledDocument(
    text=(
        "Welcome to our weekly newsletter. "
        "This week we cover three major topics from open-source software.\n\n"
        "First, the Rust community released version 1.80 with exciting new features. "
        "The borrow checker improvements are particularly noteworthy.\n\n"
        "Second, Python 3.13 entered its release candidate phase. "
        "The new interpreter optimisations promise significant speed gains.\n\n"
        "Finally, the Go team announced extended support for Go 1.22. "
        "Long-term support releases help enterprise teams plan upgrades."
    ),
    expected_segment_count=4,
    expected_boundary_types=["paragraph", "paragraph", "paragraph", "paragraph"],
)

_B1 = LabeledDocument(
    text=(
        "Today I want to share what I learned building my first production API. "
        "It was a humbling experience.\n\n"
        "Authentication was harder than expected. "
        "JWT tokens seemed simple until refresh logic entered the picture.\n\n"
        "Rate limiting saved me from a runaway client script. "
        "Always implement rate limits, even in internal services.\n\n"
        "Logging was the unsung hero of the whole project. "
        "Without structured logs I would have been blind to production issues."
    ),
    expected_segment_count=4,
    expected_boundary_types=["paragraph", "paragraph", "paragraph", "paragraph"],
)

_B2 = LabeledDocument(
    text=(
        "The history of the internet begins in the late 1960s. "
        "ARPANET connected four university computers for the first time.\n\n"
        "Email arrived in the early 1970s, transforming professional communication. "
        "Ray Tomlinson sent the first network email in 1971.\n\n"
        "The World Wide Web was invented by Tim Berners-Lee in 1989. "
        "He wanted a way to share documents across CERN."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

_B3 = LabeledDocument(
    text=(
        "Great user interfaces share a few core principles. "
        "Consistency, clarity, and feedback are paramount.\n\n"
        "Consistency means users can apply learned patterns everywhere. "
        "Surprise is the enemy of usability.\n\n"
        "Clarity ensures the purpose of each element is immediately obvious. "
        "If you need a tooltip to explain a button, the button is unclear.\n\n"
        "Feedback confirms that user actions had the intended effect. "
        "A spinner, a confirmation message, or a sound all provide feedback."
    ),
    expected_segment_count=4,
    expected_boundary_types=["paragraph", "paragraph", "paragraph", "paragraph"],
)

_B4 = LabeledDocument(
    text=(
        "Running a marathon requires months of consistent training. "
        "Most beginners underestimate the commitment involved.\n\n"
        "Base mileage must be built slowly to avoid injury. "
        "The ten-percent rule limits weekly mileage increases.\n\n"
        "Long runs on weekends are the cornerstone of marathon preparation. "
        "They build the aerobic base that race day demands."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

# Group C: Technical documentation with code terms (docs 10-12)
_C0 = LabeledDocument(
    text=(
        "The TextSegmenter class splits long-form text at prosodic boundaries. "
        "It accepts min_sentences, max_sentences, and min_words parameters.\n\n"
        "Calling segmenter.segment(text) returns a list of TextSegment objects. "
        "Each segment carries index, boundary_type, and sentence_count.\n\n"
        "The boundary hierarchy is: paragraph > sentence > clause. "
        "Short segments below min_words are merged with the preceding segment."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

_C1 = LabeledDocument(
    text=(
        "To install the package run `uv add voxid` in your project directory. "
        "This resolves all dependencies and writes the lock file.\n\n"
        "Import the main class with `from voxid import VoxID`. "
        "Instantiate it using `vox = VoxID()` to load config from the home dir."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_C2 = LabeledDocument(
    text=(
        "The StyleRouter.route() method accepts text, a list of available styles, "
        "and an optional default style string. "
        "It returns a RouteDecision dataclass.\n\n"
        "Rule-based classification runs first. "
        "If confidence is below the threshold, the centroid classifier is tried. "
        "Results are cached in SQLite for fast repeated lookups.\n\n"
        "Cache invalidation is possible via router.invalidate_cache(text). "
        "Passing None clears the entire cache."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

# Group D: Conversational text with questions (docs 13-15)
_D0 = LabeledDocument(
    text=(
        "Have you ever wondered why some people seem to learn faster than others? "
        "It turns out practice quality matters more than raw hours.\n\n"
        "Deliberate practice means working at the edge of your current skill. "
        "Are you challenging yourself, or just repeating what you already know?"
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_D1 = LabeledDocument(
    text=(
        "What makes a good mentor? "
        "A good mentor listens more than they speak. "
        "They ask questions that lead you to your own answers.\n\n"
        "Have you found someone who challenges your assumptions? "
        "If not, where could you look? "
        "Professional communities and open-source projects are good places to start."
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

_D2 = LabeledDocument(
    text=(
        "Why do meetings so often feel unproductive? "
        "Often there is no clear agenda and no defined outcome.\n\n"
        "What would happen if every meeting had a written objective? "
        "Studies suggest shorter, focused meetings improve team velocity. "
        "Is that a trade-off your team is willing to make?"
    ),
    expected_segment_count=2,
    expected_boundary_types=["paragraph", "paragraph"],
)

# Group E: Mixed paragraphs with varying sentence counts (docs 16-19)
_E0 = LabeledDocument(
    text=(
        "The data pipeline runs nightly.\n\n"
        "Step one extracts raw events from the Kafka topic. "
        "Step two applies schema validation using Pydantic models. "
        "Invalid records are quarantined in a dead-letter topic. "
        "Step three writes cleaned data to the data warehouse.\n\n"
        "Monitoring alerts fire if the pipeline fails or runs too long."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

_E1 = LabeledDocument(
    text=(
        "The recipe is surprisingly simple. "
        "You only need three ingredients.\n\n"
        "Combine two cups of flour, one cup of sugar, and three eggs in a bowl. "
        "Mix until smooth. "
        "Pour into a greased tin. "
        "Bake at 180 degrees for twenty-five minutes. "
        "Allow to cool before serving.\n\n"
        "Variations include adding vanilla extract or lemon zest."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

_E2 = LabeledDocument(
    text=(
        "Climate change is one of the defining challenges of the century. "
        "Global temperatures have risen by approximately 1.2 degrees Celsius. "
        "The effects are visible in extreme weather and sea-level rise.\n\n"
        "Mitigation requires both individual and systemic change.\n\n"
        "Adaptation strategies must be developed in parallel. "
        "Coastal cities are already investing in flood defences. "
        "Agricultural systems are being redesigned for changing rainfall patterns."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

_E3 = LabeledDocument(
    text=(
        "Sleep is foundational to cognitive performance.\n\n"
        "During deep sleep the brain consolidates memories formed during the day. "
        "Synaptic connections are pruned and strengthened. "
        "Cerebrospinal fluid flushes metabolic waste from neural tissue.\n\n"
        "Chronic sleep deprivation impairs decision-making and emotional regulation. "
        "Even a single night of poor sleep reduces working memory capacity."
    ),
    expected_segment_count=3,
    expected_boundary_types=["paragraph", "paragraph", "paragraph"],
)

LABELED_CORPUS: list[LabeledDocument] = [
    _A0, _A1, _A2, _A3, _A4,
    _B0, _B1, _B2, _B3, _B4,
    _C0, _C1, _C2,
    _D0, _D1, _D2,
    _E0, _E1, _E2, _E3,
]

assert len(LABELED_CORPUS) == 20, "Corpus must have exactly 20 documents"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _segmenter() -> TextSegmenter:
    return TextSegmenter(min_sentences=1, max_sentences=5, min_words=5)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_segmenter_segment_count_accuracy() -> None:
    """Segment count matches expected within ±1 tolerance for ≥ 80% of docs."""
    seg = _segmenter()
    hits = 0
    rows: list[str] = []

    for i, doc in enumerate(LABELED_CORPUS):
        segments = seg.segment(doc.text)
        got = len(segments)
        expected = doc.expected_segment_count
        within = abs(got - expected) <= 1
        if within:
            hits += 1
        rows.append(
            f"  doc {i:02d}: expected={expected}  got={got}"
            f"  {'OK' if within else 'FAIL'}"
        )

    accuracy_pct = hits / len(LABELED_CORPUS) * 100
    n = len(LABELED_CORPUS)
    print(f"\nSegment count accuracy: {hits}/{n} ({accuracy_pct:.0f}%)")
    for row in rows:
        print(row)

    assert hits / len(LABELED_CORPUS) >= 0.80, (
        f"Segment count accuracy {accuracy_pct:.1f}% is below 80%"
    )


def test_segmenter_boundary_type_precision() -> None:
    """Of all predicted 'paragraph' segments, ≥ 80% are true paragraph boundaries."""
    seg = _segmenter()
    true_pos = 0
    false_pos = 0

    for doc in LABELED_CORPUS:
        segments = seg.segment(doc.text)
        for j, s in enumerate(segments):
            if s.boundary_type == "paragraph":
                if j < len(doc.expected_boundary_types):
                    if doc.expected_boundary_types[j] == "paragraph":
                        true_pos += 1
                    else:
                        false_pos += 1
                else:
                    false_pos += 1

    total_predicted = true_pos + false_pos
    precision = true_pos / total_predicted if total_predicted > 0 else 0.0
    print(
        f"\nParagraph boundary precision:"
        f" {true_pos}/{total_predicted} = {precision:.2f}"
    )
    assert precision >= 0.80, f"Precision {precision:.2f} is below 0.80"


def test_segmenter_boundary_type_recall() -> None:
    """Of all true 'paragraph' boundaries, ≥ 80% are detected."""
    seg = _segmenter()
    true_pos = 0
    false_neg = 0

    for doc in LABELED_CORPUS:
        segments = seg.segment(doc.text)
        predicted_types = [s.boundary_type for s in segments]

        for j, expected_type in enumerate(doc.expected_boundary_types):
            if expected_type == "paragraph":
                if j < len(predicted_types) and predicted_types[j] == "paragraph":
                    true_pos += 1
                else:
                    false_neg += 1

    total_actual = true_pos + false_neg
    recall = true_pos / total_actual if total_actual > 0 else 0.0
    print(f"\nParagraph boundary recall: {true_pos}/{total_actual} = {recall:.2f}")
    assert recall >= 0.80, f"Recall {recall:.2f} is below 0.80"


def test_segmenter_boundary_f1_report(capsys: pytest.CaptureFixture[str]) -> None:
    """Compute and print P/R/F1 per boundary type to stdout."""
    seg = _segmenter()

    stats: dict[str, dict[str, int]] = {
        "paragraph": {"tp": 0, "fp": 0, "fn": 0},
        "sentence": {"tp": 0, "fp": 0, "fn": 0},
    }

    for doc in LABELED_CORPUS:
        segments = seg.segment(doc.text)
        predicted = [s.boundary_type for s in segments]
        expected = doc.expected_boundary_types

        for j in range(max(len(predicted), len(expected))):
            pred = predicted[j] if j < len(predicted) else None
            exp = expected[j] if j < len(expected) else None

            for label in ("paragraph", "sentence"):
                pred_match = pred == label
                exp_match = exp == label
                if pred_match and exp_match:
                    stats[label]["tp"] += 1
                elif pred_match and not exp_match:
                    stats[label]["fp"] += 1
                elif not pred_match and exp_match:
                    stats[label]["fn"] += 1

    print("\nBoundary Detection Evaluation (20 documents):")
    for label, s in stats.items():
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        print(f"  {label:<12s}  P: {p:.2f}  R: {r:.2f}  F1: {f1:.2f}")

    captured = capsys.readouterr()
    assert "Boundary Detection Evaluation" in captured.out
    assert "paragraph" in captured.out


def test_segmenter_sentence_count_accuracy() -> None:
    """sentence_count accurate for ≥ 80% of segments vs paragraph-level ground truth."""
    seg = _segmenter()

    _para_split = re.compile(r"\n\s*\n")
    _sent_split = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(])")

    def _count_expected_sentences(para_text: str) -> int:
        parts = _sent_split.split(para_text.strip())
        return max(1, len([p for p in parts if p.strip()]))

    total_compared = 0
    accurate = 0

    for doc in LABELED_CORPUS:
        paragraphs = [p.strip() for p in _para_split.split(doc.text) if p.strip()]
        segments = seg.segment(doc.text)

        for j, s in enumerate(segments):
            if j >= len(paragraphs):
                break
            manual_count = _count_expected_sentences(paragraphs[j])
            total_compared += 1
            if abs(s.sentence_count - manual_count) <= 1:
                accurate += 1

    accuracy_pct = accurate / total_compared * 100 if total_compared > 0 else 0.0
    print(
        f"\nSentence count accuracy: {accurate}/{total_compared} ({accuracy_pct:.0f}%)"
    )
    assert accuracy_pct >= 80.0, (
        f"Sentence count accuracy {accuracy_pct:.1f}% is below 80%"
    )


def test_segmenter_evaluation_summary(capsys: pytest.CaptureFixture[str]) -> None:
    """Print the full evaluation summary table."""
    seg = _segmenter()

    count_hits = sum(
        1
        for doc in LABELED_CORPUS
        if abs(len(seg.segment(doc.text)) - doc.expected_segment_count) <= 1
    )

    tp = fp = fn = 0
    for doc in LABELED_CORPUS:
        segs = seg.segment(doc.text)
        pred = [s.boundary_type for s in segs]
        exp = doc.expected_boundary_types
        for j in range(max(len(pred), len(exp))):
            p_ = pred[j] if j < len(pred) else None
            e_ = exp[j] if j < len(exp) else None
            if p_ == "paragraph" and e_ == "paragraph":
                tp += 1
            elif p_ == "paragraph" and e_ != "paragraph":
                fp += 1
            elif p_ != "paragraph" and e_ == "paragraph":
                fn += 1

    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0

    total = len(LABELED_CORPUS)
    print("\nBoundary Detection Evaluation (20 documents):")
    print(
        f"  Segment count accuracy: {count_hits}/{total}"
        f" ({count_hits / total * 100:.0f}%)"
    )
    print(f"  Paragraph boundary P: {prec:.2f}  R: {rec:.2f}  F1: {f1:.2f}")

    captured = capsys.readouterr()
    assert "Boundary Detection Evaluation" in captured.out
    assert "Segment count accuracy" in captured.out
