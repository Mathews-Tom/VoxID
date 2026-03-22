from __future__ import annotations

from pathlib import Path

import pytest

from voxid.router import StyleRouter
from voxid.segments.segmenter import TextSegmenter
from voxid.segments.smoother import StyleSmoother

# ---------------------------------------------------------------------------
# 20 diverse multi-paragraph evaluation documents
# ---------------------------------------------------------------------------

# 1: engineering update (tech → conversational → tech)
_D01 = (
    "Welcome to this week's engineering update. We shipped three major changes.\n\n"
    "Let's start with the retrieval pipeline. "
    "The new vector index cut p99 latency from 340ms to 80ms.\n\n"
    "Honestly, this one was a grind. "
    "We hit three separate race conditions before landing on the fix.\n\n"
    "The final diff was surprisingly small. "
    "Sometimes the hardest bugs have the simplest solutions."
)

# 2: product announcement (emphatic → technical → narration)
_D02 = (
    "This changes everything! We're launching the most ambitious feature yet.\n\n"
    "The architecture uses a transformer backbone fine-tuned on domain data. "
    "Inference runs on a single A10 GPU with 4-bit quantisation.\n\n"
    "Looking back on the journey that led us here, it's hard to believe "
    "it started with a weekend hackathon."
)

# 3: personal reflection on learning
_D03 = (
    "I used to think that talent was fixed. "
    "Either you had it or you didn't.\n\n"
    "That belief held me back for years. "
    "Every failure felt like evidence of my inadequacy.\n\n"
    "Then I read research on neuroplasticity. "
    "The brain physically rewires itself through practice.\n\n"
    "Now I approach every hard problem differently. "
    "Struggle is the signal that learning is happening."
)

# 4: tutorial — setting up a Python project
_D04 = (
    "Start by installing uv, the fast Python package manager. "
    "Run the install script from the Astral website.\n\n"
    "Create a new project with `uv init myproject`. "
    "This generates a pyproject.toml and a virtual environment.\n\n"
    "Add your first dependency using `uv add requests`. "
    "The lock file updates automatically.\n\n"
    "You are now ready to write your first module. "
    "Create src/myproject/main.py and start coding."
)

# 5: mixed formal announcement and technical detail
_D05 = (
    "We are pleased to announce the general availability of VoxID 2.0.\n\n"
    "The release includes a completely rewritten routing engine. "
    "Rule-based classification now runs in under one millisecond.\n\n"
    "Upgrade by running `uv add voxid@2.0`. "
    "Configuration files from version 1.x are fully compatible."
)

# 6: conversational blog post on productivity
_D06 = (
    "I've tried every productivity system out there. "
    "Pomodoro, GTD, time-blocking — you name it.\n\n"
    "They all work, sort of, for a while. "
    "Then life happens and the system collapses.\n\n"
    "What actually stuck was simpler: write down three things to do today, "
    "do them, stop.\n\n"
    "Turns out constraint is more powerful than elaborate structure."
)

# 7: technical deep-dive on databases
_D07 = (
    "Relational databases enforce ACID guarantees through write-ahead logging "
    "and locking.\n\n"
    "When a transaction writes a row, the change is first recorded in the WAL. "
    "The data page is updated later during a checkpoint.\n\n"
    "If the process crashes before the checkpoint, the WAL allows recovery to "
    "a consistent state. This is called redo recovery.\n\n"
    "Isolation levels control how much concurrent transactions can see of "
    "each other's uncommitted writes."
)

# 8: product reflection — narration style
_D08 = (
    "Three years ago we launched with twelve paying customers and a prayer.\n\n"
    "The first six months were brutal. "
    "Churn was high, support tickets never stopped, and the team was exhausted.\n\n"
    "Then something shifted. Word of mouth started working. "
    "A single tweet from an influencer brought five hundred signups overnight.\n\n"
    "Today we serve forty thousand teams across sixty countries. "
    "The journey has been anything but linear."
)

# 9: emphatic motivational content
_D09 = (
    "You are capable of more than you think. "
    "Every expert was once a complete beginner.\n\n"
    "Stop waiting for the perfect moment. "
    "It does not exist. "
    "The best time to start was yesterday, the second best is right now.\n\n"
    "Failure is not the opposite of success — it is part of the path."
)

# 10: technical tutorial on Docker
_D10 = (
    "Docker containers package your application and all its dependencies "
    "into a single portable unit.\n\n"
    "Start with a Dockerfile in your project root. "
    "The FROM instruction selects the base image.\n\n"
    "Use COPY to bring your source files into the image. "
    "Run RUN instructions to install dependencies.\n\n"
    "Build the image with docker build -t myapp:latest. "
    "Run it with docker run -p 8080:8080 myapp:latest."
)

# 11: conversational career advice
_D11 = (
    "Nobody tells you that most of the work in a senior engineering role "
    "is communication, not coding.\n\n"
    "You spend more time in design reviews, writing docs, and unblocking "
    "colleagues than you do writing code.\n\n"
    "That is not a complaint — it is a shift in leverage. "
    "One clear design document can save a week of back-and-forth."
)

# 12: science explainer — narration style
_D12 = (
    "Black holes form when a massive star exhausts its nuclear fuel. "
    "The core collapses under its own gravity.\n\n"
    "If the remaining mass exceeds about three solar masses, the collapse "
    "cannot be stopped. A singularity forms — a point of infinite density.\n\n"
    "Around the singularity lies the event horizon. "
    "No information, not even light, can escape once it crosses this boundary.\n\n"
    "Stephen Hawking showed that black holes slowly radiate energy "
    "via quantum effects over astronomical timescales."
)

# 13: tech blog — opinionated take
_D13 = (
    "Microservices are not always the right answer. "
    "For most startups, a well-structured monolith is faster to build.\n\n"
    "The overhead of service discovery, distributed tracing, and "
    "inter-service auth adds up fast.\n\n"
    "Split services when you have a clear operational boundary, "
    "not because it feels architecturally pure."
)

# 14: personal story mixed with technical observation
_D14 = (
    "The first time I deployed to production I accidentally deleted the database. "
    "It was a Tuesday.\n\n"
    "Backups saved us, but it took six hours to restore. "
    "I learned more about disaster recovery in those six hours than in any course.\n\n"
    "Now I run chaos engineering drills quarterly. "
    "The team that rehearses failure handles it calmly when it arrives."
)

# 15: instructional — formal tone
_D15 = (
    "Effective code review requires understanding its purpose. "
    "The goal is to improve code quality, not to prove superiority.\n\n"
    "Read the change in its entirety before leaving any comments. "
    "Understanding the full scope prevents piecemeal feedback.\n\n"
    "Distinguish between blocking issues and non-blocking suggestions. "
    "Mark non-blocking comments clearly to reduce friction."
)

# 16: reflective essay on open source
_D16 = (
    "Open source software underpins nearly all modern infrastructure. "
    "Most of it is maintained by a handful of unpaid volunteers.\n\n"
    "This creates fragility that the industry rarely acknowledges. "
    "A single burned-out maintainer can leave critical projects unmaintained.\n\n"
    "Sustainable funding models are emerging: GitHub Sponsors, "
    "Open Collective, and corporate backing. But adoption is uneven."
)

# 17: tutorial — machine learning workflow
_D17 = (
    "A typical supervised learning project follows a predictable sequence.\n\n"
    "First, gather and label your training data. "
    "Data quality determines the ceiling of model performance.\n\n"
    "Second, split your data into train, validation, and test sets. "
    "Never touch the test set until final evaluation.\n\n"
    "Third, select a model architecture and train it. "
    "Monitor validation loss to detect overfitting early."
)

# 18: conversational — debugging story
_D18 = (
    "Spent four hours yesterday chasing a bug that turned out to be "
    "a timezone issue. Classic.\n\n"
    "The timestamps in the database were stored as naive datetimes, "
    "assumed to be UTC. The frontend was in the user's local timezone.\n\n"
    "Off-by-one-hour errors only appeared during daylight saving transitions. "
    "No wonder it took so long to reproduce.\n\n"
    "Always store timestamps in UTC. Always. This is not negotiable."
)

# 19: formal technical — API design
_D19 = (
    "A well-designed REST API is consistent, predictable, and version-aware.\n\n"
    "Resource names should be nouns in the plural: /users, /orders, /products. "
    "Verbs belong to HTTP methods.\n\n"
    "Use HTTP status codes semantically. "
    "Return 201 for resource creation, 204 for successful deletion.\n\n"
    "Version your API from day one. "
    "Adding /v1/ to the URL path is simple and explicit."
)

# 20: mixed — company culture
_D20 = (
    "Psychological safety is the single strongest predictor of team effectiveness, "
    "according to Google's Project Aristotle.\n\n"
    "Teams where members feel safe to speak up outperform teams that do not, "
    "even when individual talent is lower.\n\n"
    "Creating safety is a leadership responsibility. "
    "It requires consistent behaviour over time, not a one-off team exercise.\n\n"
    "Start small: respond to bad news without blame. "
    "Ask questions before offering solutions."
)

EVAL_DOCUMENTS: list[str] = [
    _D01, _D02, _D03, _D04, _D05,
    _D06, _D07, _D08, _D09, _D10,
    _D11, _D12, _D13, _D14, _D15,
    _D16, _D17, _D18, _D19, _D20,
]

assert len(EVAL_DOCUMENTS) == 20, "Must have exactly 20 evaluation documents"

_STYLES = ["conversational", "technical", "narration", "emphatic"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pipeline(
    doc: str,
    router: StyleRouter,
) -> tuple[list[int], list[str], list[str]]:
    """Segment → route → smooth.

    Returns (sentence_counts, raw_styles, smoothed_styles).
    """
    segmenter = TextSegmenter()
    smoother = StyleSmoother()

    segments = segmenter.segment(doc)
    if not segments:
        return [], [], []

    decisions = [
        router.route(s.text, _STYLES, "conversational") for s in segments
    ]
    sentence_counts = [s.sentence_count for s in segments]
    smoothed = smoother.smooth(decisions, sentence_counts)

    raw_styles = [d.style for d in decisions]
    smooth_styles = [sd.style for sd in smoothed]
    return sentence_counts, raw_styles, smooth_styles


def _count_switches(styles: list[str]) -> int:
    return sum(1 for i in range(1, len(styles)) if styles[i] != styles[i - 1])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_smooth_no_switch_within_two_sentences_20_docs(
    tmp_path: Path,
) -> None:
    """No style switch allowed on a segment with < 2 sentences."""
    router = StyleRouter(cache_dir=tmp_path / "cache")
    smoother = StyleSmoother()
    segmenter = TextSegmenter()

    violations: list[str] = []

    for doc_idx, doc in enumerate(EVAL_DOCUMENTS):
        segments = segmenter.segment(doc)
        if not segments:
            continue

        decisions = [
            router.route(s.text, _STYLES, "conversational") for s in segments
        ]
        sentence_counts = [s.sentence_count for s in segments]
        smoothed = smoother.smooth(decisions, sentence_counts)

        for i in range(1, len(smoothed)):
            seg_count = sentence_counts[i]
            prev_style = smoothed[i - 1].style
            curr_style = smoothed[i].style
            is_switch = curr_style != prev_style
            if is_switch and seg_count < 2:
                violations.append(
                    f"doc {doc_idx}, seg {i}: "
                    f"switch {prev_style!r}→{curr_style!r} "
                    f"on segment with {seg_count} sentence(s)"
                )

    assert not violations, (
        f"Found {len(violations)} style switch(es) within <2-sentence window:\n"
        + "\n".join(violations)
    )

    router.close()


def test_smooth_reduces_total_switches(tmp_path: Path) -> None:
    """Smoothed switch count ≤ raw switch count across all 20 documents."""
    router = StyleRouter(cache_dir=tmp_path / "cache")

    raw_total = 0
    smoothed_total = 0

    for doc in EVAL_DOCUMENTS:
        sentence_counts, raw_styles, smooth_styles = _pipeline(doc, router)
        raw_total += _count_switches(raw_styles)
        smoothed_total += _count_switches(smooth_styles)

    print(
        f"\nSwitch reduction: raw={raw_total}, smoothed={smoothed_total}, "
        f"reduction={raw_total - smoothed_total}"
    )

    assert smoothed_total <= raw_total, (
        f"Smoothed switches ({smoothed_total}) exceeded raw switches ({raw_total})"
    )

    router.close()


def test_smooth_evaluation_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Print summary: total segments, switches before/after, switch reduction %."""
    router = StyleRouter(cache_dir=tmp_path / "cache")

    total_segments = 0
    raw_total = 0
    smoothed_total = 0

    for doc in EVAL_DOCUMENTS:
        sentence_counts, raw_styles, smooth_styles = _pipeline(doc, router)
        total_segments += len(sentence_counts)
        raw_total += _count_switches(raw_styles)
        smoothed_total += _count_switches(smooth_styles)

    reduction_pct = (
        (raw_total - smoothed_total) / raw_total * 100 if raw_total > 0 else 0.0
    )

    print("\nSmoothing Evaluation (20 documents):")
    print(f"  Total segments:    {total_segments}")
    print(f"  Switches (raw):    {raw_total}")
    print(f"  Switches (smooth): {smoothed_total}")
    print(f"  Switch reduction:  {reduction_pct:.1f}%")

    captured = capsys.readouterr()
    assert "Smoothing Evaluation" in captured.out
    assert "Total segments" in captured.out
    assert "Switch reduction" in captured.out

    router.close()
