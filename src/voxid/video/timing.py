from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WordTiming:
    word: str
    start_ms: int
    end_ms: int


def estimate_word_timings(
    text: str,
    duration_ms: int,
) -> list[WordTiming]:
    """Estimate word-level timing by proportional character count.

    Allocates duration proportional to each word's character length
    relative to total text length. Adds small gaps between words.

    This is an approximation. For production accuracy, use
    NeMo Forced Aligner or ForceAlign with actual audio.
    """
    words = text.split()
    if not words:
        return []

    # Reserve 5% of duration for inter-word gaps
    gap_ratio = 0.05
    speech_duration = int(duration_ms * (1 - gap_ratio))
    total_gap = duration_ms - speech_duration
    gap_per_word = (
        total_gap // (len(words) - 1) if len(words) > 1 else 0
    )

    total_chars = sum(len(w) for w in words)
    if total_chars == 0:
        return []

    timings: list[WordTiming] = []
    current_ms = 0

    for i, word in enumerate(words):
        word_duration = int(
            speech_duration * len(word) / total_chars
        )
        # Minimum 20ms per word
        word_duration = max(20, word_duration)

        start = current_ms
        end = start + word_duration
        timings.append(WordTiming(
            word=word, start_ms=start, end_ms=end,
        ))

        current_ms = end
        if i < len(words) - 1:
            current_ms += gap_per_word

    # Adjust last word to fill remaining duration
    if timings:
        timings[-1] = WordTiming(
            word=timings[-1].word,
            start_ms=timings[-1].start_ms,
            end_ms=duration_ms,
        )

    return timings


def timings_to_tuples(
    timings: list[WordTiming],
) -> list[tuple[str, int, int]]:
    """Convert WordTiming list to tuple format for GeneratedScene."""
    return [(t.word, t.start_ms, t.end_ms) for t in timings]
