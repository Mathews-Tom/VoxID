from __future__ import annotations

from voxid.video.timing import WordTiming, estimate_word_timings, timings_to_tuples


def test_estimate_timings_empty_text() -> None:
    # Arrange / Act
    result = estimate_word_timings("", 1000)

    # Assert
    assert result == []


def test_estimate_timings_single_word() -> None:
    # Arrange / Act
    result = estimate_word_timings("Hello", 1000)

    # Assert
    assert len(result) == 1
    assert result[0].word == "Hello"
    assert result[0].start_ms == 0
    assert result[0].end_ms == 1000


def test_estimate_timings_multiple_words() -> None:
    # Arrange / Act
    result = estimate_word_timings("Hello world test", 3000)

    # Assert
    assert len(result) == 3
    assert result[0].word == "Hello"
    assert result[1].word == "world"
    assert result[2].word == "test"


def test_estimate_timings_sequential_no_overlap() -> None:
    # Arrange / Act
    result = estimate_word_timings("Hello world test", 3000)

    # Assert — each word starts after the previous one ends
    for i in range(1, len(result)):
        assert result[i].start_ms >= result[i - 1].end_ms


def test_estimate_timings_covers_full_duration() -> None:
    # Arrange / Act
    result = estimate_word_timings("Hello world test", 3000)

    # Assert
    assert result[-1].end_ms == 3000


def test_estimate_timings_no_overlap() -> None:
    # Arrange / Act
    result = estimate_word_timings("one two three four five", 5000)

    # Assert — gaps are allowed but end of word i must be <= start of word i+1
    for i in range(len(result) - 1):
        assert result[i].end_ms <= result[i + 1].start_ms


def test_estimate_timings_proportional() -> None:
    # Arrange — "encyclopedia" (12 chars) vs "a" (1 char) with enough duration
    result = estimate_word_timings("encyclopedia a", 5000)

    # Act
    encyclopedia_ms = result[0].end_ms - result[0].start_ms
    a_ms = result[1].end_ms - result[1].start_ms

    # Assert — longer word gets more time
    assert encyclopedia_ms > a_ms


def test_estimate_timings_minimum_duration() -> None:
    # Arrange — use enough duration so per-word allocation exceeds the 20ms floor.
    # The floor applies to the computed speech slice; the last word's end_ms is
    # pinned to duration_ms, so only non-final words are checked.
    result = estimate_word_timings("a b c d e", 1000)

    # Assert — every word except the last gets at least 20ms
    for timing in result[:-1]:
        assert (timing.end_ms - timing.start_ms) >= 20
    # Last word always reaches duration_ms; just verify it's positive
    assert result[-1].end_ms > result[-1].start_ms


def test_timings_to_tuples_format() -> None:
    # Arrange
    timings = [
        WordTiming(word="Hello", start_ms=0, end_ms=400),
        WordTiming(word="world", start_ms=450, end_ms=900),
    ]

    # Act
    result = timings_to_tuples(timings)

    # Assert
    assert result == [("Hello", 0, 400), ("world", 450, 900)]
    assert all(isinstance(t, tuple) and len(t) == 3 for t in result)
    assert all(
        isinstance(word, str) and isinstance(start, int) and isinstance(end, int)
        for word, start, end in result
    )
