from __future__ import annotations

from voxid.segments.segmenter import TextSegment, TextSegmenter

BLOG_POST = (  # noqa: E501
    "Welcome to this week's engineering update.\n"
    "\n"
    "Let's start with the retrieval pipeline. We migrated from FAISS to"
    " pgvector, reducing cold-start latency from 340ms to 89ms. The"
    " embedding model is now BGE-M3 running on a dedicated inference pod"
    " with 4-bit quantization.\n"
    "\n"
    "Honestly, this one was a grind. Three false starts before we found"
    " the right configuration. But the numbers speak for themselves.\n"
    "\n"
    "The second big change was the authentication overhaul. We moved from"
    " session-based auth to JWT with refresh token rotation. Every request"
    " now carries a signed token with a 15-minute expiry. The refresh flow"
    " uses httpOnly cookies to prevent XSS token theft.\n"
    "\n"
    "Looking ahead, we're planning a major refactor of the notification"
    " system. The current implementation uses polling, which doesn't scale"
    " beyond a few hundred concurrent users. We're evaluating WebSocket"
    " and SSE as alternatives."
)


def test_segment_empty_text_returns_empty_list():
    # Arrange
    segmenter = TextSegmenter()

    # Act
    result = segmenter.segment("")

    # Assert
    assert result == []


def test_segment_single_sentence():
    # Arrange
    segmenter = TextSegmenter()

    # Act
    result = segmenter.segment("Hello world.")

    # Assert
    assert len(result) == 1
    assert isinstance(result[0], TextSegment)


def test_segment_two_paragraphs():
    # Arrange — each paragraph must exceed min_words=5 to avoid merging
    segmenter = TextSegmenter()
    text = (
        "First paragraph has plenty of words here.\n\n"
        "Second paragraph also has enough words in it."
    )

    # Act
    result = segmenter.segment(text)

    # Assert
    assert len(result) == 2
    assert result[1].boundary_type == "paragraph"


def test_segment_multiple_sentences_one_paragraph():
    # Arrange
    segmenter = TextSegmenter()
    text = "First sentence. Second sentence. Third sentence."

    # Act
    result = segmenter.segment(text)

    # Assert
    assert len(result) == 1
    assert result[0].sentence_count == 3


def test_segment_respects_max_sentences():
    # Arrange
    segmenter = TextSegmenter(max_sentences=3)
    sentences = " ".join(
        f"This is sentence number {i}." for i in range(1, 11)
    )

    # Act
    result = segmenter.segment(sentences)

    # Assert
    assert len(result) > 1
    for seg in result:
        assert seg.sentence_count <= 3


def test_segment_merges_short_segments():
    # Arrange — longer paragraph followed by a short one (2 words).
    # The merge logic appends short segments to the PREVIOUS segment, so
    # the short one must come after a longer one to trigger merging.
    segmenter = TextSegmenter(min_words=5)
    text = (
        "This is a longer paragraph with plenty of content and sentences.\n\n"
        "Short bit."
    )

    # Act
    result = segmenter.segment(text)

    # Assert — short trailing segment is merged into previous → 1 total
    assert len(result) == 1


def test_segment_preserves_abbreviations():
    # Arrange
    segmenter = TextSegmenter()
    text = "Dr. Smith went to the store. Mr. Jones stayed home."

    # Act
    result = segmenter.segment(text)

    # Assert — should be 2 sentences total, not split at abbreviations
    total_sentences = sum(s.sentence_count for s in result)
    assert total_sentences == 2


def test_segment_handles_question_marks():
    # Arrange
    segmenter = TextSegmenter()
    text = "What is this? It's a test."

    # Act
    result = segmenter.segment(text)

    # Assert
    total_sentences = sum(s.sentence_count for s in result)
    assert total_sentences == 2


def test_segment_handles_exclamation_marks():
    # Arrange
    segmenter = TextSegmenter()
    text = "Wow! That's amazing."

    # Act
    result = segmenter.segment(text)

    # Assert
    total_sentences = sum(s.sentence_count for s in result)
    assert total_sentences == 2


def test_segment_sentence_count_accurate():
    # Arrange
    segmenter = TextSegmenter(max_sentences=10)
    text = "One. Two. Three. Four. Five."

    # Act
    result = segmenter.segment(text)

    # Assert
    total_counted = sum(s.sentence_count for s in result)
    assert total_counted == 5


def test_segment_index_sequential():
    # Arrange
    segmenter = TextSegmenter(max_sentences=2)
    text = (
        "Sentence one. Sentence two.\n\n"
        "Sentence three. Sentence four.\n\n"
        "Sentence five. Sentence six."
    )

    # Act
    result = segmenter.segment(text)

    # Assert
    for expected_idx, seg in enumerate(result):
        assert seg.index == expected_idx


def test_segment_long_blog_post():
    # Arrange
    segmenter = TextSegmenter()

    # Act
    result = segmenter.segment(BLOG_POST)

    # Assert — multi-paragraph post produces multiple segments
    assert len(result) > 2
    # All segments carry text
    for seg in result:
        assert seg.text.strip()
    # Indices are sequential
    for expected_idx, seg in enumerate(result):
        assert seg.index == expected_idx
    # At least one paragraph boundary present
    boundary_types = {s.boundary_type for s in result}
    assert "paragraph" in boundary_types
