from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class TextSegment:
    text: str
    index: int
    boundary_type: str  # "paragraph" | "sentence" | "clause"
    sentence_count: int


_PARA_SPLIT = re.compile(r"\n\s*\n")
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z\"'\(])")
_ABBREV = re.compile(
    r"\b(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|etc|e\.g|i\.e|vs|approx)\."
)
_CLAUSE_SPLIT = re.compile(r"(?<=\S)(\s*(?:;| — |: ))")

_ABBREV_PLACEHOLDER = "\x00ABBREV\x00"


class TextSegmenter:
    """Split long-form text into segments at prosodic boundaries.

    Boundary detection hierarchy:
    1. Paragraph breaks (double newline or blank line)
    2. Sentence breaks (period/question/exclamation + space + uppercase)
    3. Long sentences split at clause boundaries (semicolons, em-dashes, colons)

    Constraints:
    - min_sentences: minimum sentences per segment (default 1)
    - max_sentences: maximum sentences per segment before forced split (default 5)
    - min_words: segments shorter than this are merged with the previous (default 5)
    """

    def __init__(
        self,
        min_sentences: int = 1,
        max_sentences: int = 5,
        min_words: int = 5,
    ) -> None:
        self._min_sentences = min_sentences
        self._max_sentences = max_sentences
        self._min_words = min_words

    def segment(self, text: str) -> list[TextSegment]:
        """Split text into segments."""
        text = text.strip()
        if not text:
            return []

        paragraphs = _PARA_SPLIT.split(text)
        raw_segments: list[tuple[str, str]] = []  # (text, boundary_type)

        for para_idx, para in enumerate(paragraphs):
            para = para.strip()
            if not para:
                continue

            sentences = self._split_sentences(para)

            # Group sentences into chunks respecting max_sentences
            groups = self._group_sentences(sentences)

            for group_idx, group in enumerate(groups):
                group_text = " ".join(group).strip()
                if not group_text:
                    continue

                if para_idx > 0 and group_idx == 0:
                    boundary = "paragraph"
                elif group_idx > 0:
                    boundary = "sentence"
                else:
                    boundary = "paragraph" if para_idx == 0 else "paragraph"

                raw_segments.append((group_text, boundary))

        # Merge short segments with previous
        merged = self._merge_short(raw_segments)

        result: list[TextSegment] = []
        for idx, (seg_text, boundary) in enumerate(merged):
            sent_count = self._count_sentences(seg_text)
            result.append(
                TextSegment(
                    text=seg_text,
                    index=idx,
                    boundary_type=boundary,
                    sentence_count=sent_count,
                )
            )

        return result

    def _split_sentences(self, text: str) -> list[str]:
        """Split paragraph text into sentences, respecting abbreviations."""
        # Replace abbreviation periods with placeholder
        protected = _ABBREV.sub(
            lambda m: m.group(0).replace(".", _ABBREV_PLACEHOLDER), text
        )

        parts = _SENT_SPLIT.split(protected)

        sentences: list[str] = []
        for part in parts:
            restored = part.replace(_ABBREV_PLACEHOLDER, ".")
            restored = restored.strip()
            if restored:
                sentences.append(restored)

        if not sentences:
            return [text.strip()] if text.strip() else []

        return sentences

    def _split_at_clauses(self, sentence: str) -> list[str]:
        """Split a long sentence at clause boundaries."""
        parts = _CLAUSE_SPLIT.split(sentence)
        clauses: list[str] = []
        i = 0
        while i < len(parts):
            chunk = parts[i]
            # If next part is the delimiter, append it to the current chunk
            if i + 1 < len(parts) and _CLAUSE_SPLIT.match(parts[i + 1]):
                chunk = chunk + parts[i + 1].rstrip()
                i += 2
            else:
                i += 1
            chunk = chunk.strip()
            if chunk:
                clauses.append(chunk)
        return clauses if clauses else [sentence]

    def _group_sentences(self, sentences: list[str]) -> list[list[str]]:
        """Group sentences into chunks of at most max_sentences."""
        if not sentences:
            return []

        groups: list[list[str]] = []
        current: list[str] = []

        for sent in sentences:
            if len(current) >= self._max_sentences:
                groups.append(current)
                current = []
            current.append(sent)

        if current:
            groups.append(current)

        return groups

    def _count_sentences(self, text: str) -> int:
        """Count sentences in text."""
        sentences = self._split_sentences(text)
        return max(1, len(sentences))

    def _word_count(self, text: str) -> int:
        return len(text.split())

    def _merge_short(
        self, segments: list[tuple[str, str]]
    ) -> list[tuple[str, str]]:
        """Merge segments shorter than min_words into the previous segment."""
        if not segments:
            return []

        result: list[tuple[str, str]] = []
        for seg_text, boundary in segments:
            if result and self._word_count(seg_text) < self._min_words:
                prev_text, prev_boundary = result[-1]
                result[-1] = (prev_text + " " + seg_text, prev_boundary)
            else:
                result.append((seg_text, boundary))

        return result
