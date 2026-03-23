from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .phoneme_tracker import (
    ALL_PHONEMES,
    PHONEME_WEIGHTS,
    PhonemeTracker,
    text_to_phonemes,
)

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent / "prompts"

_STYLE_TO_FILE: dict[str, str] = {
    "phonetic": "en_phonetic.json",
    "conversational": "en_conversational.json",
    "technical": "en_technical.json",
    "narration": "en_narration.json",
    "emphatic": "en_emphatic.json",
}


@dataclass(frozen=True)
class EnrollmentPrompt:
    text: str
    style: str
    phonemes: list[str]
    unique_phoneme_count: int
    nasal_count: int
    affricate_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "style": self.style,
            "phonemes": self.phonemes,
            "unique_phoneme_count": self.unique_phoneme_count,
            "nasal_count": self.nasal_count,
            "affricate_count": self.affricate_count,
        }


_NASALS = frozenset({"N", "M", "NG"})
_AFFRICATES = frozenset({"CH", "JH"})


def _build_prompt(text: str, style: str) -> EnrollmentPrompt:
    phonemes = text_to_phonemes(text)
    return EnrollmentPrompt(
        text=text,
        style=style,
        phonemes=phonemes,
        unique_phoneme_count=len(set(phonemes) & ALL_PHONEMES),
        nasal_count=sum(1 for p in phonemes if p in _NASALS),
        affricate_count=sum(1 for p in phonemes if p in _AFFRICATES),
    )


def _marginal_gain(
    phonemes: list[str],
    current: dict[str, int],
    target: int,
) -> float:
    """Compute weighted marginal gain of adding these phonemes."""
    gain = 0.0
    for p in phonemes:
        if p not in ALL_PHONEMES:
            continue
        deficit = target - current.get(p, 0)
        if deficit > 0:
            gain += PHONEME_WEIGHTS[p]
    return gain


class ScriptGenerator:
    """Generates enrollment scripts using greedy phoneme coverage selection.

    Loads style-tagged sentences from bundled JSON corpora and selects
    prompts that maximize weighted phoneme coverage.
    """

    def __init__(self, corpus_path: Path | None = None) -> None:
        self._corpus_path = corpus_path or _PROMPTS_DIR
        self._corpus_cache: dict[str, list[dict[str, Any]]] = {}

    def _load_corpus(self, style: str) -> list[dict[str, Any]]:
        if style in self._corpus_cache:
            return self._corpus_cache[style]

        filename = _STYLE_TO_FILE.get(style)
        if filename is None:
            raise ValueError(
                f"Unknown style '{style}'. "
                f"Available: {sorted(_STYLE_TO_FILE)}"
            )

        path = self._corpus_path / filename
        with open(path) as f:
            entries: list[dict[str, Any]] = json.load(f)
        self._corpus_cache[style] = entries
        return entries

    def _candidates(
        self,
        style: str,
        exclude_texts: set[str],
    ) -> list[str]:
        entries = self._load_corpus(style)
        return [
            e["text"]
            for e in entries
            if e["text"] not in exclude_texts
        ]

    def select_prompts(
        self,
        style: str,
        count: int = 5,
        target_per_phoneme: int = 2,
        exclude_texts: list[str] | None = None,
    ) -> list[EnrollmentPrompt]:
        """Select prompts using greedy weighted phoneme coverage.

        Algorithm (Bozkurt et al., Eurospeech 2003):
        1. Initialize empty coverage
        2. For each slot, score all candidates by weighted marginal gain
        3. Select the candidate with highest gain
        4. Update coverage, remove selected from candidates
        """
        excluded = set(exclude_texts) if exclude_texts else set()
        candidates = self._candidates(style, excluded)
        coverage: dict[str, int] = {p: 0 for p in ALL_PHONEMES}
        selected: list[EnrollmentPrompt] = []

        for _ in range(min(count, len(candidates))):
            if not candidates:
                break

            best_idx = -1
            best_gain = -1.0

            for idx, text in enumerate(candidates):
                phonemes = text_to_phonemes(text)
                gain = _marginal_gain(
                    phonemes, coverage, target_per_phoneme,
                )
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx < 0:
                break

            chosen_text = candidates.pop(best_idx)
            prompt = _build_prompt(chosen_text, style)
            selected.append(prompt)

            for p in prompt.phonemes:
                if p in coverage:
                    coverage[p] += 1

        return selected

    def select_next_adaptive(
        self,
        style: str,
        tracker: PhonemeTracker,
        exclude_texts: list[str] | None = None,
    ) -> EnrollmentPrompt | None:
        """Select the single best next prompt to fill coverage gaps.

        Returns None when the tracker reports full coverage.
        """
        if tracker.is_complete():
            return None

        excluded = set(exclude_texts) if exclude_texts else set()
        candidates = self._candidates(style, excluded)
        if not candidates:
            return None

        current = tracker.coverage_report()
        target = tracker.target

        best_text: str | None = None
        best_gain = -1.0

        for text in candidates:
            phonemes = text_to_phonemes(text)
            gain = _marginal_gain(phonemes, current, target)
            if gain > best_gain:
                best_gain = gain
                best_text = text

        if best_text is None:
            return None

        return _build_prompt(best_text, style)
