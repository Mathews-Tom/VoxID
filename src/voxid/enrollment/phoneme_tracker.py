from __future__ import annotations

import logging
import re
from typing import Any

import cmudict

logger = logging.getLogger(__name__)

# 39 ARPAbet phonemes (stress-stripped)
ALL_PHONEMES: frozenset[str] = frozenset({
    # Vowels (15)
    "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
    "IH", "IY", "OW", "OY", "UH", "UW",
    # Consonants (24)
    "B", "CH", "D", "DH", "F", "G", "HH", "JH", "K", "L",
    "M", "N", "NG", "P", "R", "S", "SH", "T", "TH", "V",
    "W", "Y", "Z", "ZH",
})

# Weighted importance for speaker-identifying phonemes
# Nasals and affricates carry strongest speaker signatures (1.5x)
# Vowels encode unique formant structure (1.2x)
PHONEME_WEIGHTS: dict[str, float] = {p: 1.0 for p in ALL_PHONEMES}
for _p in ("N", "M", "NG"):
    PHONEME_WEIGHTS[_p] = 1.5
for _p in ("CH", "JH"):
    PHONEME_WEIGHTS[_p] = 1.5
for _p in ("AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
           "IH", "IY", "OW", "OY", "UH", "UW"):
    PHONEME_WEIGHTS[_p] = 1.2

_WORD_RE = re.compile(r"[a-zA-Z]+(?:'[a-zA-Z]+)?")
_STRESS_RE = re.compile(r"[0-9]")

_CmuDict = dict[str, list[list[str]]]

_cmudict_cache: _CmuDict | None = None


def _strip_stress(phoneme: str) -> str:
    return _STRESS_RE.sub("", phoneme)


def load_cmudict() -> _CmuDict:
    """Load CMUdict and return {word: [[phoneme, ...]]} with stress stripped."""
    global _cmudict_cache
    if _cmudict_cache is not None:
        return _cmudict_cache

    raw = cmudict.dict()
    result: _CmuDict = {}
    for word, pronunciations in raw.items():
        stripped = [
            [_strip_stress(p) for p in pron]
            for pron in pronunciations
        ]
        result[word] = stripped
    _cmudict_cache = result
    return result


def text_to_phonemes(text: str, cmu: _CmuDict | None = None) -> list[str]:
    """Tokenize text, look up each word in CMUdict, return flat phoneme list.

    Unknown words are logged and skipped (not an error).
    Uses the first pronunciation variant for each word.
    """
    if cmu is None:
        cmu = load_cmudict()

    words = _WORD_RE.findall(text.lower())
    phonemes: list[str] = []
    for word in words:
        pronunciations = cmu.get(word)
        if pronunciations is None:
            logger.debug("OOV word: %s", word)
            continue
        phonemes.extend(pronunciations[0])
    return phonemes


class PhonemeTracker:
    """Tracks phoneme coverage across ingested texts.

    Used during enrollment to ensure all 39 ARPAbet phonemes
    are adequately represented in the recorded samples.
    """

    def __init__(self, target_per_phoneme: int = 2) -> None:
        self._target = target_per_phoneme
        self._coverage: dict[str, int] = {p: 0 for p in ALL_PHONEMES}

    @property
    def target(self) -> int:
        return self._target

    def ingest(self, text: str) -> None:
        """Update coverage counts from a text's phonemes."""
        phonemes = text_to_phonemes(text)
        for p in phonemes:
            if p in self._coverage:
                self._coverage[p] += 1

    def gap_vector(self) -> dict[str, int]:
        """Return {phoneme: deficit} for phonemes below target."""
        return {
            p: self._target - count
            for p, count in self._coverage.items()
            if count < self._target
        }

    def coverage_percent(self) -> float:
        """Percentage of phonemes that have met or exceeded target."""
        covered = sum(1 for count in self._coverage.values() if count >= self._target)
        return covered / len(ALL_PHONEMES) * 100

    def coverage_report(self) -> dict[str, int]:
        """Return full coverage counts for all phonemes."""
        return dict(self._coverage)

    def is_complete(self) -> bool:
        """True when every phoneme has met or exceeded target."""
        return all(count >= self._target for count in self._coverage.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_per_phoneme": self._target,
            "coverage": dict(self._coverage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PhonemeTracker:
        tracker = cls(target_per_phoneme=data["target_per_phoneme"])
        for phoneme, count in data["coverage"].items():
            if phoneme in tracker._coverage:
                tracker._coverage[phoneme] = int(count)
        return tracker
