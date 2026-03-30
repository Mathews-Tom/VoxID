from __future__ import annotations

import logging
from typing import Any

from .language_config import LanguageConfig, get_language_config

logger = logging.getLogger(__name__)

# IPA phoneme category weights for speaker-discriminative enrollment.
# Nasals and affricates carry strongest speaker signatures (1.5x),
# vowels encode unique formant structure (1.2x), fricatives and plosives
# are neutral (1.0x), approximants are weaker (0.8x).
IPA_PHONEME_WEIGHTS: dict[str, float] = {
    "nasal": 1.5,
    "affricate": 1.5,
    "vowel": 1.2,
    "plosive": 1.0,
    "fricative": 1.0,
    "approximant": 0.8,
}


def _phoneme_weight(phoneme: str, lang: LanguageConfig) -> float:
    """Get speaker-discriminative weight for a phoneme in a language."""
    if phoneme in lang.nasals:
        return IPA_PHONEME_WEIGHTS["nasal"]
    if phoneme in lang.affricates:
        return IPA_PHONEME_WEIGHTS["affricate"]
    if phoneme in lang.vowels:
        return IPA_PHONEME_WEIGHTS["vowel"]
    if phoneme in lang.plosives:
        return IPA_PHONEME_WEIGHTS["plosive"]
    if phoneme in lang.fricatives:
        return IPA_PHONEME_WEIGHTS["fricative"]
    if phoneme in lang.approximants:
        return IPA_PHONEME_WEIGHTS["approximant"]
    return 1.0


class UniversalPhonemeTracker:
    """Language-agnostic phoneme coverage tracker using IPA inventories.

    Unlike the English-only PhonemeTracker (which uses ARPAbet via CMUdict),
    this tracker works with any language's IPA phoneme inventory. Phonemes
    are weighted by category for speaker-discriminative enrollment.
    """

    def __init__(
        self,
        language: str,
        target_per_phoneme: int = 2,
    ) -> None:
        self._lang = get_language_config(language)
        self._target = target_per_phoneme
        self._coverage: dict[str, int] = {
            p: 0 for p in sorted(self._lang.all_phonemes)
        }

    @property
    def language(self) -> str:
        return self._lang.code

    @property
    def language_name(self) -> str:
        return self._lang.name

    @property
    def target(self) -> int:
        return self._target

    @property
    def lang_config(self) -> LanguageConfig:
        return self._lang

    def ingest_phonemes(self, phonemes: list[str]) -> None:
        """Update coverage from a list of IPA phonemes."""
        for p in phonemes:
            if p in self._coverage:
                self._coverage[p] += 1

    def weight_for(self, phoneme: str) -> float:
        """Get the speaker-discriminative weight for a phoneme."""
        return _phoneme_weight(phoneme, self._lang)

    def gap_vector(self) -> dict[str, int]:
        """Return {phoneme: deficit} for phonemes below target."""
        return {
            p: self._target - count
            for p, count in self._coverage.items()
            if count < self._target
        }

    def weighted_gap(self) -> dict[str, float]:
        """Return {phoneme: weighted_deficit} for phonemes below target."""
        return {
            p: (self._target - count) * self.weight_for(p)
            for p, count in self._coverage.items()
            if count < self._target
        }

    def coverage_percent(self) -> float:
        """Percentage of phonemes that have met or exceeded target."""
        if not self._coverage:
            return 100.0
        covered = sum(
            1 for count in self._coverage.values()
            if count >= self._target
        )
        return covered / len(self._coverage) * 100

    def coverage_report(self) -> dict[str, int]:
        """Return full coverage counts for all phonemes."""
        return dict(self._coverage)

    def is_complete(self) -> bool:
        """True when every phoneme has met or exceeded target."""
        return all(
            count >= self._target for count in self._coverage.values()
        )

    def marginal_gain(self, phonemes: list[str]) -> float:
        """Compute weighted marginal gain of adding these phonemes."""
        gain = 0.0
        for p in phonemes:
            if p not in self._coverage:
                continue
            deficit = self._target - self._coverage.get(p, 0)
            if deficit > 0:
                gain += self.weight_for(p)
        return gain

    def to_dict(self) -> dict[str, Any]:
        return {
            "language": self._lang.code,
            "target_per_phoneme": self._target,
            "coverage": dict(self._coverage),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> UniversalPhonemeTracker:
        tracker = cls(
            language=data["language"],
            target_per_phoneme=data["target_per_phoneme"],
        )
        for phoneme, count in data["coverage"].items():
            if phoneme in tracker._coverage:
                tracker._coverage[phoneme] = int(count)
        return tracker
