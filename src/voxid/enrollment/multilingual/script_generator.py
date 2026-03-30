from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .language_config import LanguageConfig, get_language_config
from .phoneme_universal import UniversalPhonemeTracker

logger = logging.getLogger(__name__)

_CORPORA_DIR = Path(__file__).parent / "corpora"


@dataclass(frozen=True)
class MultilingualPrompt:
    """A prompt entry from a multilingual corpus."""

    text: str
    language: str
    phonemes: list[str]
    unique_phoneme_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "language": self.language,
            "phonemes": self.phonemes,
            "unique_phoneme_count": self.unique_phoneme_count,
        }


class MultilingualScriptGenerator:
    """Generates enrollment scripts for any supported language.

    Uses the same greedy phoneme-gain selection algorithm as the
    English-only ScriptGenerator, but operates on IPA phoneme inventories
    loaded from per-language corpora.

    Corpus format (JSON array):
    [
        {"text": "...", "phonemes": ["p", "a", "ɹ", ...]},
        ...
    ]

    If a corpus is not available for a language, raises ValueError.
    """

    def __init__(self, corpus_path: Path | None = None) -> None:
        self._corpus_path = corpus_path or _CORPORA_DIR
        self._cache: dict[str, list[dict[str, Any]]] = {}

    def _load_corpus(self, lang: LanguageConfig) -> list[dict[str, Any]]:
        if lang.code in self._cache:
            return self._cache[lang.code]

        if lang.corpus_file is None:
            raise ValueError(
                f"No corpus configured for language '{lang.code}' ({lang.name})"
            )

        path = self._corpus_path / lang.corpus_file
        if not path.exists():
            raise FileNotFoundError(
                f"Corpus file not found: {path}. "
                f"Expected for language '{lang.code}' ({lang.name})"
            )

        with open(path, encoding="utf-8") as f:
            entries: list[dict[str, Any]] = json.load(f)

        self._cache[lang.code] = entries
        return entries

    def _candidates(
        self,
        lang: LanguageConfig,
        exclude_texts: set[str],
    ) -> list[dict[str, Any]]:
        entries = self._load_corpus(lang)
        return [e for e in entries if e["text"] not in exclude_texts]

    def select_prompts(
        self,
        language: str,
        count: int = 5,
        target_per_phoneme: int = 2,
        exclude_texts: list[str] | None = None,
    ) -> list[MultilingualPrompt]:
        """Select prompts using greedy weighted phoneme coverage.

        Same algorithm as English ScriptGenerator (Bozkurt et al.,
        Eurospeech 2003), but using IPA inventories and per-language
        category weights.
        """
        lang = get_language_config(language)
        excluded = set(exclude_texts) if exclude_texts else set()
        candidates = self._candidates(lang, excluded)
        tracker = UniversalPhonemeTracker(language, target_per_phoneme)
        selected: list[MultilingualPrompt] = []

        for _ in range(min(count, len(candidates))):
            if not candidates:
                break

            best_idx = -1
            best_gain = -1.0

            for idx, entry in enumerate(candidates):
                phonemes: list[str] = entry["phonemes"]
                gain = tracker.marginal_gain(phonemes)
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx < 0:
                break

            chosen = candidates.pop(best_idx)
            phonemes = chosen["phonemes"]
            prompt = MultilingualPrompt(
                text=chosen["text"],
                language=language,
                phonemes=phonemes,
                unique_phoneme_count=len(
                    set(phonemes) & lang.all_phonemes
                ),
            )
            selected.append(prompt)
            tracker.ingest_phonemes(phonemes)

        return selected

    def select_next_adaptive(
        self,
        language: str,
        tracker: UniversalPhonemeTracker,
        exclude_texts: list[str] | None = None,
    ) -> MultilingualPrompt | None:
        """Select the single best next prompt to fill coverage gaps.

        Returns None when the tracker reports full coverage.
        """
        if tracker.is_complete():
            return None

        lang = get_language_config(language)
        excluded = set(exclude_texts) if exclude_texts else set()
        candidates = self._candidates(lang, excluded)
        if not candidates:
            return None

        best_entry: dict[str, Any] | None = None
        best_gain = -1.0

        for entry in candidates:
            phonemes: list[str] = entry["phonemes"]
            gain = tracker.marginal_gain(phonemes)
            if gain > best_gain:
                best_gain = gain
                best_entry = entry

        if best_entry is None:
            return None

        phonemes = best_entry["phonemes"]
        return MultilingualPrompt(
            text=best_entry["text"],
            language=language,
            phonemes=phonemes,
            unique_phoneme_count=len(
                set(phonemes) & lang.all_phonemes
            ),
        )
