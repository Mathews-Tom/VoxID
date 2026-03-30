from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from voxid.enrollment.multilingual.language_config import get_language_config
from voxid.enrollment.multilingual.phoneme_universal import UniversalPhonemeTracker
from voxid.enrollment.multilingual.script_generator import (
    MultilingualPrompt,
    MultilingualScriptGenerator,
)


@pytest.fixture
def corpora_path() -> Path:
    return (
        Path(__file__).parent.parent
        / "src"
        / "voxid"
        / "enrollment"
        / "multilingual"
        / "corpora"
    )


@pytest.fixture
def generator(corpora_path: Path) -> MultilingualScriptGenerator:
    return MultilingualScriptGenerator(corpus_path=corpora_path)


class TestCorpusFiles:
    def test_all_configured_corpora_exist(self, corpora_path: Path) -> None:
        from voxid.enrollment.multilingual.language_config import (
            list_languages,
            get_language_config,
        )

        for code in list_languages():
            cfg = get_language_config(code)
            assert cfg.corpus_file is not None
            corpus = corpora_path / cfg.corpus_file
            assert corpus.exists(), f"Missing corpus: {corpus}"

    def test_corpus_format_valid(self, corpora_path: Path) -> None:
        from voxid.enrollment.multilingual.language_config import (
            list_languages,
            get_language_config,
        )

        for code in list_languages():
            cfg = get_language_config(code)
            assert cfg.corpus_file is not None
            path = corpora_path / cfg.corpus_file
            with open(path, encoding="utf-8") as f:
                entries: list[dict[str, Any]] = json.load(f)

            assert len(entries) >= 5, (
                f"Corpus {cfg.corpus_file} has only {len(entries)} entries"
            )
            for i, entry in enumerate(entries):
                assert "text" in entry, f"Entry {i} in {cfg.corpus_file} missing 'text'"
                assert "phonemes" in entry, (
                    f"Entry {i} in {cfg.corpus_file} missing 'phonemes'"
                )
                assert isinstance(entry["phonemes"], list)
                assert len(entry["phonemes"]) > 0


class TestMultilingualScriptGenerator:
    def test_select_prompts_returns_correct_count(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("en", count=3)
        assert len(prompts) == 3
        assert all(isinstance(p, MultilingualPrompt) for p in prompts)

    def test_select_prompts_for_each_language(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        from voxid.enrollment.multilingual.language_config import list_languages

        for code in list_languages():
            prompts = generator.select_prompts(code, count=3)
            assert len(prompts) >= 1, f"No prompts for {code}"
            assert all(p.language == code for p in prompts)

    def test_select_prompts_no_duplicates(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("en", count=8)
        texts = [p.text for p in prompts]
        assert len(texts) == len(set(texts))

    def test_select_prompts_respects_exclude(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        first = generator.select_prompts("es", count=3)
        exclude = [p.text for p in first]
        second = generator.select_prompts("es", count=3, exclude_texts=exclude)
        second_texts = {p.text for p in second}
        for excluded in exclude:
            assert excluded not in second_texts

    def test_prompt_phonemes_are_from_language_inventory(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        cfg = get_language_config("en")
        prompts = generator.select_prompts("en", count=5)
        for p in prompts:
            for phoneme in p.phonemes:
                assert phoneme in cfg.all_phonemes, (
                    f"Phoneme '{phoneme}' not in English inventory"
                )

    def test_greedy_selection_improves_coverage(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("en", count=5, target_per_phoneme=1)
        tracker = UniversalPhonemeTracker("en", target_per_phoneme=1)
        prev_coverage = 0.0
        for p in prompts:
            tracker.ingest_phonemes(p.phonemes)
            current = tracker.coverage_percent()
            assert current >= prev_coverage
            prev_coverage = current

    def test_select_next_adaptive_returns_none_when_complete(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        tracker = UniversalPhonemeTracker("es", target_per_phoneme=1)
        cfg = get_language_config("es")
        tracker.ingest_phonemes(list(cfg.all_phonemes))
        assert tracker.is_complete()
        result = generator.select_next_adaptive("es", tracker)
        assert result is None

    def test_select_next_adaptive_returns_prompt_when_gaps(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        tracker = UniversalPhonemeTracker("en", target_per_phoneme=2)
        result = generator.select_next_adaptive("en", tracker)
        assert result is not None
        assert isinstance(result, MultilingualPrompt)

    def test_unsupported_language_raises(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        with pytest.raises(KeyError, match="Unsupported language"):
            generator.select_prompts("xx", count=3)

    def test_prompt_unique_phoneme_count(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("de", count=3)
        cfg = get_language_config("de")
        for p in prompts:
            expected = len(set(p.phonemes) & cfg.all_phonemes)
            assert p.unique_phoneme_count == expected

    def test_prompt_serialization(
        self, generator: MultilingualScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("fr", count=1)
        d = prompts[0].to_dict()
        assert d["text"] == prompts[0].text
        assert d["language"] == "fr"
        assert d["phonemes"] == prompts[0].phonemes
