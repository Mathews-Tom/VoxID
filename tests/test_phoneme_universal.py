from __future__ import annotations

import pytest

from voxid.enrollment.multilingual.language_config import (
    LanguageConfig,
    get_language_config,
    list_languages,
)
from voxid.enrollment.multilingual.phoneme_universal import (
    IPA_PHONEME_WEIGHTS,
    UniversalPhonemeTracker,
)


class TestLanguageConfig:
    def test_list_languages_returns_sorted_codes(self) -> None:
        langs = list_languages()
        assert isinstance(langs, list)
        assert langs == sorted(langs)
        assert "en" in langs
        assert "zh" in langs
        assert len(langs) >= 10

    def test_get_language_config_english(self) -> None:
        cfg = get_language_config("en")
        assert cfg.code == "en"
        assert cfg.name == "English"
        assert "p" in cfg.consonants
        assert "iː" in cfg.vowels
        assert "m" in cfg.nasals
        assert "tʃ" in cfg.affricates
        assert cfg.corpus_file == "en.json"

    def test_get_language_config_mandarin(self) -> None:
        cfg = get_language_config("zh")
        assert cfg.code == "zh"
        assert "tɕ" in cfg.consonants
        assert "tʂ" in cfg.affricates

    def test_get_language_config_unsupported_raises(self) -> None:
        with pytest.raises(KeyError, match="Unsupported language 'xx'"):
            get_language_config("xx")

    def test_all_phonemes_is_union(self) -> None:
        cfg = get_language_config("en")
        assert cfg.all_phonemes == cfg.consonants | cfg.vowels

    def test_all_10_languages_have_valid_inventories(self) -> None:
        for code in list_languages():
            cfg = get_language_config(code)
            assert len(cfg.consonants) > 0
            assert len(cfg.vowels) > 0
            assert cfg.nasals <= cfg.consonants
            assert cfg.corpus_file is not None


class TestUniversalPhonemeTracker:
    def test_init_creates_empty_coverage(self) -> None:
        tracker = UniversalPhonemeTracker("en")
        assert tracker.language == "en"
        assert tracker.language_name == "English"
        assert tracker.coverage_percent() == 0.0
        assert not tracker.is_complete()

    def test_ingest_phonemes_updates_coverage(self) -> None:
        tracker = UniversalPhonemeTracker("en", target_per_phoneme=1)
        tracker.ingest_phonemes(["p", "b", "t"])
        report = tracker.coverage_report()
        assert report["p"] == 1
        assert report["b"] == 1
        assert report["t"] == 1
        assert report.get("k", 0) == 0

    def test_ingest_ignores_unknown_phonemes(self) -> None:
        tracker = UniversalPhonemeTracker("en")
        tracker.ingest_phonemes(["UNKNOWN", "xyz"])
        assert all(v == 0 for v in tracker.coverage_report().values())

    def test_coverage_percent_with_partial_coverage(self) -> None:
        tracker = UniversalPhonemeTracker("es", target_per_phoneme=1)
        cfg = get_language_config("es")
        all_phonemes = sorted(cfg.all_phonemes)
        # Cover half the phonemes
        half = all_phonemes[: len(all_phonemes) // 2]
        tracker.ingest_phonemes(half)
        pct = tracker.coverage_percent()
        assert 40.0 < pct < 60.0

    def test_is_complete_when_all_phonemes_met(self) -> None:
        tracker = UniversalPhonemeTracker("es", target_per_phoneme=1)
        cfg = get_language_config("es")
        tracker.ingest_phonemes(list(cfg.all_phonemes))
        assert tracker.is_complete()

    def test_gap_vector_shows_deficits(self) -> None:
        tracker = UniversalPhonemeTracker("en", target_per_phoneme=2)
        tracker.ingest_phonemes(["p"])
        gap = tracker.gap_vector()
        assert gap["p"] == 1  # needs 1 more
        assert "b" in gap  # not seen at all, deficit = 2

    def test_weighted_gap_uses_category_weights(self) -> None:
        tracker = UniversalPhonemeTracker("en", target_per_phoneme=1)
        wgap = tracker.weighted_gap()
        # Nasals should have higher weighted deficit
        assert wgap["m"] == IPA_PHONEME_WEIGHTS["nasal"]
        assert wgap["tʃ"] == IPA_PHONEME_WEIGHTS["affricate"]
        # Approximants should have lower weight
        assert wgap["l"] == IPA_PHONEME_WEIGHTS["approximant"]

    def test_marginal_gain_computation(self) -> None:
        tracker = UniversalPhonemeTracker("en", target_per_phoneme=1)
        # Nasal "m" should contribute 1.5
        gain = tracker.marginal_gain(["m"])
        assert gain == IPA_PHONEME_WEIGHTS["nasal"]

        # Already covered phoneme should contribute 0
        tracker.ingest_phonemes(["m"])
        gain_after = tracker.marginal_gain(["m"])
        assert gain_after == 0.0

    def test_weight_for_each_category(self) -> None:
        tracker = UniversalPhonemeTracker("en")
        assert tracker.weight_for("m") == 1.5  # nasal
        assert tracker.weight_for("tʃ") == 1.5  # affricate
        assert tracker.weight_for("iː") == 1.2  # vowel
        assert tracker.weight_for("p") == 1.0  # plosive
        assert tracker.weight_for("f") == 1.0  # fricative
        assert tracker.weight_for("l") == 0.8  # approximant

    def test_serialization_roundtrip(self) -> None:
        tracker = UniversalPhonemeTracker("ja", target_per_phoneme=3)
        tracker.ingest_phonemes(["k", "a", "n", "a"])
        data = tracker.to_dict()
        restored = UniversalPhonemeTracker.from_dict(data)

        assert restored.language == "ja"
        assert restored.target == 3
        assert restored.coverage_report()["k"] == 1
        assert restored.coverage_report()["a"] == 2
        assert restored.coverage_report()["n"] == 1

    def test_multiple_languages_independent(self) -> None:
        en = UniversalPhonemeTracker("en")
        zh = UniversalPhonemeTracker("zh")
        assert en.lang_config.all_phonemes != zh.lang_config.all_phonemes
