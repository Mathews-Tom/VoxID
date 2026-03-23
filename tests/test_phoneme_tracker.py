from __future__ import annotations

import pytest

from voxid.enrollment.phoneme_tracker import (
    ALL_PHONEMES,
    PHONEME_WEIGHTS,
    PhonemeTracker,
    load_cmudict,
    text_to_phonemes,
)


# --- CMUdict loader ---


class TestLoadCmudict:
    def test_load_cmudict_returns_nonempty_dict(self) -> None:
        cmu = load_cmudict()
        assert len(cmu) > 100_000

    def test_load_cmudict_contains_common_words(self) -> None:
        cmu = load_cmudict()
        for word in ("hello", "world", "the", "computer", "speech"):
            assert word in cmu, f"'{word}' missing from CMUdict"

    def test_load_cmudict_stress_stripped(self) -> None:
        cmu = load_cmudict()
        for pronunciations in cmu.values():
            for pron in pronunciations:
                for phoneme in pron:
                    assert phoneme.isalpha(), (
                        f"Stress digit found in '{phoneme}'"
                    )


# --- text_to_phonemes ---


class TestTextToPhonemes:
    def test_simple_sentence_returns_correct_phonemes(self) -> None:
        phonemes = text_to_phonemes("hello")
        assert len(phonemes) > 0
        assert all(p in ALL_PHONEMES for p in phonemes)

    def test_unknown_word_returns_partial(self) -> None:
        # "xyzzyplugh" is not in CMUdict, "the" is
        phonemes = text_to_phonemes("the xyzzyplugh cat")
        assert len(phonemes) > 0
        # should have phonemes from "the" and "cat" only
        the_phonemes = text_to_phonemes("the")
        cat_phonemes = text_to_phonemes("cat")
        assert phonemes == the_phonemes + cat_phonemes

    def test_empty_string_returns_empty(self) -> None:
        assert text_to_phonemes("") == []

    def test_punctuation_stripped(self) -> None:
        with_punct = text_to_phonemes("Hello, world!")
        without_punct = text_to_phonemes("Hello world")
        assert with_punct == without_punct

    def test_case_insensitive(self) -> None:
        upper = text_to_phonemes("HELLO")
        lower = text_to_phonemes("hello")
        assert upper == lower

    def test_contractions_handled(self) -> None:
        phonemes = text_to_phonemes("I can't believe it's working")
        assert len(phonemes) > 0

    def test_accepts_explicit_cmudict(self) -> None:
        cmu = load_cmudict()
        phonemes = text_to_phonemes("hello", cmu=cmu)
        assert len(phonemes) > 0


# --- Phoneme constants ---


class TestPhonemeConstants:
    def test_all_phonemes_has_39_entries(self) -> None:
        assert len(ALL_PHONEMES) == 39

    def test_all_phonemes_are_alphabetic(self) -> None:
        for p in ALL_PHONEMES:
            assert p.isalpha()
            assert p.isupper()

    def test_phoneme_weights_covers_all_phonemes(self) -> None:
        assert set(PHONEME_WEIGHTS.keys()) == ALL_PHONEMES

    def test_phoneme_weights_nasals_weighted_higher(self) -> None:
        for nasal in ("N", "M", "NG"):
            assert PHONEME_WEIGHTS[nasal] == 1.5

    def test_phoneme_weights_affricates_weighted_higher(self) -> None:
        for affricate in ("CH", "JH"):
            assert PHONEME_WEIGHTS[affricate] == 1.5

    def test_phoneme_weights_vowels_weighted_medium(self) -> None:
        vowels = {"AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
                  "IH", "IY", "OW", "OY", "UH", "UW"}
        for v in vowels:
            assert PHONEME_WEIGHTS[v] == 1.2

    def test_phoneme_weights_plain_consonants_at_1(self) -> None:
        plain = ALL_PHONEMES - {"N", "M", "NG", "CH", "JH"} - {
            "AA", "AE", "AH", "AO", "AW", "AY", "EH", "ER", "EY",
            "IH", "IY", "OW", "OY", "UH", "UW"
        }
        for p in plain:
            assert PHONEME_WEIGHTS[p] == 1.0


# --- PhonemeTracker ---


class TestPhonemeTracker:
    def test_tracker_ingest_updates_coverage(self) -> None:
        tracker = PhonemeTracker()
        tracker.ingest("hello world")
        report = tracker.coverage_report()
        assert any(count > 0 for count in report.values())

    def test_tracker_ingest_multiple_texts_accumulates(self) -> None:
        tracker = PhonemeTracker()
        tracker.ingest("hello")
        first = sum(tracker.coverage_report().values())
        tracker.ingest("world")
        second = sum(tracker.coverage_report().values())
        assert second > first

    def test_tracker_gap_vector_shows_deficit(self) -> None:
        tracker = PhonemeTracker(target_per_phoneme=2)
        gap = tracker.gap_vector()
        # initially all phonemes have deficit of 2
        assert len(gap) == 39
        assert all(v == 2 for v in gap.values())

    def test_tracker_gap_vector_empty_when_complete(self) -> None:
        tracker = PhonemeTracker(target_per_phoneme=0)
        assert tracker.gap_vector() == {}

    def test_tracker_coverage_percent_zero_initially(self) -> None:
        tracker = PhonemeTracker()
        assert tracker.coverage_percent() == 0.0

    def test_tracker_coverage_percent_100_when_all_met(self) -> None:
        tracker = PhonemeTracker(target_per_phoneme=0)
        assert tracker.coverage_percent() == 100.0

    def test_tracker_is_complete_false_initially(self) -> None:
        tracker = PhonemeTracker()
        assert tracker.is_complete() is False

    def test_tracker_is_complete_true_after_sufficient_ingestion(self) -> None:
        tracker = PhonemeTracker(target_per_phoneme=1)
        # Ingest a phonetically rich set of sentences
        sentences = [
            "The quick brown fox jumps over the lazy dog",
            "She sells seashells by the seashore",
            "How much wood would a woodchuck chuck",
            "The judge charged the jury with a major decision",
            "Playing music brings joy and relaxation to young people",
            "The thin man breathed through his teeth",
            "A huge azure explosion rocked the garage",
        ]
        for s in sentences:
            tracker.ingest(s)
        # With target=1, these rich sentences should cover all phonemes
        uncovered = tracker.gap_vector()
        assert tracker.is_complete(), f"Still missing: {uncovered}"

    def test_tracker_roundtrip_serialization(self) -> None:
        tracker = PhonemeTracker(target_per_phoneme=3)
        tracker.ingest("hello world")

        data = tracker.to_dict()
        restored = PhonemeTracker.from_dict(data)

        assert restored.target == tracker.target
        assert restored.coverage_report() == tracker.coverage_report()
        assert restored.gap_vector() == tracker.gap_vector()

    def test_tracker_target_per_phoneme_configurable(self) -> None:
        tracker = PhonemeTracker(target_per_phoneme=5)
        assert tracker.target == 5
        gap = tracker.gap_vector()
        assert all(v == 5 for v in gap.values())

    def test_tracker_ignores_unknown_phonemes_in_from_dict(self) -> None:
        data = {
            "target_per_phoneme": 2,
            "coverage": {"AA": 1, "FAKE_PHONEME": 99},
        }
        tracker = PhonemeTracker.from_dict(data)
        assert tracker.coverage_report()["AA"] == 1
        assert "FAKE_PHONEME" not in tracker.coverage_report()
