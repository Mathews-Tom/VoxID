from __future__ import annotations

import json
from pathlib import Path

import pytest

from voxid.enrollment.phoneme_tracker import ALL_PHONEMES, PhonemeTracker
from voxid.enrollment.script_generator import (
    EnrollmentPrompt,
    ScriptGenerator,
)

# Phonetically diverse sentences covering all 39 ARPAbet phonemes
_TEST_CORPUS = [
    {"text": "The quick brown fox jumps over the lazy dog",
     "style_tags": ["phonetic"]},
    {"text": "She sells seashells by the seashore every morning",
     "style_tags": ["phonetic"]},
    {"text": "A huge azure explosion rocked the garage on Tuesday",
     "style_tags": ["phonetic"]},
    {"text": "The judge charged the jury with a major decision",
     "style_tags": ["phonetic"]},
    {"text": "Playing music brings joy and relaxation to young people",
     "style_tags": ["phonetic"]},
    {"text": "The thin man breathed through his teeth and thought carefully",
     "style_tags": ["phonetic"]},
    {"text": "How much wood would a woodchuck chuck if given the chance",
     "style_tags": ["phonetic"]},
    {"text": "Measure the pleasure of leisure in a beige treasure chest",
     "style_tags": ["phonetic"]},
    {"text": "The boy sang a song while pushing the swing higher",
     "style_tags": ["phonetic"]},
    {"text": "Vision and revision guide every important decision we make",
     "style_tags": ["phonetic"]},
    {"text": "Church bells chimed beautifully on that cold January morning",
     "style_tags": ["phonetic"]},
    {"text": "Nothing ventured nothing gained when you reach for the stars",
     "style_tags": ["phonetic"]},
    {"text": "Books about cooking techniques fill the wooden shelving unit",
     "style_tags": ["phonetic"]},
    {"text": "I honestly can't believe it worked on the first try",
     "style_tags": ["conversational"]},
    {"text": "So yeah the meeting ran late again which was frustrating",
     "style_tags": ["conversational"]},
]


@pytest.fixture
def corpus_dir(tmp_path: Path) -> Path:
    """Create a temporary corpus directory with test data."""
    phonetic = [e for e in _TEST_CORPUS if "phonetic" in e["style_tags"]]
    conversational = [
        e for e in _TEST_CORPUS if "conversational" in e["style_tags"]
    ]

    (tmp_path / "en_phonetic.json").write_text(json.dumps(phonetic))
    (tmp_path / "en_conversational.json").write_text(
        json.dumps(conversational),
    )
    return tmp_path


@pytest.fixture
def generator(corpus_dir: Path) -> ScriptGenerator:
    return ScriptGenerator(corpus_path=corpus_dir)


class TestEnrollmentPrompt:
    def test_dataclass_fields(self) -> None:
        prompt = EnrollmentPrompt(
            text="hello world",
            style="phonetic",
            phonemes=["HH", "AH", "L", "OW", "W", "ER", "L", "D"],
            unique_phoneme_count=7,
            nasal_count=0,
            affricate_count=0,
        )
        assert prompt.text == "hello world"
        assert prompt.style == "phonetic"
        assert len(prompt.phonemes) == 8
        assert prompt.unique_phoneme_count == 7
        assert prompt.nasal_count == 0
        assert prompt.affricate_count == 0

    def test_to_dict(self) -> None:
        prompt = EnrollmentPrompt(
            text="test",
            style="phonetic",
            phonemes=["T", "EH", "S", "T"],
            unique_phoneme_count=3,
            nasal_count=0,
            affricate_count=0,
        )
        d = prompt.to_dict()
        assert d["text"] == "test"
        assert d["style"] == "phonetic"
        assert d["phonemes"] == ["T", "EH", "S", "T"]

    def test_frozen(self) -> None:
        prompt = EnrollmentPrompt(
            text="test", style="phonetic", phonemes=[],
            unique_phoneme_count=0, nasal_count=0, affricate_count=0,
        )
        with pytest.raises(AttributeError):
            prompt.text = "changed"  # type: ignore[misc]


class TestSelectPrompts:
    def test_returns_requested_count(
        self, generator: ScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("phonetic", count=3)
        assert len(prompts) == 3

    def test_first_prompt_has_highest_unique_phonemes(
        self, generator: ScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("phonetic", count=5)
        # First prompt selected by greedy should have high phoneme diversity
        assert prompts[0].unique_phoneme_count >= prompts[-1].unique_phoneme_count // 2

    def test_respects_style_filter(
        self, generator: ScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts("phonetic", count=3)
        for p in prompts:
            assert p.style == "phonetic"

    def test_excludes_specified_texts(
        self, generator: ScriptGenerator,
    ) -> None:
        first_text = _TEST_CORPUS[0]["text"]
        prompts = generator.select_prompts(
            "phonetic", count=5, exclude_texts=[first_text],
        )
        assert all(p.text != first_text for p in prompts)

    def test_with_small_corpus_returns_available(
        self, corpus_dir: Path,
    ) -> None:
        gen = ScriptGenerator(corpus_path=corpus_dir)
        prompts = gen.select_prompts("conversational", count=100)
        # Only 2 conversational entries in test corpus
        assert len(prompts) == 2

    def test_unknown_style_raises(
        self, generator: ScriptGenerator,
    ) -> None:
        with pytest.raises(ValueError, match="Unknown style"):
            generator.select_prompts("nonexistent")

    def test_weights_nasals_higher(
        self, generator: ScriptGenerator,
    ) -> None:
        prompts = generator.select_prompts(
            "phonetic", count=5, target_per_phoneme=1,
        )
        # At least one prompt in top-5 should contain nasals
        total_nasals = sum(p.nasal_count for p in prompts)
        assert total_nasals > 0


class TestSelectNextAdaptive:
    def test_fills_gap(self, generator: ScriptGenerator) -> None:
        tracker = PhonemeTracker(target_per_phoneme=1)
        prompt = generator.select_next_adaptive("phonetic", tracker)
        assert prompt is not None
        assert len(prompt.phonemes) > 0

    def test_returns_none_when_complete(
        self, generator: ScriptGenerator,
    ) -> None:
        tracker = PhonemeTracker(target_per_phoneme=0)
        result = generator.select_next_adaptive("phonetic", tracker)
        assert result is None

    def test_respects_exclude(
        self, generator: ScriptGenerator,
    ) -> None:
        tracker = PhonemeTracker(target_per_phoneme=1)
        all_texts = [e["text"] for e in _TEST_CORPUS if "phonetic" in e["style_tags"]]
        # Exclude all but one
        exclude = all_texts[1:]
        prompt = generator.select_next_adaptive(
            "phonetic", tracker, exclude_texts=exclude,
        )
        assert prompt is not None
        assert prompt.text == all_texts[0]

    def test_adaptive_improves_coverage(
        self, generator: ScriptGenerator,
    ) -> None:
        tracker = PhonemeTracker(target_per_phoneme=1)
        for _ in range(5):
            prompt = generator.select_next_adaptive("phonetic", tracker)
            if prompt is None:
                break
            tracker.ingest(prompt.text)
        assert tracker.coverage_percent() > 0


class TestBundledCorpora:
    """Tests that validate the real bundled corpora (not test fixtures)."""

    @pytest.fixture
    def bundled_generator(self) -> ScriptGenerator:
        return ScriptGenerator()

    def test_bundled_corpus_en_phonetic_loads(
        self, bundled_generator: ScriptGenerator,
    ) -> None:
        prompts = bundled_generator.select_prompts("phonetic", count=1)
        assert len(prompts) == 1

    def test_bundled_corpus_en_conversational_loads(
        self, bundled_generator: ScriptGenerator,
    ) -> None:
        prompts = bundled_generator.select_prompts(
            "conversational", count=1,
        )
        assert len(prompts) == 1

    def test_bundled_corpus_each_entry_has_required_fields(
        self,
    ) -> None:
        prompts_dir = Path(__file__).parent.parent / "src" / "voxid" / "enrollment" / "prompts"
        for json_file in prompts_dir.glob("en_*.json"):
            with open(json_file) as f:
                entries = json.load(f)
            for i, entry in enumerate(entries):
                assert "text" in entry, (
                    f"{json_file.name}[{i}] missing 'text'"
                )
                assert "style_tags" in entry, (
                    f"{json_file.name}[{i}] missing 'style_tags'"
                )
                assert isinstance(entry["style_tags"], list)

    def test_bundled_corpus_en_phonetic_covers_all_39_phonemes(
        self, bundled_generator: ScriptGenerator,
    ) -> None:
        prompts = bundled_generator.select_prompts(
            "phonetic", count=10, target_per_phoneme=2,
        )
        tracker = PhonemeTracker(target_per_phoneme=1)
        for p in prompts:
            tracker.ingest(p.text)
        uncovered = {
            ph for ph in ALL_PHONEMES
            if tracker.coverage_report().get(ph, 0) == 0
        }
        assert uncovered == set(), f"Phonemes not covered: {uncovered}"

    def test_bundled_corpus_minimum_sentence_count_per_style(
        self,
    ) -> None:
        prompts_dir = Path(__file__).parent.parent / "src" / "voxid" / "enrollment" / "prompts"
        minimums = {
            "en_phonetic.json": 100,
            "en_conversational.json": 50,
            "en_technical.json": 50,
            "en_narration.json": 50,
            "en_emphatic.json": 50,
        }
        for filename, minimum in minimums.items():
            path = prompts_dir / filename
            assert path.exists(), f"{filename} not found"
            with open(path) as f:
                entries = json.load(f)
            assert len(entries) >= minimum, (
                f"{filename} has {len(entries)} entries, need {minimum}"
            )

    def test_select_prompts_covers_all_39_phonemes_within_10(
        self, bundled_generator: ScriptGenerator,
    ) -> None:
        prompts = bundled_generator.select_prompts(
            "phonetic", count=10, target_per_phoneme=2,
        )
        tracker = PhonemeTracker(target_per_phoneme=2)
        for p in prompts:
            tracker.ingest(p.text)
        assert tracker.is_complete(), (
            f"Gaps remain: {tracker.gap_vector()}"
        )
