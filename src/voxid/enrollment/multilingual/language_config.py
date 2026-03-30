from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LanguageConfig:
    """Per-language phoneme inventory and enrollment parameters.

    Each language defines its IPA consonant and vowel inventories,
    speaker-discriminative phoneme categories, and minimum enrollment
    coverage targets.
    """

    code: str
    name: str
    consonants: frozenset[str]
    vowels: frozenset[str]
    nasals: frozenset[str]
    affricates: frozenset[str]
    fricatives: frozenset[str]
    plosives: frozenset[str]
    approximants: frozenset[str]
    min_coverage_target: int = 2
    corpus_file: str | None = None

    @property
    def all_phonemes(self) -> frozenset[str]:
        return self.consonants | self.vowels


# ── Language inventories (IPA) ──────────────────────────────────────────
# Sources: IPA Handbook, PHOIBLE, UCLA Phonological Segment Inventory DB

_LANGUAGES: dict[str, LanguageConfig] = {}


def _register(cfg: LanguageConfig) -> None:
    _LANGUAGES[cfg.code] = cfg


_register(LanguageConfig(
    code="en",
    name="English",
    consonants=frozenset({
        "p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ",
        "f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h",
        "m", "n", "ŋ", "l", "ɹ", "j", "w",
    }),
    vowels=frozenset({
        "iː", "ɪ", "eɪ", "ɛ", "æ", "ɑː", "ɒ", "ɔː", "oʊ",
        "ʊ", "uː", "ʌ", "ɜː", "ə", "aɪ", "aʊ", "ɔɪ",
    }),
    nasals=frozenset({"m", "n", "ŋ"}),
    affricates=frozenset({"tʃ", "dʒ"}),
    fricatives=frozenset({"f", "v", "θ", "ð", "s", "z", "ʃ", "ʒ", "h"}),
    plosives=frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    approximants=frozenset({"l", "ɹ", "j", "w"}),
    corpus_file="en.json",
))

_register(LanguageConfig(
    code="zh",
    name="Mandarin Chinese",
    consonants=frozenset({
        "p", "pʰ", "m", "f", "t", "tʰ", "n", "l",
        "k", "kʰ", "x", "tɕ", "tɕʰ", "ɕ",
        "ts", "tsʰ", "s", "tʂ", "tʂʰ", "ʂ", "ʐ",
        "ŋ",
    }),
    vowels=frozenset({
        "a", "o", "ɤ", "i", "u", "y", "ɛ", "ai", "ei",
        "au", "ou", "an", "ən", "aŋ", "əŋ",
    }),
    nasals=frozenset({"m", "n", "ŋ"}),
    affricates=frozenset({"tɕ", "tɕʰ", "ts", "tsʰ", "tʂ", "tʂʰ"}),
    fricatives=frozenset({"f", "x", "ɕ", "s", "ʂ", "ʐ"}),
    plosives=frozenset({"p", "pʰ", "t", "tʰ", "k", "kʰ"}),
    approximants=frozenset({"l"}),
    corpus_file="zh.json",
))

_register(LanguageConfig(
    code="ja",
    name="Japanese",
    consonants=frozenset({
        "p", "b", "t", "d", "k", "ɡ", "ts", "dz", "tɕ", "dʑ",
        "m", "n", "ɲ", "ŋ", "ɴ", "s", "z", "ɕ", "ʑ", "ç", "h",
        "ɸ", "ɾ", "j", "w",
    }),
    vowels=frozenset({"a", "i", "ɯ", "e", "o", "aː", "iː", "ɯː", "eː", "oː"}),
    nasals=frozenset({"m", "n", "ɲ", "ŋ", "ɴ"}),
    affricates=frozenset({"ts", "dz", "tɕ", "dʑ"}),
    fricatives=frozenset({"s", "z", "ɕ", "ʑ", "ç", "h", "ɸ"}),
    plosives=frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    approximants=frozenset({"ɾ", "j", "w"}),
    corpus_file="ja.json",
))

_register(LanguageConfig(
    code="ko",
    name="Korean",
    consonants=frozenset({
        "p", "pʰ", "p*", "t", "tʰ", "t*", "k", "kʰ", "k*",
        "tɕ", "tɕʰ", "tɕ*", "s", "s*", "h",
        "m", "n", "ŋ", "l",
    }),
    vowels=frozenset({
        "a", "ʌ", "o", "u", "ɯ", "i", "e", "ɛ", "ø", "y",
        "wa", "wʌ", "we", "wi", "ja", "jʌ", "jo", "ju", "je", "jɛ",
    }),
    nasals=frozenset({"m", "n", "ŋ"}),
    affricates=frozenset({"tɕ", "tɕʰ", "tɕ*"}),
    fricatives=frozenset({"s", "s*", "h"}),
    plosives=frozenset({"p", "pʰ", "p*", "t", "tʰ", "t*", "k", "kʰ", "k*"}),
    approximants=frozenset({"l"}),
    corpus_file="ko.json",
))

_register(LanguageConfig(
    code="es",
    name="Spanish",
    consonants=frozenset({
        "p", "b", "t", "d", "k", "ɡ", "tʃ",
        "f", "θ", "s", "x", "ʝ",
        "m", "n", "ɲ", "l", "ʎ", "r", "ɾ",
    }),
    vowels=frozenset({"a", "e", "i", "o", "u"}),
    nasals=frozenset({"m", "n", "ɲ"}),
    affricates=frozenset({"tʃ"}),
    fricatives=frozenset({"f", "θ", "s", "x", "ʝ"}),
    plosives=frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    approximants=frozenset({"l", "ʎ", "r", "ɾ"}),
    corpus_file="es.json",
))

_register(LanguageConfig(
    code="de",
    name="German",
    consonants=frozenset({
        "p", "b", "t", "d", "k", "ɡ", "pf", "ts", "tʃ",
        "f", "v", "s", "z", "ʃ", "ʒ", "ç", "x", "h",
        "m", "n", "ŋ", "l", "ʁ", "j",
    }),
    vowels=frozenset({
        "iː", "ɪ", "eː", "ɛ", "ɛː", "aː", "a",
        "oː", "ɔ", "uː", "ʊ", "yː", "ʏ", "øː", "œ", "ə",
    }),
    nasals=frozenset({"m", "n", "ŋ"}),
    affricates=frozenset({"pf", "ts", "tʃ"}),
    fricatives=frozenset({"f", "v", "s", "z", "ʃ", "ʒ", "ç", "x", "h"}),
    plosives=frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    approximants=frozenset({"l", "ʁ", "j"}),
    corpus_file="de.json",
))

_register(LanguageConfig(
    code="fr",
    name="French",
    consonants=frozenset({
        "p", "b", "t", "d", "k", "ɡ",
        "f", "v", "s", "z", "ʃ", "ʒ",
        "m", "n", "ɲ", "ŋ", "l", "ʁ", "j", "w", "ɥ",
    }),
    vowels=frozenset({
        "i", "e", "ɛ", "a", "ɑ", "ɔ", "o", "u", "y", "ø", "œ", "ə",
        "ɛ̃", "ɑ̃", "ɔ̃", "œ̃",
    }),
    nasals=frozenset({"m", "n", "ɲ", "ŋ"}),
    affricates=frozenset(),
    fricatives=frozenset({"f", "v", "s", "z", "ʃ", "ʒ"}),
    plosives=frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    approximants=frozenset({"l", "ʁ", "j", "w", "ɥ"}),
    corpus_file="fr.json",
))

_register(LanguageConfig(
    code="pt",
    name="Portuguese",
    consonants=frozenset({
        "p", "b", "t", "d", "k", "ɡ", "tʃ", "dʒ",
        "f", "v", "s", "z", "ʃ", "ʒ",
        "m", "n", "ɲ", "l", "ʎ", "ɾ", "ʁ", "j", "w",
    }),
    vowels=frozenset({
        "a", "ɐ", "e", "ɛ", "i", "o", "ɔ", "u",
        "ã", "ẽ", "ĩ", "õ", "ũ",
    }),
    nasals=frozenset({"m", "n", "ɲ"}),
    affricates=frozenset({"tʃ", "dʒ"}),
    fricatives=frozenset({"f", "v", "s", "z", "ʃ", "ʒ"}),
    plosives=frozenset({"p", "b", "t", "d", "k", "ɡ"}),
    approximants=frozenset({"l", "ʎ", "ɾ", "ʁ", "j", "w"}),
    corpus_file="pt.json",
))

_register(LanguageConfig(
    code="hi",
    name="Hindi",
    consonants=frozenset({
        "p", "pʰ", "b", "bʰ", "t̪", "t̪ʰ", "d̪", "d̪ʰ",
        "ʈ", "ʈʰ", "ɖ", "ɖʰ", "k", "kʰ", "ɡ", "ɡʰ",
        "tʃ", "tʃʰ", "dʒ", "dʒʰ",
        "f", "s", "z", "ʃ", "h",
        "m", "n", "ɳ", "ŋ",
        "l", "r", "ɾ", "j", "ʋ",
    }),
    vowels=frozenset({
        "ə", "aː", "ɪ", "iː", "ʊ", "uː", "eː", "ɛː",
        "oː", "ɔː", "ãː", "ẽː", "ĩː", "õː", "ũː",
    }),
    nasals=frozenset({"m", "n", "ɳ", "ŋ"}),
    affricates=frozenset({"tʃ", "tʃʰ", "dʒ", "dʒʰ"}),
    fricatives=frozenset({"f", "s", "z", "ʃ", "h"}),
    plosives=frozenset({
        "p", "pʰ", "b", "bʰ", "t̪", "t̪ʰ", "d̪", "d̪ʰ",
        "ʈ", "ʈʰ", "ɖ", "ɖʰ", "k", "kʰ", "ɡ", "ɡʰ",
    }),
    approximants=frozenset({"l", "r", "ɾ", "j", "ʋ"}),
    corpus_file="hi.json",
))

_register(LanguageConfig(
    code="ar",
    name="Arabic",
    consonants=frozenset({
        "b", "t", "tˤ", "d", "dˤ", "k", "q", "ʔ",
        "dʒ", "f", "θ", "ð", "ðˤ", "s", "sˤ", "z",
        "ʃ", "x", "ɣ", "ħ", "ʕ", "h",
        "m", "n", "l", "r", "j", "w",
    }),
    vowels=frozenset({
        "a", "aː", "i", "iː", "u", "uː",
        "ai", "au",
    }),
    nasals=frozenset({"m", "n"}),
    affricates=frozenset({"dʒ"}),
    fricatives=frozenset({
        "f", "θ", "ð", "ðˤ", "s", "sˤ", "z",
        "ʃ", "x", "ɣ", "ħ", "ʕ", "h",
    }),
    plosives=frozenset({"b", "t", "tˤ", "d", "dˤ", "k", "q", "ʔ"}),
    approximants=frozenset({"l", "r", "j", "w"}),
    corpus_file="ar.json",
))


def get_language_config(code: str) -> LanguageConfig:
    """Get configuration for a language by ISO 639-1 code.

    Raises KeyError if the language is not supported.
    """
    if code not in _LANGUAGES:
        supported = sorted(_LANGUAGES)
        raise KeyError(
            f"Unsupported language '{code}'. "
            f"Supported: {supported}"
        )
    return _LANGUAGES[code]


def list_languages() -> list[str]:
    """Return sorted list of supported language codes."""
    return sorted(_LANGUAGES)
