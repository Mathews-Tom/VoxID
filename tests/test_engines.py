from __future__ import annotations

import voxid.adapters.chatterbox  # noqa: F401
import voxid.adapters.cosyvoice2  # noqa: F401
import voxid.adapters.fish_speech  # noqa: F401
import voxid.adapters.indextts2  # noqa: F401
import voxid.adapters.qwen3_tts  # noqa: F401
import voxid.adapters.stub  # noqa: F401
from voxid.adapters import _registry, list_adapters

_EXPECTED_ENGINES = {
    "stub",
    "qwen3-tts",
    "fish-speech",
    "cosyvoice2",
    "indextts2",
    "chatterbox",
}


def test_all_adapters_registered() -> None:
    registered = set(list_adapters())
    assert _EXPECTED_ENGINES.issubset(registered)


def test_fish_speech_engine_name() -> None:
    adapter = _registry["fish-speech"]()
    assert adapter.engine_name == "fish-speech"


def test_fish_speech_capabilities() -> None:
    adapter = _registry["fish-speech"]()
    caps = adapter.capabilities
    assert caps.supports_streaming is True
    assert len(caps.supported_languages) >= 10
    # Sample rate is internal; verify by checking known languages
    assert "en" in caps.supported_languages
    assert "ar" in caps.supported_languages


def test_cosyvoice2_engine_name() -> None:
    adapter = _registry["cosyvoice2"]()
    assert adapter.engine_name == "cosyvoice2"


def test_cosyvoice2_capabilities() -> None:
    adapter = _registry["cosyvoice2"]()
    caps = adapter.capabilities
    assert caps.streaming_latency_ms == 150
    assert len(caps.supported_languages) >= 9


def test_indextts2_engine_name() -> None:
    adapter = _registry["indextts2"]()
    assert adapter.engine_name == "indextts2"


def test_indextts2_capabilities() -> None:
    adapter = _registry["indextts2"]()
    caps = adapter.capabilities
    assert caps.supports_emotion_control is True


def test_chatterbox_engine_name() -> None:
    adapter = _registry["chatterbox"]()
    assert adapter.engine_name == "chatterbox"


def test_chatterbox_capabilities() -> None:
    adapter = _registry["chatterbox"]()
    caps = adapter.capabilities
    assert caps.supports_paralinguistic_tags is True
    assert len(caps.supported_languages) == 23


def test_each_adapter_has_build_prompt() -> None:
    for name in _EXPECTED_ENGINES:
        cls = _registry[name]
        assert callable(getattr(cls, "build_prompt", None)), (
            f"{name} missing build_prompt"
        )


def test_each_adapter_has_generate() -> None:
    for name in _EXPECTED_ENGINES:
        cls = _registry[name]
        assert callable(getattr(cls, "generate", None)), (
            f"{name} missing generate"
        )


def test_each_adapter_has_generate_streaming() -> None:
    for name in _EXPECTED_ENGINES:
        cls = _registry[name]
        assert callable(getattr(cls, "generate_streaming", None)), (
            f"{name} missing generate_streaming"
        )


def test_adapters_engine_names_unique() -> None:
    names: list[str] = []
    for name in _EXPECTED_ENGINES:
        adapter = _registry[name]()
        names.append(adapter.engine_name)
    assert len(names) == len(set(names)), (
        f"Duplicate engine names found: {names}"
    )
