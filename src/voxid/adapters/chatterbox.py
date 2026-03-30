from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..serialization import load_prompt, save_prompt
from . import register_adapter
from .protocol import EngineCapabilities

_SAMPLE_RATE = 24000
# Chatterbox embeds a Perth neural watermark in all generated audio.
_WATERMARK = "perth"


@register_adapter
class ChatterboxAdapter:
    """Chatterbox TTS engine adapter with paralinguistic tag support.

    Paralinguistic tags (e.g. [laugh], [cough], [sigh]) in input text are
    passed through to the model unchanged. All outputs carry a Perth neural
    watermark recorded in prompt metadata.
    """

    engine_name: str = "chatterbox"

    def __init__(
        self,
        model_name: str = "ResembleAI/chatterbox",
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=True,
            supports_emotion_control=False,
            supports_paralinguistic_tags=True,
            max_ref_audio_seconds=30.0,
            supported_languages=(
                "en",
                "zh",
                "ja",
                "ko",
                "de",
                "fr",
                "ru",
                "pt",
                "es",
                "it",
                "nl",
                "pl",
                "tr",
                "sv",
                "da",
                "no",
                "fi",
                "el",
                "cs",
                "ro",
                "hu",
                "th",
                "vi",
            ),
            streaming_latency_ms=200,
            supports_word_timing=False,
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from chatterbox import ChatterboxModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "chatterbox is required. "
                    "Install with: pip install voxid[chatterbox]"
                ) from exc
            self._model = ChatterboxModel.from_pretrained(
                self._model_name,
                device=self._device,
            )
        return self._model

    def build_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
    ) -> Path:
        model = self._ensure_model()
        embedding: np.ndarray = model.extract_speaker_embedding(str(ref_audio))
        tensors: dict[str, np.ndarray] = {
            "ref_spk_embedding": np.asarray(embedding, dtype=np.float32),
        }
        metadata: dict[str, str] = {
            "engine": "chatterbox",
            "model": self._model_name,
            "ref_text": ref_text,
            "watermark": _WATERMARK,
        }
        save_prompt(tensors, output_path, metadata=metadata)
        return output_path

    def generate(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
        context_params: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, int]:
        model = self._ensure_model()
        tensors, metadata = load_prompt(prompt_path)
        spk_emb = tensors["ref_spk_embedding"]
        ref_text = metadata.get("ref_text", "")
        lang_key = language.split("-")[0]
        # Paralinguistic tags in text (e.g. [laugh]) pass through unmodified.
        audio: np.ndarray = model.generate(
            text=text,
            speaker_embedding=spk_emb,
            ref_text=ref_text,
            language=lang_key,
        )
        return np.asarray(audio, dtype=np.float32), _SAMPLE_RATE

    def generate_streaming(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> Iterator[np.ndarray]:
        model = self._ensure_model()
        tensors, metadata = load_prompt(prompt_path)
        spk_emb = tensors["ref_spk_embedding"]
        ref_text = metadata.get("ref_text", "")
        lang_key = language.split("-")[0]
        # Paralinguistic tags in text (e.g. [laugh]) pass through unmodified.
        for chunk in model.generate_streaming(
            text=text,
            speaker_embedding=spk_emb,
            ref_text=ref_text,
            language=lang_key,
        ):
            yield np.asarray(chunk, dtype=np.float32)
