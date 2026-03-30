from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..serialization import load_prompt, save_prompt
from . import register_adapter
from .protocol import EngineCapabilities

_SAMPLE_RATE = 24000

# VoxID style description → emotion vector index.
# IndexTTS-2 uses disentangled emotion + speaker identity.
_STYLE_TO_EMOTION: dict[str, int] = {
    "conversational": 0,  # neutral
    "emphatic": 1,        # excited
    "calm": 2,
    "sad": 3,
    "angry": 4,
    "happy": 1,           # maps to excited
    "neutral": 0,
}
_EMOTION_DIM = 8
_DEFAULT_STYLE = "conversational"


def _build_emotion_vector(style: str) -> np.ndarray:
    """Map a VoxID style label to an emotion one-hot vector."""
    idx = _STYLE_TO_EMOTION.get(style.lower(), 0)
    vec = np.zeros(_EMOTION_DIM, dtype=np.float32)
    vec[idx] = 1.0
    return vec


@register_adapter
class IndexTTS2Adapter:
    """IndexTTS-2 TTS engine adapter with disentangled emotion control."""

    engine_name: str = "indextts2"

    def __init__(
        self,
        model_name: str = "IndexTeam/IndexTTS-2",
        device: str = "auto",
        style: str = _DEFAULT_STYLE,
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._style = style
        self._model: Any = None

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=True,
            supports_emotion_control=True,
            supports_paralinguistic_tags=False,
            max_ref_audio_seconds=30.0,
            supported_languages=("en", "zh"),
            streaming_latency_ms=200,
            supports_word_timing=False,
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                from indextts2 import IndexTTSModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "indextts2 is required. "
                    "Install with: pip install voxid[indextts2]"
                ) from exc
            self._model = IndexTTSModel.from_pretrained(
                self._model_name,
                device=self._device,
            )
        return self._model

    def build_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
        style: str | None = None,
    ) -> Path:
        model = self._ensure_model()
        embedding: np.ndarray = model.extract_speaker_embedding(str(ref_audio))
        resolved_style = style if style is not None else self._style
        emotion_vec = _build_emotion_vector(resolved_style)
        tensors: dict[str, np.ndarray] = {
            "ref_spk_embedding": np.asarray(embedding, dtype=np.float32),
            "emotion_vector": emotion_vec,
        }
        metadata: dict[str, str] = {
            "engine": "indextts2",
            "model": self._model_name,
            "ref_text": ref_text,
            "style": resolved_style,
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
        emotion_vec = tensors.get("emotion_vector")
        ref_text = metadata.get("ref_text", "")
        lang_key = language.split("-")[0]
        audio: np.ndarray = model.generate(
            text=text,
            speaker_embedding=spk_emb,
            emotion_vector=emotion_vec,
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
        emotion_vec = tensors.get("emotion_vector")
        ref_text = metadata.get("ref_text", "")
        lang_key = language.split("-")[0]
        for chunk in model.generate_streaming(
            text=text,
            speaker_embedding=spk_emb,
            emotion_vector=emotion_vec,
            ref_text=ref_text,
            language=lang_key,
        ):
            yield np.asarray(chunk, dtype=np.float32)
