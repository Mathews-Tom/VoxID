from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters import register_adapter
from ..adapters.protocol import EngineCapabilities
from ..serialization import load_prompt, save_prompt


@register_adapter
class Qwen3TTSAdapter:
    # Plain class-level string so register_adapter uses it as the registry key.
    engine_name: str = "qwen3-tts"

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device: str = "auto",
    ) -> None:
        self._model_name = model_name
        self._device = device
        self._model: Any = None  # Lazy-loaded

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=False,
            supports_emotion_control=False,
            supports_paralinguistic_tags=False,
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
            ),
            streaming_latency_ms=0,
            supports_word_timing=False,
        )

    def _ensure_model(self) -> Any:
        if self._model is None:
            try:
                import torch  # type: ignore[import-not-found]
                from qwen_tts import Qwen3TTSModel  # type: ignore[import-not-found]
            except ImportError as exc:
                raise ImportError(
                    "qwen-tts is required for the Qwen3-TTS adapter. "
                    "Install it with: pip install qwen-tts"
                ) from exc
            self._model = Qwen3TTSModel.from_pretrained(
                self._model_name,
                device_map=self._device,
                dtype=torch.bfloat16,
            )
        return self._model

    def build_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
    ) -> Path:
        model = self._ensure_model()
        prompt_items = model.create_voice_clone_prompt(
            str(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        item = prompt_items[0]
        tensors: dict[str, np.ndarray] = {
            "ref_spk_embedding": item.ref_spk_embedding.cpu().numpy(),
        }
        if item.ref_code is not None:
            tensors["ref_code"] = item.ref_code.cpu().numpy()

        metadata: dict[str, str] = {
            "engine": "qwen3-tts",
            "model": self._model_name,
            "x_vector_only_mode": str(item.x_vector_only_mode),
            "icl_mode": str(item.icl_mode),
            "ref_text": ref_text,
        }
        save_prompt(tensors, output_path, metadata=metadata)
        return output_path

    def generate(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        model = self._ensure_model()
        import torch
        from qwen_tts import VoiceClonePromptItem

        tensors, metadata = load_prompt(prompt_path)

        ref_code = None
        if "ref_code" in tensors:
            ref_code = torch.from_numpy(tensors["ref_code"]).to(model.device)

        spk_emb = torch.from_numpy(tensors["ref_spk_embedding"]).to(
            model.device
        )

        x_vec_only = metadata.get("x_vector_only_mode", "False") == "True"
        icl = metadata.get("icl_mode", "True") == "True"
        ref_text = metadata.get("ref_text")

        prompt_item = VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=spk_emb,
            x_vector_only_mode=x_vec_only,
            icl_mode=icl,
            ref_text=ref_text,
        )

        # BCP-47 language code → Qwen3-TTS language name
        lang_map: dict[str, str] = {
            "en": "English",
            "zh": "Chinese",
            "ja": "Japanese",
            "ko": "Korean",
            "de": "German",
            "fr": "French",
            "ru": "Russian",
            "pt": "Portuguese",
            "es": "Spanish",
            "it": "Italian",
        }
        lang_name = lang_map.get(language.split("-")[0], "English")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang_name,
            voice_clone_prompt=[prompt_item],
        )
        result: tuple[np.ndarray, int] = (wavs[0], sr)
        return result

    def generate_streaming(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> Iterator[np.ndarray]:
        raise NotImplementedError(
            "Qwen3-TTS official package does not support streaming. "
            "See andimarafioti/faster-qwen3-tts for community streaming."
        )
