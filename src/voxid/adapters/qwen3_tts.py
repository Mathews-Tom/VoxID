from __future__ import annotations

import platform
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np

from ..adapters import register_adapter
from ..adapters.protocol import EngineCapabilities
from ..serialization import load_prompt, save_prompt

# BCP-47 → Qwen3-TTS language name mapping
_LANG_MAP: dict[str, str] = {
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


def _detect_backend() -> str:
    """Detect available Qwen3-TTS backend.

    Prefer mlx-audio on Apple Silicon, qwen-tts elsewhere.
    Raises ImportError if neither is installed.
    """
    is_apple_silicon = (
        sys.platform == "darwin"
        and platform.machine() == "arm64"
    )
    if is_apple_silicon:
        try:
            import mlx_audio  # noqa: F401
            return "mlx-audio"
        except ImportError:
            pass
    try:
        import qwen_tts  # noqa: F401
        return "qwen-tts"
    except ImportError:
        pass
    # Fallback: try mlx-audio on non-Apple platforms too
    try:
        import mlx_audio  # noqa: F401
        return "mlx-audio"
    except ImportError:
        pass
    raise ImportError(
        "No Qwen3-TTS backend found. Install one of:\n"
        "  pip install voxid[qwen3-tts]       # CUDA/MPS\n"
        "  pip install voxid[qwen3-tts-mlx]   # Apple Silicon"
    )


@register_adapter
class Qwen3TTSAdapter:
    """Qwen3-TTS engine adapter with dual-backend support.

    Backends:
      - 'qwen-tts': Official package (CUDA/MPS). Extracts speaker
        embeddings upfront, generates from cached tensors.
      - 'mlx-audio': MLX port for Apple Silicon. Takes ref_audio
        directly at generation time — prompt stores metadata only.
      - 'auto': Detect available backend at runtime.
    """

    engine_name: str = "qwen3-tts"

    def __init__(
        self,
        model_name: str | None = None,
        device: str = "auto",
        backend: str = "auto",
    ) -> None:
        self._device = device
        self._backend = backend
        self._model: Any = None

        if backend == "auto":
            self._resolved_backend: str | None = None
        else:
            self._resolved_backend = backend

        # Default model depends on backend
        self._model_name = model_name
        self._default_model_qwen = (
            "Qwen/Qwen3-TTS-12Hz-1.7B-Base"
        )
        self._default_model_mlx = (
            "mlx-community/Qwen3-TTS-12Hz-1.7B-Base-6bit"
        )

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=False,
            supports_emotion_control=False,
            supports_paralinguistic_tags=False,
            max_ref_audio_seconds=30.0,
            supported_languages=tuple(_LANG_MAP),
            streaming_latency_ms=0,
            supports_word_timing=False,
        )

    def _resolve_backend(self) -> str:
        if self._resolved_backend is None:
            self._resolved_backend = _detect_backend()
        return self._resolved_backend

    def _get_model_name(self) -> str:
        if self._model_name is not None:
            return self._model_name
        backend = self._resolve_backend()
        if backend == "mlx-audio":
            return self._default_model_mlx
        return self._default_model_qwen

    # ── qwen-tts backend ──────────────────────────────

    def _ensure_model_qwen(self) -> Any:
        if self._model is None:
            import torch
            from qwen_tts import Qwen3TTSModel

            self._model = Qwen3TTSModel.from_pretrained(
                self._get_model_name(),
                device_map=self._device,
                dtype=torch.bfloat16,
            )
        return self._model

    def _build_prompt_qwen(
        self, ref_audio: Path, ref_text: str, output_path: Path,
    ) -> Path:
        model = self._ensure_model_qwen()
        prompt_items = model.create_voice_clone_prompt(
            str(ref_audio),
            ref_text=ref_text,
            x_vector_only_mode=False,
        )
        item = prompt_items[0]
        tensors: dict[str, np.ndarray] = {
            "ref_spk_embedding": (
                item.ref_spk_embedding.cpu().numpy()
            ),
        }
        if item.ref_code is not None:
            tensors["ref_code"] = (
                item.ref_code.cpu().numpy()
            )

        metadata: dict[str, str] = {
            "engine": "qwen3-tts",
            "backend": "qwen-tts",
            "model": self._get_model_name(),
            "x_vector_only_mode": str(
                item.x_vector_only_mode
            ),
            "icl_mode": str(item.icl_mode),
            "ref_text": ref_text,
        }
        save_prompt(tensors, output_path, metadata=metadata)
        return output_path

    def _generate_qwen(
        self, text: str, prompt_path: Path, language: str,
    ) -> tuple[np.ndarray, int]:
        model = self._ensure_model_qwen()
        import torch
        from qwen_tts import VoiceClonePromptItem

        tensors, metadata = load_prompt(prompt_path)

        ref_code = None
        if "ref_code" in tensors:
            ref_code = torch.from_numpy(
                tensors["ref_code"]
            ).to(model.device)

        spk_emb = torch.from_numpy(
            tensors["ref_spk_embedding"]
        ).to(model.device)

        prompt_item = VoiceClonePromptItem(
            ref_code=ref_code,
            ref_spk_embedding=spk_emb,
            x_vector_only_mode=(
                metadata.get(
                    "x_vector_only_mode", "False"
                )
                == "True"
            ),
            icl_mode=(
                metadata.get("icl_mode", "True") == "True"
            ),
            ref_text=metadata.get("ref_text"),
        )

        lang_key = language.split("-")[0]
        lang_name = _LANG_MAP.get(lang_key, "English")

        wavs, sr = model.generate_voice_clone(
            text=text,
            language=lang_name,
            voice_clone_prompt=[prompt_item],
        )
        result: tuple[np.ndarray, int] = (wavs[0], sr)
        return result

    # ── mlx-audio backend ─────────────────────────────

    def _ensure_model_mlx(self) -> Any:
        if self._model is None:
            from mlx_audio.tts.utils import load_model

            self._model = load_model(
                self._get_model_name()
            )
        return self._model

    def _build_prompt_mlx(
        self, ref_audio: Path, ref_text: str, output_path: Path,
    ) -> Path:
        # mlx-audio doesn't extract embeddings separately —
        # it takes ref_audio + ref_text at generation time.
        # Store a metadata-only prompt pointing to the source.
        tensors: dict[str, np.ndarray] = {
            "placeholder": np.zeros(1, dtype=np.float32),
        }
        metadata: dict[str, str] = {
            "engine": "qwen3-tts",
            "backend": "mlx-audio",
            "model": self._get_model_name(),
            "ref_audio_path": str(ref_audio.resolve()),
            "ref_text": ref_text,
        }
        save_prompt(tensors, output_path, metadata=metadata)
        return output_path

    def _generate_mlx(
        self, text: str, prompt_path: Path, language: str,
    ) -> tuple[np.ndarray, int]:
        model = self._ensure_model_mlx()
        _, metadata = load_prompt(prompt_path)

        ref_audio = metadata["ref_audio_path"]
        ref_text = metadata.get("ref_text", "")

        lang_key = language.split("-")[0]
        lang_name = _LANG_MAP.get(lang_key, "English")

        results = list(model.generate(
            text=text,
            ref_audio=ref_audio,
            ref_text=ref_text,
            language=lang_name,
        ))
        audio = np.array(results[0].audio, dtype=np.float32)
        sr = 24000
        return audio, sr

    # ── Public interface ──────────────────────────────

    def build_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
    ) -> Path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        backend = self._resolve_backend()
        if backend == "mlx-audio":
            return self._build_prompt_mlx(
                ref_audio, ref_text, output_path,
            )
        return self._build_prompt_qwen(
            ref_audio, ref_text, output_path,
        )

    def generate(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> tuple[np.ndarray, int]:
        # Detect backend from prompt metadata if possible
        _, metadata = load_prompt(prompt_path)
        backend = metadata.get(
            "backend", self._resolve_backend(),
        )
        if backend == "mlx-audio":
            return self._generate_mlx(
                text, prompt_path, language,
            )
        return self._generate_qwen(
            text, prompt_path, language,
        )

    def generate_streaming(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> Iterator[np.ndarray]:
        raise NotImplementedError(
            "Streaming not supported. "
            "See andimarafioti/faster-qwen3-tts "
            "for community streaming support."
        )
