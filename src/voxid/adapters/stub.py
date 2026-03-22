from __future__ import annotations

import math
from collections.abc import Iterator
from pathlib import Path

import numpy as np

from ..serialization import save_prompt
from . import register_adapter
from .protocol import EngineCapabilities


@register_adapter
class StubAdapter:
    """Test adapter generating sine wave audio. No model required."""

    engine_name: str = "stub"

    @property
    def capabilities(self) -> EngineCapabilities:
        return EngineCapabilities(
            supports_streaming=True,
            supports_emotion_control=False,
            supports_paralinguistic_tags=False,
            max_ref_audio_seconds=60.0,
            supported_languages=("en", "zh", "ja"),
            streaming_latency_ms=0,
            supports_word_timing=False,
        )

    def build_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
    ) -> Path:
        tensors: dict[str, np.ndarray] = {
            "ref_spk_embedding": np.zeros(192, dtype=np.float32),
        }
        metadata: dict[str, str] = {
            "engine": "stub",
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
        sr = 24000
        duration = max(0.5, len(text) * 0.05)  # ~50ms per char
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)
        waveform = (0.3 * np.sin(2 * math.pi * 440 * t)).astype(np.float32)
        return waveform, sr

    def generate_streaming(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> Iterator[np.ndarray]:
        waveform, _ = self.generate(text, prompt_path, language)
        chunk_size = 4800  # 200ms chunks at 24kHz
        for i in range(0, len(waveform), chunk_size):
            yield waveform[i : i + chunk_size]
