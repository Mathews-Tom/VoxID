from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class EngineCapabilities:
    supports_streaming: bool = False
    supports_emotion_control: bool = False
    supports_paralinguistic_tags: bool = False
    max_ref_audio_seconds: float = 30.0
    supported_languages: tuple[str, ...] = ("en",)
    streaming_latency_ms: int = 0
    supports_word_timing: bool = False


@runtime_checkable
class TTSEngineAdapter(Protocol):
    @property
    def engine_name(self) -> str: ...

    @property
    def capabilities(self) -> EngineCapabilities: ...

    def build_prompt(
        self,
        ref_audio: Path,
        ref_text: str,
        output_path: Path,
    ) -> Path:
        """Extract voice features, save as SafeTensors, return path."""
        ...

    def generate(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
        context_params: dict[str, float] | None = None,
    ) -> tuple[np.ndarray, int]:
        """Generate audio. Returns (waveform, sample_rate).

        context_params: optional continuity parameters from ContextConditioner.
            Keys may include "speed", "pitch_hz", "energy". Adapters that do
            not support context conditioning ignore this parameter.
        """
        ...

    def generate_streaming(
        self,
        text: str,
        prompt_path: Path,
        language: str = "en",
    ) -> Iterator[np.ndarray]:
        """Stream audio chunks."""
        ...
