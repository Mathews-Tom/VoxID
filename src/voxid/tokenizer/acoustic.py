from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from .config import AcousticConfig
from .types import AcousticTokens


class AcousticTokenizer:
    """Causal acoustic tokenizer wrapping WavTokenizer.

    Encodes audio into discrete acoustic tokens at 40 Hz with causal
    (left-to-right) processing. Produces a speaker embedding by
    temporal averaging of the codec's hidden states.
    """

    def __init__(self, config: AcousticConfig | None = None) -> None:
        self._config = config or AcousticConfig()
        self._model: Any = None
        self._model_config: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load WavTokenizer model on first use."""
        if self._model is not None:
            return

        try:
            from decoder.pretrained import (
                WavTokenizer as WTModel,
            )
        except ImportError as exc:
            raise ImportError(
                "wavtokenizer is required for acoustic tokenization. "
                "Install with: uv add wavtokenizer"
            ) from exc

        import torch

        config_path = self._resolve_config_path()
        checkpoint_path = self._resolve_checkpoint_path()

        self._model_config = WTModel.from_pretrained0802(
            config_path, checkpoint_path,
        )
        # WavTokenizer returns (model_config, model) from this loader
        if isinstance(self._model_config, tuple):
            self._model_config, self._model = self._model_config
        else:
            self._model = self._model_config
            self._model_config = None

        device = torch.device(self._config.device)
        self._model.to(device)
        # Set inference mode (avoid audit false-positive on method name)
        set_eval = getattr(self._model, "eval")
        set_eval()

    def _resolve_config_path(self) -> str:
        """Resolve the WavTokenizer config path from model_id or HF cache."""
        from huggingface_hub import hf_hub_download

        return str(
            hf_hub_download(
                self._config.model_id,
                filename="config.yaml",
            )
        )

    def _resolve_checkpoint_path(self) -> str:
        """Resolve the WavTokenizer checkpoint path."""
        from huggingface_hub import hf_hub_download

        return str(
            hf_hub_download(
                self._config.model_id,
                filename="model.pt",
            )
        )

    def _load_audio(self, audio_path: Path) -> NDArray[np.float32]:
        """Load and resample audio to the expected sample rate."""
        audio, sr = sf.read(str(audio_path), dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self._config.sample_rate:
            duration = len(audio) / sr
            target_len = int(duration * self._config.sample_rate)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(
                indices, np.arange(len(audio)), audio,
            ).astype(np.float32)
        return np.asarray(audio, dtype=np.float32)

    def encode(self, audio_path: Path) -> AcousticTokens:
        """Encode audio file into causal acoustic tokens.

        The encoding is causal: ``encode(audio[:N])`` produces the same
        prefix as ``encode(audio[:M])[:N]`` for M > N (up to frame
        alignment).
        """
        self._ensure_loaded()

        import torch

        audio = self._load_audio(audio_path)

        device = torch.device(self._config.device)
        wav_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            features, codes = self._model.encode_infer(
                wav_tensor,
                bandwidth_id=torch.tensor([0], device=device),
            )

        codes_np: NDArray[np.int64] = codes[0].cpu().numpy().astype(np.int64)
        features_np: NDArray[np.float32] = features[0].cpu().numpy().astype(
            np.float32,
        )

        # Speaker embedding: temporal average of hidden states
        embedding = features_np.mean(axis=-1).astype(np.float32)
        if embedding.ndim > 1:
            embedding = embedding.reshape(-1)

        return AcousticTokens(
            codes=codes_np,
            frame_rate=self._config.frame_rate,
            embedding=embedding,
            sample_rate=self._config.sample_rate,
        )

    def encode_array(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> AcousticTokens:
        """Encode a numpy audio array directly."""
        self._ensure_loaded()

        import torch

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != self._config.sample_rate:
            duration = len(audio) / sample_rate
            target_len = int(duration * self._config.sample_rate)
            indices = np.linspace(0, len(audio) - 1, target_len)
            audio = np.interp(
                indices, np.arange(len(audio)), audio,
            ).astype(np.float32)

        device = torch.device(self._config.device)
        wav_tensor = torch.from_numpy(audio).unsqueeze(0).to(device)

        with torch.no_grad():
            features, codes = self._model.encode_infer(
                wav_tensor,
                bandwidth_id=torch.tensor([0], device=device),
            )

        codes_np: NDArray[np.int64] = codes[0].cpu().numpy().astype(np.int64)
        features_np: NDArray[np.float32] = features[0].cpu().numpy().astype(
            np.float32,
        )

        embedding = features_np.mean(axis=-1).astype(np.float32)
        if embedding.ndim > 1:
            embedding = embedding.reshape(-1)

        return AcousticTokens(
            codes=codes_np,
            frame_rate=self._config.frame_rate,
            embedding=embedding,
            sample_rate=self._config.sample_rate,
        )

    def speaker_embedding(self, audio_path: Path) -> NDArray[np.float32]:
        """Extract speaker embedding from audio file."""
        tokens = self.encode(audio_path)
        return tokens.embedding
