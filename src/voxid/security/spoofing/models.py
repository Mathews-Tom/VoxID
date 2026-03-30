from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from .config import SpoofingConfig
from .features import extract_lfcc, extract_mel_spectrogram

logger = logging.getLogger(__name__)


class SpoofModel(Protocol):
    """Common interface for anti-spoofing model wrappers."""

    name: str

    def predict(self, audio: NDArray[np.float32], sr: int) -> float:
        """Return spoofing probability in [0, 1]. Higher = more likely synthetic."""
        ...


def _select_device() -> str:
    """Select best available torch device: CUDA > MPS > CPU."""
    import torch  # type: ignore[import-not-found,unused-ignore]

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class AASISTWrapper:
    """AASIST (Audio Anti-Spoofing using Integrated Spectro-Temporal) model.

    Expects pretrained weights at ``{weights_dir}/aasist.pth``.
    Input: log-mel spectrogram (80-band). Architecture uses spectral
    and temporal graph attention to detect spoofing artifacts.
    """

    name: str = "aasist"

    def __init__(self, config: SpoofingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._device: str = "cpu"

    def _weights_path(self) -> Path:
        return self._config.weights_dir / "aasist.pth"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch  # type: ignore[import-not-found,unused-ignore]

        weights_path = self._weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"AASIST weights not found at {weights_path}. "
                "Download ASVspoof 2021 pretrained weights and place "
                f"them at: {weights_path}"
            )

        self._device = _select_device()
        self._model = torch.jit.load(  # type: ignore[no-untyped-call]
            str(weights_path), map_location=self._device
        )
        self._model.eval()
        logger.info("AASIST loaded on %s from %s", self._device, weights_path)

    def predict(self, audio: NDArray[np.float32], sr: int) -> float:
        """Score audio using AASIST. Returns spoofing probability [0, 1]."""
        self._ensure_loaded()

        import torch  # type: ignore[import-not-found,unused-ignore]

        mel = extract_mel_spectrogram(audio, sr)
        # Model expects (batch, 1, n_mels, T)
        tensor = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            # Assume logits shape (batch, 2): [genuine_score, spoof_score]
            probs = torch.softmax(logits, dim=-1)
            spoof_prob: float = probs[0, 1].item()

        return spoof_prob


class RawNet2Wrapper:
    """RawNet2 model wrapper for raw-waveform anti-spoofing.

    Processes raw audio in 4-second chunks with 50% overlap.
    Expects pretrained weights at ``{weights_dir}/rawnet2.pth``.
    """

    name: str = "rawnet2"

    def __init__(self, config: SpoofingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._device: str = "cpu"

    def _weights_path(self) -> Path:
        return self._config.weights_dir / "rawnet2.pth"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch  # type: ignore[import-not-found,unused-ignore]

        weights_path = self._weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"RawNet2 weights not found at {weights_path}. "
                "Download pretrained weights and place "
                f"them at: {weights_path}"
            )

        self._device = _select_device()
        self._model = torch.jit.load(  # type: ignore[no-untyped-call]
            str(weights_path), map_location=self._device
        )
        self._model.eval()
        logger.info("RawNet2 loaded on %s from %s", self._device, weights_path)

    def _chunk_audio(
        self, audio: NDArray[np.float32]
    ) -> list[NDArray[np.float32]]:
        """Split audio into overlapping chunks per config."""
        chunk_samples = int(
            self._config.chunk_duration_s * self._config.sample_rate
        )
        hop_samples = int(chunk_samples * (1.0 - self._config.chunk_overlap))

        if len(audio) <= chunk_samples:
            # Pad short audio to chunk length
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[: len(audio)] = audio
            return [padded]

        chunks: list[NDArray[np.float32]] = []
        for start in range(0, len(audio) - chunk_samples + 1, hop_samples):
            chunks.append(audio[start : start + chunk_samples])

        return chunks

    def predict(self, audio: NDArray[np.float32], sr: int) -> float:
        """Score audio using RawNet2. Returns spoofing probability [0, 1].

        Processes in 4s chunks with 50% overlap and averages scores.
        """
        self._ensure_loaded()

        import torch  # type: ignore[import-not-found,unused-ignore]

        chunks = self._chunk_audio(audio)
        scores: list[float] = []

        for chunk in chunks:
            tensor = (
                torch.from_numpy(chunk).unsqueeze(0).unsqueeze(0).to(self._device)
            )
            with torch.no_grad():
                logits = self._model(tensor)
                probs = torch.softmax(logits, dim=-1)
                scores.append(probs[0, 1].item())

        return float(np.mean(scores))


class LCNNWrapper:
    """Light CNN (LCNN) model wrapper for anti-spoofing.

    Uses LFCC features as input. Lightweight architecture suitable
    for resource-constrained environments.
    Expects pretrained weights at ``{weights_dir}/lcnn.pth``.
    """

    name: str = "lcnn"

    def __init__(self, config: SpoofingConfig) -> None:
        self._config = config
        self._model: Any = None
        self._device: str = "cpu"

    def _weights_path(self) -> Path:
        return self._config.weights_dir / "lcnn.pth"

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return

        import torch  # type: ignore[import-not-found,unused-ignore]

        weights_path = self._weights_path()
        if not weights_path.exists():
            raise FileNotFoundError(
                f"LCNN weights not found at {weights_path}. "
                "Download pretrained weights and place "
                f"them at: {weights_path}"
            )

        self._device = _select_device()
        self._model = torch.jit.load(  # type: ignore[no-untyped-call]
            str(weights_path), map_location=self._device
        )
        self._model.eval()
        logger.info("LCNN loaded on %s from %s", self._device, weights_path)

    def predict(self, audio: NDArray[np.float32], sr: int) -> float:
        """Score audio using LCNN with LFCC features. Returns [0, 1]."""
        self._ensure_loaded()

        import torch  # type: ignore[import-not-found,unused-ignore]

        lfcc = extract_lfcc(audio, sr)
        # Model expects (batch, 1, n_lfcc, T)
        tensor = torch.from_numpy(lfcc).unsqueeze(0).unsqueeze(0).to(self._device)

        with torch.no_grad():
            logits = self._model(tensor)
            probs = torch.softmax(logits, dim=-1)
            spoof_prob: float = probs[0, 1].item()

        return spoof_prob


def load_available_models(
    config: SpoofingConfig,
) -> list[AASISTWrapper | RawNet2Wrapper | LCNNWrapper]:
    """Instantiate model wrappers for which weight files exist.

    Does not load weights — that happens lazily on first ``predict()``.
    Returns only models whose weight files are present on disk.
    """
    candidates: list[AASISTWrapper | RawNet2Wrapper | LCNNWrapper] = [
        AASISTWrapper(config),
        RawNet2Wrapper(config),
        LCNNWrapper(config),
    ]
    available = [m for m in candidates if m._weights_path().exists()]

    for model in available:
        logger.info("Anti-spoofing model available: %s", model.name)

    return available
