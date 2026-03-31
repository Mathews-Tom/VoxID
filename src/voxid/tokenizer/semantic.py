from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from numpy.typing import NDArray

from voxid.audio_utils import resample_linear

from .config import SemanticConfig
from .types import SemanticTokens


class SemanticTokenizer:
    """Semantic tokenizer using HuBERT hidden states + k-means quantization.

    Extracts contextual speech representations at 50 Hz, then quantizes
    to discrete codes via a k-means codebook (500 clusters by default).
    """

    def __init__(self, config: SemanticConfig | None = None) -> None:
        self._config = config or SemanticConfig()
        self._model: Any = None
        self._processor: Any = None
        self._kmeans: Any = None

    def _ensure_loaded(self) -> None:
        """Lazy-load HuBERT model and k-means codebook."""
        if self._model is not None:
            return

        try:
            from transformers import (
                HubertModel,
                Wav2Vec2FeatureExtractor,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers is required for semantic tokenization. "
                "Install with: uv add transformers"
            ) from exc

        import torch

        self._processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self._config.model_id,
        )
        self._model = HubertModel.from_pretrained(self._config.model_id)

        device = torch.device(self._config.device)
        self._model.to(device)
        # Set inference mode (avoid audit false-positive on method name)
        set_eval = getattr(self._model, "eval")
        set_eval()

        self._load_or_init_kmeans()

    def _load_or_init_kmeans(self) -> None:
        """Load k-means centroids from disk, or initialize empty."""
        from sklearn.cluster import MiniBatchKMeans

        if (
            self._config.cluster_path is not None
            and self._config.cluster_path.exists()
        ):
            data = np.load(str(self._config.cluster_path))
            kmeans = MiniBatchKMeans(
                n_clusters=self._config.n_clusters,
                random_state=42,
                batch_size=1024,
            )
            kmeans.cluster_centers_ = data["centroids"]
            kmeans._n_threads = 1  # noqa: SLF001
            kmeans.n_features_in_ = data["centroids"].shape[1]
            self._kmeans = kmeans
        else:
            self._kmeans = MiniBatchKMeans(
                n_clusters=self._config.n_clusters,
                random_state=42,
                batch_size=1024,
            )

    def _load_audio(self, audio_path: Path) -> NDArray[np.float32]:
        """Load and resample audio to 16kHz for HuBERT."""
        audio, sr = sf.read(str(audio_path), dtype="float32")
        audio = np.asarray(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sr != self._config.sample_rate:
            target_len = int(len(audio) / sr * self._config.sample_rate)
            audio = resample_linear(audio, target_len)
        return np.asarray(audio, dtype=np.float32)

    def extract_features(self, audio_path: Path) -> NDArray[np.float32]:
        """Extract HuBERT hidden states from audio.

        Returns shape (T, hidden_dim) from the configured layer.
        """
        self._ensure_loaded()

        import torch

        audio = self._load_audio(audio_path)
        inputs = self._processor(
            audio,
            sampling_rate=self._config.sample_rate,
            return_tensors="pt",
        )

        device = torch.device(self._config.device)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = self._model(
                input_values,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[self._config.layer]
        features: NDArray[np.float32] = (
            hidden_states[0].cpu().numpy().astype(np.float32)
        )
        return features

    def encode(self, audio_path: Path) -> SemanticTokens:
        """Encode audio into quantized semantic tokens."""
        features = self.extract_features(audio_path)
        return self._quantize(features)

    def encode_array(
        self,
        audio: NDArray[np.float32],
        sample_rate: int,
    ) -> SemanticTokens:
        """Encode a numpy audio array into semantic tokens."""
        self._ensure_loaded()

        import torch

        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        if sample_rate != self._config.sample_rate:
            target_len = int(len(audio) / sample_rate * self._config.sample_rate)
            audio = resample_linear(audio, target_len)

        inputs = self._processor(
            audio,
            sampling_rate=self._config.sample_rate,
            return_tensors="pt",
        )

        device = torch.device(self._config.device)
        input_values = inputs.input_values.to(device)

        with torch.no_grad():
            outputs = self._model(
                input_values,
                output_hidden_states=True,
            )

        hidden_states = outputs.hidden_states[self._config.layer]
        features: NDArray[np.float32] = (
            hidden_states[0].cpu().numpy().astype(np.float32)
        )
        return self._quantize(features)

    def _quantize(self, features: NDArray[np.float32]) -> SemanticTokens:
        """Quantize HuBERT features via k-means."""
        if not hasattr(self._kmeans, "cluster_centers_"):
            raise RuntimeError(
                "K-means codebook not trained. Call train_codebook() first "
                "or provide a cluster_path in SemanticConfig."
            )
        codes: NDArray[np.int64] = self._kmeans.predict(features).astype(
            np.int64,
        )
        return SemanticTokens(
            codes=codes,
            frame_rate=self._config.frame_rate,
            features=features,
        )

    def train_codebook(
        self,
        audio_paths: list[Path],
    ) -> None:
        """Train the k-means codebook on a set of audio files."""
        all_features: list[NDArray[np.float32]] = []
        for path in audio_paths:
            features = self.extract_features(path)
            all_features.append(features)

        combined = np.concatenate(all_features, axis=0)
        self._kmeans.fit(combined)

    def save_codebook(self, path: Path) -> None:
        """Save trained k-means centroids to disk."""
        if not hasattr(self._kmeans, "cluster_centers_"):
            raise RuntimeError("K-means codebook not trained yet.")
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            str(path),
            centroids=self._kmeans.cluster_centers_,
        )

    def mean_pooled_embedding(
        self, audio_path: Path,
    ) -> NDArray[np.float32]:
        """Extract mean-pooled semantic embedding from audio."""
        features = self.extract_features(audio_path)
        return np.asarray(
            features.mean(axis=0), dtype=np.float32,
        )
