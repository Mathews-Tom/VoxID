from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from voxid.security.spoofing.config import SpoofingConfig
from voxid.security.spoofing.detector import SynthesisDetector
from voxid.security.spoofing.diffusion import (
    DiffusionArtifactAnalyzer,
    _find_spectral_peaks,
)
from voxid.security.spoofing.types import (
    ArtifactType,
    SpoofDecision,
    SpoofingUnavailableError,
    SpoofLabel,
)


class TestSpoofLabel:
    def test_enum_values(self) -> None:
        assert SpoofLabel.GENUINE.value == "genuine"
        assert SpoofLabel.SYNTHETIC.value == "synthetic"
        assert SpoofLabel.UNCERTAIN.value == "uncertain"


class TestArtifactType:
    def test_enum_values(self) -> None:
        assert ArtifactType.VOCODER.value == "vocoder"
        assert ArtifactType.DIFFUSION.value == "diffusion"
        assert ArtifactType.AUTOREGRESSIVE.value == "autoregressive"
        assert ArtifactType.CONCATENATIVE.value == "concatenative"
        assert ArtifactType.UNKNOWN.value == "unknown"


class TestSpoofDecision:
    def test_frozen_dataclass(self) -> None:
        decision = SpoofDecision(
            label=SpoofLabel.GENUINE,
            score=0.1,
            artifact_type=ArtifactType.UNKNOWN,
            model_scores={"aasist": 0.1},
            confidence=1.0,
        )
        assert decision.label == SpoofLabel.GENUINE
        with pytest.raises(AttributeError):
            decision.score = 0.5  # type: ignore[misc]


class TestSpoofingConfig:
    def test_defaults(self) -> None:
        config = SpoofingConfig()
        assert config.synthetic_threshold == 0.7
        assert config.uncertain_threshold == 0.4
        assert config.sample_rate == 16000
        assert config.chunk_duration_s == 4.0
        assert config.chunk_overlap == 0.5
        assert "aasist" in config.ensemble_weights
        assert "rawnet2" in config.ensemble_weights
        assert "lcnn" in config.ensemble_weights

    def test_custom_thresholds(self) -> None:
        config = SpoofingConfig(
            synthetic_threshold=0.8,
            uncertain_threshold=0.5,
        )
        assert config.synthetic_threshold == 0.8
        assert config.uncertain_threshold == 0.5


class TestSynthesisDetectorInit:
    def test_raises_when_no_models(self, tmp_path: Path) -> None:
        config = SpoofingConfig(weights_dir=tmp_path)
        with pytest.raises(SpoofingUnavailableError, match="No anti-spoofing models"):
            SynthesisDetector(config)


class TestSynthesisDetectorClassification:
    """Test the internal classification logic by constructing a detector
    with mocked model loading, then calling internal methods directly."""

    @pytest.fixture
    def config(self) -> SpoofingConfig:
        return SpoofingConfig()

    def test_classify_genuine(self, config: SpoofingConfig) -> None:
        detector = _build_detector_no_models(config)
        label = detector._classify(0.2, 0.0)
        assert label == SpoofLabel.GENUINE

    def test_classify_uncertain(self, config: SpoofingConfig) -> None:
        detector = _build_detector_no_models(config)
        label = detector._classify(0.5, 0.3)
        assert label == SpoofLabel.UNCERTAIN

    def test_classify_synthetic_needs_agreement(
        self, config: SpoofingConfig
    ) -> None:
        detector = _build_detector_no_models(config)
        # High score but low agreement → uncertain
        label = detector._classify(0.8, 0.3)
        assert label == SpoofLabel.UNCERTAIN

    def test_classify_synthetic_with_agreement(
        self, config: SpoofingConfig
    ) -> None:
        detector = _build_detector_no_models(config)
        label = detector._classify(0.8, 0.6)
        assert label == SpoofLabel.SYNTHETIC

    def test_weighted_score_single_model(
        self, config: SpoofingConfig
    ) -> None:
        detector = _build_detector_no_models(config)
        score = detector._weighted_score({"aasist": 0.9})
        assert score == pytest.approx(0.9)

    def test_weighted_score_all_models(
        self, config: SpoofingConfig
    ) -> None:
        detector = _build_detector_no_models(config)
        scores = {"aasist": 0.8, "rawnet2": 0.6, "lcnn": 0.4}
        # 0.4*0.8 + 0.35*0.6 + 0.25*0.4 = 0.32 + 0.21 + 0.10 = 0.63
        expected = 0.63
        result = detector._weighted_score(scores)
        assert result == pytest.approx(expected)

    def test_compute_agreement(self, config: SpoofingConfig) -> None:
        detector = _build_detector_no_models(config)
        agreement = detector._compute_agreement(
            {"aasist": 0.9, "rawnet2": 0.8, "lcnn": 0.3}
        )
        # 2 out of 3 above 0.7
        assert agreement == pytest.approx(2 / 3)

    def test_infer_artifact_type_vocoder(
        self, config: SpoofingConfig
    ) -> None:
        detector = _build_detector_no_models(config)
        result = detector._infer_artifact_type(
            {"aasist": 0.9, "rawnet2": 0.5, "lcnn": 0.4}
        )
        assert result == ArtifactType.VOCODER

    def test_infer_artifact_type_diffusion(
        self, config: SpoofingConfig
    ) -> None:
        detector = _build_detector_no_models(config)
        result = detector._infer_artifact_type(
            {"aasist": 0.9, "rawnet2": 0.85, "lcnn": 0.9}
        )
        assert result == ArtifactType.DIFFUSION


class TestDiffusionArtifactAnalyzer:
    @pytest.fixture
    def analyzer(self) -> DiffusionArtifactAnalyzer:
        return DiffusionArtifactAnalyzer()

    @pytest.fixture
    def sine_wave(self) -> tuple[np.ndarray, int]:
        sr = 16000
        t = np.linspace(0, 2.0, sr * 2, endpoint=False, dtype=np.float32)
        audio = np.sin(2 * np.pi * 440 * t).astype(np.float32)
        return audio, sr

    def test_analyze_returns_analysis(
        self,
        analyzer: DiffusionArtifactAnalyzer,
        sine_wave: tuple[np.ndarray, int],
    ) -> None:
        audio, sr = sine_wave
        result = analyzer.analyze(audio, sr)
        assert isinstance(result.spectral_smoothness, float)
        assert isinstance(result.temporal_discontinuity, float)
        assert isinstance(result.harmonic_regularity, float)
        assert isinstance(result.is_suspicious, bool)

    def test_analyze_metrics_finite(
        self,
        analyzer: DiffusionArtifactAnalyzer,
        sine_wave: tuple[np.ndarray, int],
    ) -> None:
        audio, sr = sine_wave
        result = analyzer.analyze(audio, sr)
        assert np.isfinite(result.spectral_smoothness)
        assert np.isfinite(result.temporal_discontinuity)
        assert np.isfinite(result.harmonic_regularity)

    def test_custom_thresholds(self) -> None:
        analyzer = DiffusionArtifactAnalyzer(
            smoothness_threshold=0.5,
            discontinuity_threshold=10.0,
            regularity_threshold=0.01,
        )
        audio = np.random.default_rng(42).standard_normal(32000).astype(
            np.float32
        )
        result = analyzer.analyze(audio, 16000)
        assert isinstance(result.is_suspicious, bool)


class TestFindSpectralPeaks:
    def test_finds_peaks_in_sine(self) -> None:
        # Generate spectrum with a clear peak
        n = 512
        freqs = np.zeros(n, dtype=np.float32)
        freqs[100] = 1.0
        freqs[200] = 0.5
        peaks = _find_spectral_peaks(freqs, min_prominence=0.1)
        assert 100 in peaks
        assert 200 in peaks

    def test_empty_spectrum(self) -> None:
        peaks = _find_spectral_peaks(np.array([], dtype=np.float32))
        assert len(peaks) == 0

    def test_short_spectrum(self) -> None:
        peaks = _find_spectral_peaks(np.array([1.0, 2.0], dtype=np.float32))
        assert len(peaks) == 0


class TestSynthesisDetectorDetect:
    """Test the full detect() flow with fake models."""

    def test_detect_genuine(self) -> None:
        config = SpoofingConfig()
        models = [_FakeModel("aasist", 0.1), _FakeModel("rawnet2", 0.2)]
        with patch(
            "voxid.security.spoofing.detector.load_available_models",
            return_value=models,
        ):
            detector = SynthesisDetector(config)
        audio = np.random.default_rng(42).standard_normal(16000).astype(
            np.float32
        )
        decision = detector.detect(audio, 16000)
        assert decision.label == SpoofLabel.GENUINE
        assert "aasist" in decision.model_scores
        assert "rawnet2" in decision.model_scores

    def test_detect_synthetic(self) -> None:
        config = SpoofingConfig()
        models = [_FakeModel("aasist", 0.9), _FakeModel("rawnet2", 0.85)]
        with patch(
            "voxid.security.spoofing.detector.load_available_models",
            return_value=models,
        ):
            detector = SynthesisDetector(config)
        audio = np.random.default_rng(42).standard_normal(16000).astype(
            np.float32
        )
        decision = detector.detect(audio, 16000)
        assert decision.label == SpoofLabel.SYNTHETIC
        assert decision.artifact_type != ArtifactType.UNKNOWN

    def test_available_models_property(self) -> None:
        config = SpoofingConfig()
        models = [_FakeModel("aasist", 0.5)]
        with patch(
            "voxid.security.spoofing.detector.load_available_models",
            return_value=models,
        ):
            detector = SynthesisDetector(config)
        assert detector.available_models == ["aasist"]

    def test_detect_with_resample(self) -> None:
        config = SpoofingConfig(sample_rate=16000)
        models = [_FakeModel("aasist", 0.1)]
        with patch(
            "voxid.security.spoofing.detector.load_available_models",
            return_value=models,
        ):
            detector = SynthesisDetector(config)
        # Audio at different sample rate triggers resample
        audio = np.random.default_rng(42).standard_normal(48000).astype(
            np.float32
        )
        decision = detector.detect(audio, 48000)
        assert decision.label == SpoofLabel.GENUINE

    def test_infer_autoregressive(self) -> None:
        config = SpoofingConfig()
        detector = _build_detector_no_models(config)
        result = detector._infer_artifact_type(
            {"aasist": 0.5, "rawnet2": 0.9, "lcnn": 0.4}
        )
        assert result == ArtifactType.AUTOREGRESSIVE

    def test_infer_concatenative(self) -> None:
        config = SpoofingConfig()
        detector = _build_detector_no_models(config)
        result = detector._infer_artifact_type(
            {"aasist": 0.4, "rawnet2": 0.5, "lcnn": 0.9}
        )
        assert result == ArtifactType.CONCATENATIVE

    def test_infer_unknown_equal_scores(self) -> None:
        config = SpoofingConfig()
        detector = _build_detector_no_models(config)
        result = detector._infer_artifact_type(
            {"aasist": 0.5, "rawnet2": 0.5, "lcnn": 0.5}
        )
        # Equal scores — no clear winner, first condition (diffusion)
        # won't match (min < 0.8), then all comparisons are equal
        assert result in (
            ArtifactType.UNKNOWN,
            ArtifactType.VOCODER,
            ArtifactType.AUTOREGRESSIVE,
            ArtifactType.CONCATENATIVE,
        )

    def test_weighted_score_no_weights(self) -> None:
        config = SpoofingConfig()
        detector = _build_detector_no_models(config)
        score = detector._weighted_score({"unknown_model": 0.7})
        # Falls back to simple mean since weight is 0
        assert score == pytest.approx(0.7)

    def test_compute_agreement_empty(self) -> None:
        config = SpoofingConfig()
        detector = _build_detector_no_models(config)
        assert detector._compute_agreement({}) == 0.0


class TestModelWrappers:
    """Test model wrapper methods that don't require torch."""

    def test_aasist_weights_path(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import AASISTWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        wrapper = AASISTWrapper(config)
        assert wrapper._weights_path() == tmp_path / "aasist.pth"
        assert wrapper.name == "aasist"

    def test_rawnet2_weights_path(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import RawNet2Wrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        wrapper = RawNet2Wrapper(config)
        assert wrapper._weights_path() == tmp_path / "rawnet2.pth"
        assert wrapper.name == "rawnet2"

    def test_lcnn_weights_path(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import LCNNWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        wrapper = LCNNWrapper(config)
        assert wrapper._weights_path() == tmp_path / "lcnn.pth"
        assert wrapper.name == "lcnn"

    def test_aasist_missing_weights_raises(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import AASISTWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        wrapper = AASISTWrapper(config)
        with pytest.raises(FileNotFoundError, match="AASIST weights"):
            wrapper._ensure_loaded()

    def test_rawnet2_missing_weights_raises(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import RawNet2Wrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        wrapper = RawNet2Wrapper(config)
        with pytest.raises(FileNotFoundError, match="RawNet2 weights"):
            wrapper._ensure_loaded()

    def test_lcnn_missing_weights_raises(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import LCNNWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        wrapper = LCNNWrapper(config)
        with pytest.raises(FileNotFoundError, match="LCNN weights"):
            wrapper._ensure_loaded()

    def test_rawnet2_chunk_audio_short(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import RawNet2Wrapper

        config = SpoofingConfig(
            weights_dir=tmp_path,
            chunk_duration_s=4.0,
            chunk_overlap=0.5,
            sample_rate=16000,
        )
        wrapper = RawNet2Wrapper(config)
        # Short audio gets zero-padded to one chunk
        audio = np.ones(1000, dtype=np.float32)
        chunks = wrapper._chunk_audio(audio)
        assert len(chunks) == 1
        assert len(chunks[0]) == 64000  # 4s * 16000

    def test_rawnet2_chunk_audio_long(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import RawNet2Wrapper

        config = SpoofingConfig(
            weights_dir=tmp_path,
            chunk_duration_s=4.0,
            chunk_overlap=0.5,
            sample_rate=16000,
        )
        wrapper = RawNet2Wrapper(config)
        # 10 seconds of audio → multiple chunks
        audio = np.ones(160000, dtype=np.float32)
        chunks = wrapper._chunk_audio(audio)
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) == 64000

    def test_load_available_models_none(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import load_available_models

        config = SpoofingConfig(weights_dir=tmp_path)
        models = load_available_models(config)
        assert len(models) == 0

    def test_load_available_models_partial(self, tmp_path: Path) -> None:
        from voxid.security.spoofing.models import load_available_models

        config = SpoofingConfig(weights_dir=tmp_path)
        # Create only aasist weights file
        (tmp_path / "aasist.pth").touch()
        models = load_available_models(config)
        assert len(models) == 1
        assert models[0].name == "aasist"

    def test_select_device_cpu(self) -> None:
        from voxid.security.spoofing.models import _select_device

        device = _select_device()
        assert device in ("cuda", "mps", "cpu")


class TestModelPredictWithMockedTorch:
    """Test model predict paths by mocking torch.jit.load."""

    def test_aasist_predict(self, tmp_path: Path) -> None:
        import torch

        from voxid.security.spoofing.models import AASISTWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        (tmp_path / "aasist.pth").touch()

        fake_output = torch.tensor([[0.3, 0.7]])
        mock_model = _MockTorchModel(fake_output)

        wrapper = AASISTWrapper(config)
        with patch("torch.jit.load", return_value=mock_model):
            score = wrapper.predict(
                np.random.default_rng(1).standard_normal(16000).astype(
                    np.float32
                ),
                16000,
            )
        assert 0.0 <= score <= 1.0

    def test_rawnet2_predict(self, tmp_path: Path) -> None:
        import torch

        from voxid.security.spoofing.models import RawNet2Wrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        (tmp_path / "rawnet2.pth").touch()

        fake_output = torch.tensor([[0.6, 0.4]])
        mock_model = _MockTorchModel(fake_output)

        wrapper = RawNet2Wrapper(config)
        with patch("torch.jit.load", return_value=mock_model):
            score = wrapper.predict(
                np.random.default_rng(1).standard_normal(16000).astype(
                    np.float32
                ),
                16000,
            )
        assert 0.0 <= score <= 1.0

    def test_lcnn_predict(self, tmp_path: Path) -> None:
        import torch

        from voxid.security.spoofing.models import LCNNWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        (tmp_path / "lcnn.pth").touch()

        fake_output = torch.tensor([[0.2, 0.8]])
        mock_model = _MockTorchModel(fake_output)

        wrapper = LCNNWrapper(config)
        with patch("torch.jit.load", return_value=mock_model):
            score = wrapper.predict(
                np.random.default_rng(1).standard_normal(16000).astype(
                    np.float32
                ),
                16000,
            )
        assert 0.0 <= score <= 1.0

    def test_aasist_already_loaded_skips_reload(self, tmp_path: Path) -> None:
        import torch

        from voxid.security.spoofing.models import AASISTWrapper

        config = SpoofingConfig(weights_dir=tmp_path)
        (tmp_path / "aasist.pth").touch()

        fake_output = torch.tensor([[0.5, 0.5]])
        mock_model = _MockTorchModel(fake_output)

        wrapper = AASISTWrapper(config)
        with patch("torch.jit.load", return_value=mock_model) as mock_load:
            wrapper.predict(
                np.ones(16000, dtype=np.float32), 16000
            )
            wrapper.predict(
                np.ones(16000, dtype=np.float32), 16000
            )
            assert mock_load.call_count == 1


class _MockTorchModel:
    """Mock torch model that returns fixed output."""

    def __init__(self, output: object) -> None:
        self._output = output

    def eval(self) -> None:
        pass

    def __call__(self, x: object) -> object:
        return self._output


class TestDiffusionEdgeCases:
    """Test diffusion analyzer edge cases for coverage."""

    def test_short_mel_smoothness(self) -> None:
        analyzer = DiffusionArtifactAnalyzer()
        # Very short audio
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = analyzer.analyze(audio, 16000)
        assert isinstance(result.spectral_smoothness, float)

    def test_short_mel_discontinuity(self) -> None:
        analyzer = DiffusionArtifactAnalyzer()
        # Very short audio — mel has few frames
        audio = np.ones(512, dtype=np.float32) * 0.1
        result = analyzer.analyze(audio, 16000)
        assert np.isfinite(result.temporal_discontinuity)

    def test_short_audio_regularity(self) -> None:
        analyzer = DiffusionArtifactAnalyzer()
        audio = np.ones(100, dtype=np.float32) * 0.1
        result = analyzer.analyze(audio, 16000)
        # Short audio → regularity returns 1.0
        assert result.harmonic_regularity == 1.0

    def test_silence_handling(self) -> None:
        analyzer = DiffusionArtifactAnalyzer()
        audio = np.zeros(16000, dtype=np.float32)
        result = analyzer.analyze(audio, 16000)
        assert isinstance(result.is_suspicious, bool)

    def test_no_peaks_regularity(self) -> None:
        analyzer = DiffusionArtifactAnalyzer()
        # Audio with no spectral peaks (constant signal)
        audio = np.ones(32000, dtype=np.float32) * 0.001
        result = analyzer.analyze(audio, 16000)
        assert np.isfinite(result.harmonic_regularity)


def _build_detector_no_models(config: SpoofingConfig) -> SynthesisDetector:
    """Build a SynthesisDetector with empty model list for testing
    classification logic without requiring weight files."""
    with patch(
        "voxid.security.spoofing.detector.load_available_models",
        return_value=[_FakeModel("aasist")],
    ):
        return SynthesisDetector(config)


class _FakeModel:
    """Minimal model stand-in that satisfies the SpoofModel protocol."""

    def __init__(self, name: str, score: float = 0.0) -> None:
        self.name = name
        self._score = score

    def predict(self, audio: np.ndarray, sr: int) -> float:
        return self._score

    def _weights_path(self) -> Path:
        return Path("/fake") / f"{self.name}.pth"
