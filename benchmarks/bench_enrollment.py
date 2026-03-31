from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import numpy as np
import numpy.typing as npt

from voxid.enrollment.preprocessor import AudioPreprocessor
from voxid.enrollment.quality_gate import _compute_speech_ratio

_SAMPLE_RATE = 24_000
# Resample benchmark: treat fixture audio as 16 kHz source → upsample to 24 kHz
_SOURCE_SR = 16_000


def _measure(
    fn: Callable[[], Any],
    iterations: int,
    warmup: int,
) -> dict[str, float]:
    """Run fn for warmup+iterations calls; return p50/p95/p99 wall-clock seconds."""
    for _ in range(warmup):
        fn()

    times: list[float] = []
    for _ in range(iterations):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)

    arr = np.array(times, dtype=np.float64)
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
        "p99": float(np.percentile(arr, 99)),
    }


def test_bench_trim_silence(
    audio_2s: npt.NDArray[np.float64],
    audio_15s: npt.NDArray[np.float64],
    audio_60s: npt.NDArray[np.float64],
    benchmark_config: dict[str, int],
    results_collector: dict[str, Any],
) -> None:
    preprocessor = AudioPreprocessor()
    iters = benchmark_config["iterations"]
    warmup = benchmark_config["warmup"]

    results_collector["trim_silence_2s"] = _measure(
        lambda: preprocessor.trim_silence(audio_2s, _SAMPLE_RATE),
        iters,
        warmup,
    )
    results_collector["trim_silence_15s"] = _measure(
        lambda: preprocessor.trim_silence(audio_15s, _SAMPLE_RATE),
        iters,
        warmup,
    )
    results_collector["trim_silence_60s"] = _measure(
        lambda: preprocessor.trim_silence(audio_60s, _SAMPLE_RATE),
        iters,
        warmup,
    )


def test_bench_compute_speech_ratio(
    audio_2s: npt.NDArray[np.float64],
    audio_15s: npt.NDArray[np.float64],
    audio_60s: npt.NDArray[np.float64],
    benchmark_config: dict[str, int],
    results_collector: dict[str, Any],
) -> None:
    iters = benchmark_config["iterations"]
    warmup = benchmark_config["warmup"]

    results_collector["compute_speech_ratio_2s"] = _measure(
        lambda: _compute_speech_ratio(audio_2s, _SAMPLE_RATE),
        iters,
        warmup,
    )
    results_collector["compute_speech_ratio_15s"] = _measure(
        lambda: _compute_speech_ratio(audio_15s, _SAMPLE_RATE),
        iters,
        warmup,
    )
    results_collector["compute_speech_ratio_60s"] = _measure(
        lambda: _compute_speech_ratio(audio_60s, _SAMPLE_RATE),
        iters,
        warmup,
    )


def test_bench_resample_linear(
    audio_2s: npt.NDArray[np.float64],
    audio_15s: npt.NDArray[np.float64],
    audio_60s: npt.NDArray[np.float64],
    benchmark_config: dict[str, int],
    results_collector: dict[str, Any],
) -> None:
    preprocessor = AudioPreprocessor()
    iters = benchmark_config["iterations"]
    warmup = benchmark_config["warmup"]

    results_collector["resample_linear_2s"] = _measure(
        lambda: preprocessor.resample(audio_2s, _SOURCE_SR, _SAMPLE_RATE),
        iters,
        warmup,
    )
    results_collector["resample_linear_15s"] = _measure(
        lambda: preprocessor.resample(audio_15s, _SOURCE_SR, _SAMPLE_RATE),
        iters,
        warmup,
    )
    results_collector["resample_linear_60s"] = _measure(
        lambda: preprocessor.resample(audio_60s, _SOURCE_SR, _SAMPLE_RATE),
        iters,
        warmup,
    )
