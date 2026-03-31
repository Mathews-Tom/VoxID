from __future__ import annotations

import json
import os
import platform
from collections.abc import Generator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest

_SAMPLE_RATE = 24_000
_DEFAULT_RESULTS_FILE = os.environ.get("BENCHMARK_OUTPUT", "baseline.json")
_RESULTS_PATH = Path(__file__).parent / "results" / _DEFAULT_RESULTS_FILE


def _make_audio(seconds: int) -> npt.NDArray[np.float64]:
    """Synthetic 24kHz mono audio: 440 Hz sine + white noise, seed=42."""
    rng = np.random.default_rng(42)
    n = seconds * _SAMPLE_RATE
    t = np.linspace(0, seconds, n, endpoint=False)
    signal = np.sin(2 * np.pi * 440 * t)
    noise = rng.normal(0.0, 0.01, n)
    result: npt.NDArray[np.float64] = (signal + noise).astype(np.float64)
    return result


@pytest.fixture(scope="session")
def audio_2s() -> npt.NDArray[np.float64]:
    return _make_audio(2)


@pytest.fixture(scope="session")
def audio_15s() -> npt.NDArray[np.float64]:
    return _make_audio(15)


@pytest.fixture(scope="session")
def audio_60s() -> npt.NDArray[np.float64]:
    return _make_audio(60)


@pytest.fixture(scope="session")
def benchmark_config() -> dict[str, int]:
    return {"iterations": 100, "warmup": 5}


@pytest.fixture(scope="session")
def results_collector() -> dict[str, Any]:
    return {}


def _ram_gb() -> float:
    """Estimate physical RAM in GB without third-party dependencies."""
    try:
        page_size = int(os.sysconf("SC_PAGE_SIZE"))
        phys_pages = int(os.sysconf("SC_PHYS_PAGES"))
        return round(page_size * phys_pages / (1024**3), 1)
    except (AttributeError, ValueError, OSError):
        return 0.0


@pytest.fixture(scope="session", autouse=True)
def write_results(
    results_collector: dict[str, Any],
) -> Generator[None, None, None]:
    """Write benchmark results to the configured output file after the session ends."""
    yield

    cpu = platform.processor() or platform.machine() or "unknown"
    output: dict[str, Any] = {
        "hardware": {
            "cpu": cpu,
            "ram_gb": _ram_gb(),
            "os": f"{platform.system()} {platform.release()}",
        },
        "timestamp": datetime.now(tz=UTC).isoformat(),
        "results": results_collector,
    }
    _RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    _RESULTS_PATH.write_text(json.dumps(output, indent=2))
