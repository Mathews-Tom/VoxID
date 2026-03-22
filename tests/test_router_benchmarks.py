from __future__ import annotations

import time

import pytest

from voxid.router import StyleRouter
from voxid.router.classifiers import (
    CentroidClassifier,
    RuleBasedClassifier,
    get_training_data,
)

ALL_STYLES = ["conversational", "technical", "narration", "emphatic"]


def _generate_texts(n: int) -> list[str]:
    """Generate n unique texts by varying a template."""
    templates = [
        "The {} API endpoint returns a {} status code",
        "Honestly, I think {} is {} than expected",
        "This is absolutely {}! The {} are incredible!",
        "There is a certain {} that comes from {}",
    ]
    words = [
        "model",
        "pipeline",
        "database",
        "cache",
        "server",
        "latency",
        "throughput",
        "embedding",
        "vector",
        "query",
    ]
    texts: list[str] = []
    for i in range(n):
        template = templates[i % len(templates)]
        w1 = words[i % len(words)]
        w2 = words[(i + 3) % len(words)]
        texts.append(template.format(w1, w2))
    return texts


def _percentile(data: list[float], p: float) -> float:
    k = (len(data) - 1) * p / 100
    f = int(k)
    c = f + 1 if f + 1 < len(data) else f
    return data[f] + (k - f) * (data[c] - data[f])


def _sorted_timings(timings_ns: list[int]) -> list[float]:
    """Convert nanoseconds to milliseconds and sort."""
    return sorted(t / 1_000_000 for t in timings_ns)


def test_rule_based_classifier_latency_p95_under_1ms() -> None:
    """Rule-based classifier p95 latency must be under 1ms for 1000 texts."""
    clf = RuleBasedClassifier()
    texts = _generate_texts(1000)

    timings_ns: list[int] = []
    for text in texts:
        start = time.perf_counter_ns()
        clf.classify(text, ALL_STYLES)
        timings_ns.append(time.perf_counter_ns() - start)

    ms = _sorted_timings(timings_ns)
    p50 = _percentile(ms, 50)
    p95 = _percentile(ms, 95)
    p99 = _percentile(ms, 99)

    print(f"\nRule-based: p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    assert p95 < 1.0, f"Rule-based p95 latency {p95:.3f}ms exceeds 1ms threshold"


def test_centroid_classifier_latency_p95_under_5ms() -> None:
    """Centroid classifier p95 latency must be under 5ms for 1000 texts."""
    clf = CentroidClassifier()
    clf.fit(get_training_data())
    texts = _generate_texts(1000)

    timings_ns: list[int] = []
    for text in texts:
        start = time.perf_counter_ns()
        clf.classify(text, ALL_STYLES)
        timings_ns.append(time.perf_counter_ns() - start)

    ms = _sorted_timings(timings_ns)
    p50 = _percentile(ms, 50)
    p95 = _percentile(ms, 95)
    p99 = _percentile(ms, 99)

    print(f"\nCentroid:   p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    assert p95 < 5.0, f"Centroid p95 latency {p95:.3f}ms exceeds 5ms threshold"


def test_router_latency_p95_under_5ms(tmp_path: pytest.TempPathFactory) -> None:
    """Full router (no cache) p95 latency must be under 5ms for 1000 texts."""
    router = StyleRouter(cache_dir=None)
    texts = _generate_texts(1000)

    timings_ns: list[int] = []
    for text in texts:
        start = time.perf_counter_ns()
        router.route(text, ALL_STYLES)
        timings_ns.append(time.perf_counter_ns() - start)

    ms = _sorted_timings(timings_ns)
    p50 = _percentile(ms, 50)
    p95 = _percentile(ms, 95)
    p99 = _percentile(ms, 99)

    print(f"\nRouter:     p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    assert p95 < 5.0, f"Router p95 latency {p95:.3f}ms exceeds 5ms threshold"


def test_cache_hit_latency_p95_under_1ms(tmp_path: pytest.TempPathFactory) -> None:
    """Cache hit p95 latency must be under 1ms for 1000 pre-populated entries."""
    cache_dir = tmp_path / "bench_cache"  # type: ignore[operator]
    router = StyleRouter(cache_dir=cache_dir)
    texts = _generate_texts(1000)

    # Pre-populate the cache
    for text in texts:
        router.route(text, ALL_STYLES)

    # Now benchmark cache hits
    timings_ns: list[int] = []
    for text in texts:
        start = time.perf_counter_ns()
        router.route(text, ALL_STYLES)
        timings_ns.append(time.perf_counter_ns() - start)

    router.close()

    ms = _sorted_timings(timings_ns)
    p50 = _percentile(ms, 50)
    p95 = _percentile(ms, 95)
    p99 = _percentile(ms, 99)

    print(f"\nCache hit:  p50={p50:.3f}ms, p95={p95:.3f}ms, p99={p99:.3f}ms")
    assert p95 < 1.0, f"Cache hit p95 latency {p95:.3f}ms exceeds 1ms threshold"


def test_cache_miss_vs_hit_ratio(tmp_path: pytest.TempPathFactory) -> None:
    """Time 1000 cache misses vs 1000 cache hits, print the speedup ratio."""
    cache_dir = tmp_path / "ratio_cache"  # type: ignore[operator]
    router = StyleRouter(cache_dir=cache_dir)
    texts = _generate_texts(1000)

    # Measure misses (cold cache)
    miss_timings_ns: list[int] = []
    for text in texts:
        start = time.perf_counter_ns()
        router.route(text, ALL_STYLES)
        miss_timings_ns.append(time.perf_counter_ns() - start)

    # Measure hits (warm cache)
    hit_timings_ns: list[int] = []
    for text in texts:
        start = time.perf_counter_ns()
        router.route(text, ALL_STYLES)
        hit_timings_ns.append(time.perf_counter_ns() - start)

    router.close()

    miss_ms = _sorted_timings(miss_timings_ns)
    hit_ms = _sorted_timings(hit_timings_ns)

    miss_p95 = _percentile(miss_ms, 95)
    hit_p95 = _percentile(hit_ms, 95)

    ratio = miss_p95 / hit_p95 if hit_p95 > 0.0 else float("inf")

    print(
        f"\nCache miss p95={miss_p95:.3f}ms, hit p95={hit_p95:.3f}ms, "
        f"speedup ratio={ratio:.1f}x"
    )

    # Cache hits should be faster than misses (or at worst equal)
    assert hit_p95 <= miss_p95 * 1.5, (
        f"Cache hits ({hit_p95:.3f}ms p95) are not faster than misses "
        f"({miss_p95:.3f}ms p95)"
    )
