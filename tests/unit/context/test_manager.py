from __future__ import annotations

import pytest

from voxid.context.manager import ContextManager
from voxid.context.types import SegmentHistory


def _make_history(
    text: str = "hello world",
    style: str = "conversational",
    duration_ms: int = 1000,
    final_f0: float = 150.0,
    final_energy: float = 0.1,
    speaking_rate: float = 3.0,
) -> SegmentHistory:
    return SegmentHistory(
        text=text,
        style=style,
        duration_ms=duration_ms,
        final_f0=final_f0,
        final_energy=final_energy,
        speaking_rate=speaking_rate,
    )


class TestContextManager:
    def test_init_default_window(self) -> None:
        mgr = ContextManager()
        assert mgr.window_size == 5
        assert mgr.history == []

    def test_init_custom_window(self) -> None:
        mgr = ContextManager(window_size=3)
        assert mgr.window_size == 3

    def test_init_invalid_window_raises(self) -> None:
        with pytest.raises(ValueError, match="window_size must be >= 1"):
            ContextManager(window_size=0)

    def test_record_and_history(self) -> None:
        mgr = ContextManager(window_size=3)
        entries = [_make_history(text=f"seg {i}") for i in range(3)]
        for e in entries:
            mgr.record(e)
        assert mgr.history == entries

    def test_fifo_overflow(self) -> None:
        mgr = ContextManager(window_size=2)
        for i in range(5):
            mgr.record(_make_history(text=f"seg {i}"))
        assert len(mgr.history) == 2
        assert mgr.history[0].text == "seg 3"
        assert mgr.history[1].text == "seg 4"

    def test_set_total_segments(self) -> None:
        mgr = ContextManager()
        mgr.set_total_segments(10)
        ctx = mgr.build_context(5)
        assert ctx.total_segments == 10

    def test_set_total_segments_negative_raises(self) -> None:
        mgr = ContextManager()
        with pytest.raises(ValueError, match="total must be >= 0"):
            mgr.set_total_segments(-1)

    def test_set_style_sequence(self) -> None:
        mgr = ContextManager()
        styles = ["conversational", "technical", "conversational"]
        mgr.set_style_sequence(styles)
        ctx = mgr.build_context(0)
        assert ctx.style_sequence == styles

    def test_build_context_doc_position(self) -> None:
        mgr = ContextManager()
        mgr.set_total_segments(10)
        ctx = mgr.build_context(0)
        assert ctx.doc_position == 0.0
        ctx = mgr.build_context(5)
        assert ctx.doc_position == 0.5
        ctx = mgr.build_context(10)
        assert ctx.doc_position == 1.0

    def test_build_context_zero_total_segments(self) -> None:
        mgr = ContextManager()
        ctx = mgr.build_context(3)
        assert ctx.doc_position == 0.0

    def test_build_context_includes_history(self) -> None:
        mgr = ContextManager()
        entry = _make_history()
        mgr.record(entry)
        ctx = mgr.build_context(1)
        assert ctx.history == [entry]

    def test_reset_clears_all(self) -> None:
        mgr = ContextManager()
        mgr.set_total_segments(10)
        mgr.set_style_sequence(["a", "b"])
        mgr.record(_make_history())
        mgr.reset()
        assert mgr.history == []
        ctx = mgr.build_context(0)
        assert ctx.total_segments == 0
        assert ctx.style_sequence == []

    def test_history_returns_copy(self) -> None:
        mgr = ContextManager()
        mgr.record(_make_history())
        h1 = mgr.history
        h2 = mgr.history
        assert h1 == h2
        assert h1 is not h2
