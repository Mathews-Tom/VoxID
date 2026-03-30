from __future__ import annotations

from collections import deque

from .types import GenerationContext, SegmentHistory


class ContextManager:
    """Rolling-window context manager for long-form generation.

    Maintains a FIFO buffer of recent segment histories and builds
    GenerationContext objects consumed by ContextConditioner.
    """

    def __init__(self, window_size: int = 5) -> None:
        if window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {window_size}")
        self._window_size = window_size
        self._history: deque[SegmentHistory] = deque(maxlen=window_size)
        self._style_sequence: list[str] = []
        self._total_segments: int = 0

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def history(self) -> list[SegmentHistory]:
        return list(self._history)

    def set_total_segments(self, total: int) -> None:
        """Set the total number of segments for document position tracking."""
        if total < 0:
            raise ValueError(f"total must be >= 0, got {total}")
        self._total_segments = total

    def set_style_sequence(self, styles: list[str]) -> None:
        """Set the full style sequence from the generation plan."""
        self._style_sequence = list(styles)

    def record(self, entry: SegmentHistory) -> None:
        """Record a generated segment's prosodic features."""
        self._history.append(entry)

    def build_context(self, segment_index: int) -> GenerationContext:
        """Build generation context for the segment at the given index."""
        if self._total_segments > 0:
            doc_position = segment_index / self._total_segments
        else:
            doc_position = 0.0

        return GenerationContext(
            history=list(self._history),
            doc_position=doc_position,
            total_segments=self._total_segments,
            style_sequence=self._style_sequence,
        )

    def reset(self) -> None:
        """Clear all tracked state."""
        self._history.clear()
        self._style_sequence = []
        self._total_segments = 0
