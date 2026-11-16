"""Frame timing utilities for capture and pipeline profiling."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class TimingStats:
    """Rolling statistics for frame intervals and capture latency."""

    frame_count: int = 0
    fps: float = 0.0
    avg_interval_ms: float = 0.0
    min_interval_ms: float = float("inf")
    max_interval_ms: float = 0.0
    last_capture_latency_ms: float = 0.0
    avg_capture_latency_ms: float = 0.0

    def to_dict(self) -> dict[str, float | int]:
        """Serialize stats for logging or GUI display."""
        return {
            "frame_count": self.frame_count,
            "fps": round(self.fps, 2),
            "avg_interval_ms": round(self.avg_interval_ms, 3),
            "min_interval_ms": round(self.min_interval_ms, 3),
            "max_interval_ms": round(self.max_interval_ms, 3),
            "last_capture_latency_ms": round(self.last_capture_latency_ms, 3),
            "avg_capture_latency_ms": round(self.avg_capture_latency_ms, 3),
        }


class FrameTimer:
    """High-resolution monotonic timer for frame pacing and latency measurement."""

    def __init__(self, window_size: int = 120) -> None:
        self._window_size = max(1, window_size)
        self._intervals: list[float] = []
        self._latencies: list[float] = []
        self._last_tick: float | None = None
        self.stats = TimingStats()

    @staticmethod
    def now() -> float:
        """Return monotonic timestamp in seconds."""
        return time.perf_counter()

    def tick(self, capture_latency_ms: float = 0.0) -> TimingStats:
        """Record a frame boundary and update rolling statistics."""
        current = self.now()

        if self._last_tick is not None:
            interval_ms = (current - self._last_tick) * 1000.0
            self._intervals.append(interval_ms)
            if len(self._intervals) > self._window_size:
                self._intervals.pop(0)

            self.stats.frame_count += 1
            self.stats.avg_interval_ms = sum(self._intervals) / len(self._intervals)
            self.stats.min_interval_ms = min(self._intervals)
            self.stats.max_interval_ms = max(self._intervals)
            if self.stats.avg_interval_ms > 0.0:
                self.stats.fps = 1000.0 / self.stats.avg_interval_ms

        self._last_tick = current
        self.stats.last_capture_latency_ms = capture_latency_ms

        if capture_latency_ms > 0.0:
            self._latencies.append(capture_latency_ms)
            if len(self._latencies) > self._window_size:
                self._latencies.pop(0)
            self.stats.avg_capture_latency_ms = sum(self._latencies) / len(
                self._latencies
            )

        return self.stats

    def reset(self) -> None:
        """Clear all accumulated timing data."""
        self._intervals.clear()
        self._latencies.clear()
        self._last_tick = None
        self.stats = TimingStats()

    def elapsed_ms(self, start: float) -> float:
        """Milliseconds elapsed since ``start`` timestamp."""
        return (self.now() - start) * 1000.0
