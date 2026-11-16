"""Latency and frame timing profiler."""

from __future__ import annotations

import time
from dataclasses import dataclass

from src.engine.latency_budget import LatencyBudget, LatencySnapshot


@dataclass
class FrameProfile:
    """Per-frame latency breakdown."""

    fps: float = 0.0
    frame_time_ms: float = 0.0
    capture_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0
    render_latency_ms: float = 0.0
    pose_age_ms: float = 0.0
    pipeline_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    quality_percent: int = 100
    compensation_ms: float = 0.0


class Profiler:
    """Rolling FPS and end-to-end latency measurements."""

    def __init__(self, window: int = 120, target_latency_ms: float = 20.0) -> None:
        self._window = window
        self._intervals: list[float] = []
        self._last_frame = time.perf_counter()
        self.profile = FrameProfile()
        self.capture_latency_ms = 0.0
        self.tracking_latency_ms = 0.0
        self.render_latency_ms = 0.0
        self.pose_age_ms = 0.0
        self.quality_percent = 100
        self._budget = LatencyBudget(target_ms=target_latency_ms)

    @property
    def budget(self) -> LatencyBudget:
        return self._budget

    def begin_frame(self) -> float:
        """Mark frame start and return timestamp."""
        return time.perf_counter()

    def end_frame(self, frame_start: float) -> FrameProfile:
        """Update rolling statistics at frame end."""
        now = time.perf_counter()
        interval_ms = (now - self._last_frame) * 1000.0
        self._last_frame = now
        self._intervals.append(interval_ms)
        if len(self._intervals) > self._window:
            self._intervals.pop(0)

        frame_time_ms = (now - frame_start) * 1000.0
        avg_interval = sum(self._intervals) / len(self._intervals)
        fps = 1000.0 / avg_interval if avg_interval > 0 else 0.0

        snapshot = self._budget.update(
            self.capture_latency_ms,
            self.tracking_latency_ms,
            self.render_latency_ms,
            self.pose_age_ms,
        )

        self.profile = FrameProfile(
            fps=fps,
            frame_time_ms=frame_time_ms,
            capture_latency_ms=snapshot.capture_ms,
            tracking_latency_ms=snapshot.tracking_ms,
            render_latency_ms=snapshot.render_ms,
            pose_age_ms=snapshot.pose_age_ms,
            pipeline_latency_ms=snapshot.pipeline_ms,
            total_latency_ms=snapshot.pipeline_ms + snapshot.pose_age_ms,
            quality_percent=self.quality_percent,
            compensation_ms=snapshot.compensation_ms,
        )
        return self.profile
