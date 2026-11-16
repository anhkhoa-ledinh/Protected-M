"""Runtime latency measurement and prediction budget."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LatencySnapshot:
    """Smoothed latency breakdown used for prediction and profiling."""

    capture_ms: float
    tracking_ms: float
    render_ms: float
    pose_age_ms: float
    pipeline_ms: float
    compensation_ms: float


class LatencyBudget:
    """
    Track rolling latency across pipeline stages.

    Feeds adaptive motion prediction so render compensates for
    measured capture + tracking + render delay.
    """

    def __init__(self, alpha: float = 0.15, target_ms: float = 20.0) -> None:
        self._alpha = alpha
        self._target_ms = target_ms
        self._capture_ms = 0.0
        self._tracking_ms = 0.0
        self._render_ms = 0.0
        self._pose_age_ms = 0.0

    @property
    def target_ms(self) -> float:
        return self._target_ms

    def update(
        self,
        capture_ms: float,
        tracking_ms: float,
        render_ms: float,
        pose_age_ms: float,
    ) -> LatencySnapshot:
        """Ingest fresh measurements with exponential smoothing."""
        self._capture_ms = self._smooth(self._capture_ms, capture_ms)
        self._tracking_ms = self._smooth(self._tracking_ms, tracking_ms)
        self._render_ms = self._smooth(self._render_ms, render_ms)
        self._pose_age_ms = self._smooth(self._pose_age_ms, pose_age_ms)

        pipeline = self._capture_ms + self._tracking_ms + self._render_ms
        compensation = min(pipeline + self._pose_age_ms * 0.5, self._target_ms * 2.0)

        return LatencySnapshot(
            capture_ms=self._capture_ms,
            tracking_ms=self._tracking_ms,
            render_ms=self._render_ms,
            pose_age_ms=self._pose_age_ms,
            pipeline_ms=pipeline,
            compensation_ms=compensation,
        )

    def _smooth(self, previous: float, sample: float) -> float:
        if previous <= 0.0:
            return sample
        return previous * (1.0 - self._alpha) + sample * self._alpha
