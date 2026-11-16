"""Unified pose filtering with optional motion prediction."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.schema import FilterSettings
from src.filters.ema import EmaConfig
from src.filters.kalman import KalmanConfig
from src.filters.one_euro import OneEuroConfig
from src.filters.pose_smoother import PoseSmoother
from src.filters.prediction import MotionPredictor, PredictionConfig
from src.math.pose_utils import Pose6DoF


@dataclass(frozen=True)
class FilterStageInfo:
    """Runtime description of active filter chain."""

    smoother: str
    prediction_enabled: bool
    prediction_frames: float
    adaptive_prediction: bool


class PoseFilterPipeline:
    """
    Switchable temporal filter chain for head pose.

    Supported smoothers: one_euro, kalman, ema, none.
    Optional prediction stage runs after smoothing.
    """

    VALID_TYPES = ("one_euro", "kalman", "ema", "none")

    def __init__(self, settings: FilterSettings | None = None) -> None:
        self._settings = settings or FilterSettings()
        self._smoother = PoseSmoother(
            filter_type=self._settings.type,
            one_euro=OneEuroConfig(
                min_cutoff=self._settings.min_cutoff,
                beta=self._settings.beta,
            ),
            kalman=KalmanConfig(
                process_noise=self._settings.process_noise,
                measurement_noise=self._settings.measurement_noise,
            ),
            ema=EmaConfig(alpha=self._settings.ema_alpha),
        )
        self._predictor = self._build_predictor()

    @property
    def stage_info(self) -> FilterStageInfo:
        return FilterStageInfo(
            smoother=self._smoother.filter_type,
            prediction_enabled=self._settings.prediction_enabled,
            prediction_frames=self._settings.prediction_frames,
            adaptive_prediction=self._settings.adaptive_prediction,
        )

    def set_measured_latency_ms(self, latency_ms: float) -> None:
        """Feed pipeline latency into adaptive predictor."""
        if self._settings.adaptive_prediction:
            self._predictor.set_measured_latency_ms(latency_ms)

    def set_smoother(self, filter_type: str) -> None:
        """Switch active smoothing algorithm."""
        if filter_type not in self.VALID_TYPES:
            raise ValueError(f"Invalid filter type: {filter_type}")
        self._settings.type = filter_type
        self._smoother.set_type(filter_type)

    def cycle_smoother(self) -> str:
        """Cycle through smoother types and return the new active type."""
        index = self.VALID_TYPES.index(self._smoother.filter_type)
        next_type = self.VALID_TYPES[(index + 1) % len(self.VALID_TYPES)]
        self.set_smoother(next_type)
        return next_type

    def toggle_prediction(self) -> bool:
        """Enable or disable motion prediction."""
        self._settings.prediction_enabled = not self._settings.prediction_enabled
        self._predictor = self._build_predictor()
        return self._settings.prediction_enabled

    def reset(self) -> None:
        """Reset smoother and predictor state."""
        self._smoother.reset()
        self._predictor.reset()

    def update(self, pose: Pose6DoF, *, new_measurement: bool = True) -> Pose6DoF:
        """
        Run smoothing and optional prediction on a raw pose.

        ``new_measurement=False`` means this pose repeats the previous one
        because no fresh camera frame arrived. The velocity estimate is then
        left alone: differentiating a repeated position over a fresh timestamp
        is what inflated apparent velocity at rest.
        """
        smoothed = self._smoother.smooth(pose)
        if not new_measurement:
            if self._settings.prediction_enabled:
                return self._predictor.extrapolate(smoothed, 0.0)
            return smoothed
        if self._settings.prediction_enabled:
            return self._predictor.update(smoothed)
        self._predictor.observe(smoothed)
        return smoothed

    def extrapolate_for_display(self, pose: Pose6DoF, age_seconds: float) -> Pose6DoF:
        """Apply render-time extrapolation for stale poses."""
        if not self._settings.prediction_enabled:
            return pose
        return self._predictor.extrapolate(pose, age_seconds)

    def _build_predictor(self) -> MotionPredictor:
        return MotionPredictor(
            PredictionConfig(
                enabled=self._settings.prediction_enabled,
                prediction_frames=self._settings.prediction_frames,
                max_prediction_ms=self._settings.max_prediction_ms,
                adaptive=self._settings.adaptive_prediction,
                min_velocity_mm_s=self._settings.min_velocity_mm_s,
                velocity_smoothing=self._settings.velocity_smoothing,
                velocity_dt_min_s=self._settings.velocity_dt_min_s,
                velocity_dt_max_s=self._settings.velocity_dt_max_s,
            )
        )
