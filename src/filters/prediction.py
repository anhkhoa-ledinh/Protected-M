"""Motion prediction to compensate tracking and render latency."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.pose_utils import Pose6DoF


@dataclass(frozen=True)
class PredictionConfig:
    """Settings for head pose extrapolation."""

    enabled: bool = True
    prediction_frames: float = 1.5
    max_prediction_ms: float = 25.0
    min_velocity_mm_s: float = 5.0
    adaptive: bool = True
    measured_latency_ms: float = 0.0

    velocity_smoothing: float = 0.0
    """EMA factor on the velocity estimate. 0 = off, and it ships off."""

    velocity_dt_min_s: float = 0.0
    velocity_dt_max_s: float = 0.0
    """
    Clamp on the interval used to differentiate. 0 disables the clamp, which is
    how this ships -- deliberately. The dt instability these were built for was
    a wrong-timestamp symptom, fixed at source in engine/tracking_worker.py
    (rest p99 3.41 px -> 1.14 px). See config/schema.py::FilterSettings for the
    full rationale and the condition under which to revisit them.
    """


class MotionPredictor:
    """
    Predict head pose ahead using velocity extrapolation.

    Supports tracking-time prediction and render-time extrapolation
    based on measured pipeline latency.
    """

    def __init__(self, config: PredictionConfig | None = None) -> None:
        self._config = config or PredictionConfig()
        self._previous: Pose6DoF | None = None
        self._velocity_mm_s = np.zeros(3, dtype=np.float64)
        self._angular_velocity_rad_s = np.zeros(3, dtype=np.float64)

    @property
    def config(self) -> PredictionConfig:
        return self._config

    @property
    def velocity_mm_s(self) -> np.ndarray:
        return self._velocity_mm_s.copy()

    def set_measured_latency_ms(self, latency_ms: float) -> None:
        """Update adaptive prediction horizon from pipeline measurements."""
        self._config = PredictionConfig(
            enabled=self._config.enabled,
            prediction_frames=self._config.prediction_frames,
            max_prediction_ms=self._config.max_prediction_ms,
            min_velocity_mm_s=self._config.min_velocity_mm_s,
            adaptive=self._config.adaptive,
            measured_latency_ms=max(0.0, latency_ms),
            velocity_smoothing=self._config.velocity_smoothing,
            velocity_dt_min_s=self._config.velocity_dt_min_s,
            velocity_dt_max_s=self._config.velocity_dt_max_s,
        )

    def reset(self) -> None:
        """Clear motion history."""
        self._previous = None
        self._velocity_mm_s.fill(0.0)
        self._angular_velocity_rad_s.fill(0.0)

    def observe(self, pose: Pose6DoF) -> None:
        """Update velocity state without extrapolating."""
        self._update_velocity(pose)

    def update(self, pose: Pose6DoF) -> Pose6DoF:
        """Ingest filtered pose; return latency-compensated prediction."""
        self._update_velocity(pose)
        if not self._config.enabled:
            return pose
        horizon = self._prediction_seconds()
        return self._extrapolate(pose, horizon)

    def extrapolate(self, pose: Pose6DoF, delta_seconds: float) -> Pose6DoF:
        """Extrapolate pose forward by delta_seconds (render-thread use)."""
        if not self._config.enabled or delta_seconds <= 0.0:
            return pose
        capped = min(delta_seconds, self._config.max_prediction_ms * 0.001)
        return self._extrapolate(pose, capped)

    def _update_velocity(self, pose: Pose6DoF) -> None:
        if self._previous is None:
            self._previous = pose
            return

        dt = pose.timestamp - self._previous.timestamp
        if dt <= 1e-6:
            return

        # A dt far below the real frame interval inflates the velocity: the
        # tracking loop runs up to 120 Hz while the camera delivers 60, so a
        # measurement change gets divided by the loop interval rather than the
        # capture interval. Measured at rest, that pushed apparent |v| to a p99
        # of 84 mm/s against a 5 mm/s gate.
        lo = float(self._config.velocity_dt_min_s)
        hi = float(self._config.velocity_dt_max_s)
        if lo > 0.0:
            dt = max(dt, lo)
        if hi > 0.0:
            dt = min(dt, hi)

        velocity = (pose.translation_mm - self._previous.translation_mm) / dt
        angular = _quaternion_to_axis_angle(
            _quaternion_delta(self._previous.rotation, pose.rotation)
        ) / dt

        alpha = float(np.clip(self._config.velocity_smoothing, 0.0, 1.0))
        if alpha > 0.0:
            self._velocity_mm_s = (
                alpha * velocity + (1.0 - alpha) * self._velocity_mm_s
            )
            self._angular_velocity_rad_s = (
                alpha * angular + (1.0 - alpha) * self._angular_velocity_rad_s
            )
        else:
            self._velocity_mm_s = velocity
            self._angular_velocity_rad_s = angular
        self._previous = pose

    def _extrapolate(self, pose: Pose6DoF, delta_seconds: float) -> Pose6DoF:
        if delta_seconds <= 0.0:
            return pose
        speed = float(np.linalg.norm(self._velocity_mm_s))
        if speed < self._config.min_velocity_mm_s:
            return pose

        predicted_translation = pose.translation_mm + self._velocity_mm_s * delta_seconds
        predicted_rotation = _integrate_rotation(
            pose.rotation,
            self._angular_velocity_rad_s,
            delta_seconds,
        )
        return Pose6DoF(
            translation_mm=predicted_translation,
            rotation=predicted_rotation,
            timestamp=pose.timestamp + delta_seconds,
        )

    def _prediction_seconds(self) -> float:
        frame_based = self._config.prediction_frames / 60.0
        max_s = self._config.max_prediction_ms * 0.001
        if self._config.adaptive and self._config.measured_latency_ms > 0.0:
            latency_based = self._config.measured_latency_ms * 0.001
            return min(max(frame_based, latency_based), max_s)
        return min(frame_based, max_s)


def _quaternion_delta(from_q: np.ndarray, to_q: np.ndarray) -> np.ndarray:
    """Return delta quaternion that rotates from_q to to_q."""
    from_w, from_x, from_y, from_z = from_q
    to_w, to_x, to_y, to_z = to_q

    fw, fx, fy, fz = from_w, -from_x, -from_y, -from_z
    dw = fw * to_w - fx * to_x - fy * to_y - fz * to_z
    dx = fw * to_x + fx * to_w + fy * to_z - fz * to_y
    dy = fw * to_y - fx * to_z + fy * to_w + fz * to_x
    dz = fw * to_z + fx * to_y - fy * to_x + fz * to_w

    delta = np.array([dw, dx, dy, dz], dtype=np.float64)
    norm = np.linalg.norm(delta)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return delta / norm


def _quaternion_to_axis_angle(quaternion: np.ndarray) -> np.ndarray:
    """Convert quaternion to axis-angle vector (axis * angle)."""
    w = float(np.clip(quaternion[0], -1.0, 1.0))
    angle = 2.0 * np.arccos(w)
    sin_half = np.sqrt(max(0.0, 1.0 - w * w))
    if sin_half < 1e-8:
        return np.zeros(3, dtype=np.float64)
    axis = quaternion[1:] / sin_half
    return axis * angle


def _integrate_rotation(
    rotation: np.ndarray,
    angular_velocity: np.ndarray,
    dt: float,
) -> np.ndarray:
    """Integrate angular velocity into a new quaternion."""
    angle = float(np.linalg.norm(angular_velocity) * dt)
    if angle < 1e-8:
        return rotation

    axis = angular_velocity / np.linalg.norm(angular_velocity)
    half = angle * 0.5
    sin_half = np.sin(half)
    delta = np.array(
        [np.cos(half), axis[0] * sin_half, axis[1] * sin_half, axis[2] * sin_half],
        dtype=np.float64,
    )

    w1, x1, y1, z1 = rotation
    w2, x2, y2, z2 = delta
    result = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(result)
    return result if norm < 1e-12 else result / norm
