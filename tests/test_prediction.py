"""Unit tests for motion prediction filter."""

from __future__ import annotations

import unittest

import numpy as np

from src.filters.prediction import MotionPredictor, PredictionConfig
from src.math.pose_utils import Pose6DoF


class TestMotionPrediction(unittest.TestCase):
    """Head pose extrapolation."""

    def test_first_sample_returns_input(self) -> None:
        predictor = MotionPredictor(PredictionConfig(enabled=True, prediction_frames=1.0))
        pose = Pose6DoF(
            translation_mm=np.zeros(3),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        result = predictor.update(pose)
        np.testing.assert_allclose(result.translation_mm, pose.translation_mm)

    def test_constant_velocity_extrapolation(self) -> None:
        predictor = MotionPredictor(
            PredictionConfig(enabled=True, prediction_frames=1.0, max_prediction_ms=100.0)
        )
        p0 = Pose6DoF(
            translation_mm=np.array([0.0, 0.0, 600.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        p1 = Pose6DoF(
            translation_mm=np.array([10.0, 0.0, 600.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.016,
        )
        predictor.update(p0)
        predicted = predictor.update(p1)
        self.assertGreater(predicted.translation_mm[0], p1.translation_mm[0])

    def test_disabled_prediction(self) -> None:
        predictor = MotionPredictor(PredictionConfig(enabled=False))
        p0 = Pose6DoF(
            translation_mm=np.zeros(3),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        p1 = Pose6DoF(
            translation_mm=np.array([50.0, 0.0, 0.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.016,
        )
        predictor.update(p0)
        result = predictor.update(p1)
        np.testing.assert_allclose(result.translation_mm, p1.translation_mm)


if __name__ == "__main__":
    unittest.main()


class VelocityConditioningTest(unittest.TestCase):
    """dt clamp, EMA smoothing, and the repeated-pose guard."""

    @staticmethod
    def _pose(z: float, t: float) -> Pose6DoF:
        return Pose6DoF(
            translation_mm=np.array([0.0, 0.0, z]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=t,
        )

    def test_defaults_reproduce_original_behaviour(self) -> None:
        """Zero clamps and zero smoothing must leave the old maths untouched."""
        p = MotionPredictor(PredictionConfig())
        p.observe(self._pose(600.0, 0.0))
        p.observe(self._pose(601.0, 0.010))
        self.assertAlmostEqual(float(p.velocity_mm_s[2]), 100.0, places=6)

    def test_dt_clamp_stops_a_tiny_interval_inflating_velocity(self) -> None:
        """
        A 0.5 ms gap between poses is impossible at camera rate; without the
        clamp it turns a 1 mm step into 2000 mm/s.
        """
        unclamped = MotionPredictor(PredictionConfig())
        unclamped.observe(self._pose(600.0, 0.0))
        unclamped.observe(self._pose(601.0, 0.0005))
        self.assertGreater(float(unclamped.velocity_mm_s[2]), 1500.0)

        clamped = MotionPredictor(
            PredictionConfig(velocity_dt_min_s=0.005, velocity_dt_max_s=0.100)
        )
        clamped.observe(self._pose(600.0, 0.0))
        clamped.observe(self._pose(601.0, 0.0005))
        self.assertAlmostEqual(float(clamped.velocity_mm_s[2]), 200.0, places=6)

    def test_dt_clamp_upper_bound(self) -> None:
        p = MotionPredictor(
            PredictionConfig(velocity_dt_min_s=0.005, velocity_dt_max_s=0.100)
        )
        p.observe(self._pose(600.0, 0.0))
        p.observe(self._pose(610.0, 5.0))       # a 5 s stall
        self.assertAlmostEqual(float(p.velocity_mm_s[2]), 100.0, places=6)

    def test_smoothing_damps_an_isolated_velocity_spike(self) -> None:
        cfg = PredictionConfig(velocity_smoothing=0.3)
        p = MotionPredictor(cfg)
        p.observe(self._pose(600.0, 0.000))
        p.observe(self._pose(600.0, 0.016))     # at rest
        p.observe(self._pose(605.0, 0.032))     # one noisy jump
        smoothed = abs(float(p.velocity_mm_s[2]))

        raw = MotionPredictor(PredictionConfig())
        raw.observe(self._pose(600.0, 0.000))
        raw.observe(self._pose(600.0, 0.016))
        raw.observe(self._pose(605.0, 0.032))
        self.assertLess(smoothed, abs(float(raw.velocity_mm_s[2])))


class RepeatedPoseTest(unittest.TestCase):
    """
    The tracking loop runs faster than the camera, so it re-publishes the
    previous pose. Differentiating that over a fresh timestamp is what pushed
    apparent velocity at rest to a p99 of 84 mm/s against a 5 mm/s gate.
    """

    @staticmethod
    def _pose(z: float, t: float) -> Pose6DoF:
        return Pose6DoF(
            translation_mm=np.array([0.0, 0.0, z]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=t,
        )

    def test_repeated_pose_leaves_velocity_untouched(self) -> None:
        from src.config.schema import FilterSettings
        from src.filters.pipeline import PoseFilterPipeline

        pipe = PoseFilterPipeline(FilterSettings(type="none"))
        pipe.update(self._pose(600.0, 0.000), new_measurement=True)
        pipe.update(self._pose(610.0, 0.016), new_measurement=True)
        moving = pipe._predictor.velocity_mm_s.copy()

        # Same position re-published with a fresh timestamp.
        pipe.update(self._pose(610.0, 0.024), new_measurement=False)
        np.testing.assert_allclose(pipe._predictor.velocity_mm_s, moving)
