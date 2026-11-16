"""Unit tests for latency optimization."""

from __future__ import annotations

import time
import unittest

import numpy as np

from src.engine.latency_budget import LatencyBudget
from src.engine.pose_bus import PoseBus
from src.filters.prediction import MotionPredictor, PredictionConfig
from src.math.pose_utils import Pose6DoF


class TestLatencyBudget(unittest.TestCase):
    """Pipeline latency smoothing."""

    def test_compensation_grows_with_latency(self) -> None:
        budget = LatencyBudget(alpha=1.0, target_ms=20.0)
        low = budget.update(2.0, 5.0, 3.0, 1.0)
        high = budget.update(8.0, 15.0, 10.0, 12.0)
        self.assertLess(low.compensation_ms, high.compensation_ms)


class TestAdaptivePrediction(unittest.TestCase):
    """Latency-aware motion prediction."""

    def test_adaptive_horizon_uses_measured_latency(self) -> None:
        predictor = MotionPredictor(
            PredictionConfig(
                enabled=True,
                prediction_frames=1.0,
                max_prediction_ms=30.0,
                adaptive=True,
            )
        )
        predictor.set_measured_latency_ms(20.0)
        p0 = Pose6DoF(
            translation_mm=np.zeros(3),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        p1 = Pose6DoF(
            translation_mm=np.array([100.0, 0.0, 600.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.016,
        )
        predictor.update(p0)
        predicted = predictor.update(p1)
        self.assertGreater(predicted.translation_mm[0], p1.translation_mm[0])

    def test_extrapolate_for_render(self) -> None:
        predictor = MotionPredictor(PredictionConfig(enabled=True, min_velocity_mm_s=0.0))
        p0 = Pose6DoF(
            translation_mm=np.zeros(3),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        p1 = Pose6DoF(
            translation_mm=np.array([50.0, 0.0, 600.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.016,
        )
        predictor.update(p0)
        predictor.update(p1)
        extrapolated = predictor.extrapolate(p1, 0.010)
        self.assertGreaterEqual(extrapolated.translation_mm[0], p1.translation_mm[0])


class TestPoseBusAge(unittest.TestCase):
    """Pose staleness measurement."""

    def test_pose_age_increases(self) -> None:
        bus = PoseBus()
        time.sleep(0.01)
        self.assertGreater(bus.latest().age_ms(), 5.0)


if __name__ == "__main__":
    unittest.main()
