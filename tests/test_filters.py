"""Unit tests for temporal filters."""

from __future__ import annotations

import unittest

import numpy as np

from src.filters import (
    EmaFilter,
    KalmanFilter1D,
    OneEuroFilter,
    PoseFilterPipeline,
)
from src.math.pose_utils import Pose6DoF


class TestScalarFilters(unittest.TestCase):
    """One Euro, Kalman, and EMA behavior."""

    def test_one_euro_reduces_noise(self) -> None:
        filt = OneEuroFilter()
        values = [0.0, 0.1, -0.1, 0.05, -0.05]
        outputs = [filt.filter(v, i * 0.016) for i, v in enumerate(values)]
        self.assertLess(np.std(outputs), np.std(values))

    def test_kalman_tracks_ramp(self) -> None:
        filt = KalmanFilter1D()
        result = [filt.filter(float(i), i * 0.016) for i in range(10)]
        self.assertGreater(result[-1], result[0])

    def test_ema_smooths(self) -> None:
        filt = EmaFilter()
        first = filt.filter(0.0, 0.0)
        second = filt.filter(10.0, 0.016)
        self.assertGreater(second, first)
        self.assertLess(second, 10.0)


class TestPoseFilterPipeline(unittest.TestCase):
    """Pose pipeline switching and prediction."""

    def test_cycle_smoother(self) -> None:
        pipeline = PoseFilterPipeline()
        self.assertEqual(pipeline.stage_info.smoother, "one_euro")
        next_type = pipeline.cycle_smoother()
        self.assertEqual(next_type, "kalman")

    def test_update_returns_pose(self) -> None:
        pipeline = PoseFilterPipeline()
        pose = Pose6DoF(
            translation_mm=np.array([1.0, 2.0, 600.0]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=0.0,
        )
        result = pipeline.update(pose)
        self.assertEqual(result.translation_mm.shape, (3,))


if __name__ == "__main__":
    unittest.main()
