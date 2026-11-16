"""Unit tests for landmark utilities."""

from __future__ import annotations

import unittest

import numpy as np

from src.tracking.landmarks import PNP_LANDMARKS, normalized_to_pixels
from src.tracking.iris import iris_midpoint, inter_pupillary_distance_px
from src.tracking.landmarks import LandmarkSet


class TestLandmarks(unittest.TestCase):
    """Landmark conversion helpers."""

    def test_normalized_to_pixels(self) -> None:
        pts = np.array([[0.5, 0.5], [0.0, 1.0]], dtype=np.float64)
        result = normalized_to_pixels(pts, 640, 480)
        np.testing.assert_allclose(result[0], [320.0, 240.0])
        np.testing.assert_allclose(result[1], [0.0, 480.0])

    def test_iris_midpoint(self) -> None:
        lm = LandmarkSet(
            image_points=np.zeros((6, 2)),
            left_iris=np.array([100.0, 200.0]),
            right_iris=np.array([200.0, 200.0]),
        )
        mid = iris_midpoint(lm)
        np.testing.assert_allclose(mid, [150.0, 200.0])

    def test_pnp_indices_count(self) -> None:
        self.assertEqual(len(PNP_LANDMARKS), 6)


if __name__ == "__main__":
    unittest.main()
