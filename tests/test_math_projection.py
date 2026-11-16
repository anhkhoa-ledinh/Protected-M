"""Unit tests for Kooima off-axis projection."""

from __future__ import annotations

import unittest

import numpy as np

from src.math.projection import compute_off_axis_projection, off_axis_projection


class TestProjection(unittest.TestCase):
    """Off-axis projection behavior."""

    def test_kooima_projection_centered_eye(self) -> None:
        pa = np.array([-0.2655, -0.1495, 0.0], dtype=np.float64)
        pb = np.array([0.2655, -0.1495, 0.0], dtype=np.float64)
        pc = np.array([-0.2655, 0.1495, 0.0], dtype=np.float64)
        pe = np.array([0.0, 0.0, 0.6], dtype=np.float64)
        projection, _ = compute_off_axis_projection(pa, pb, pc, pe, 0.01, 8.0)
        self.assertEqual(projection.shape, (4, 4))
        self.assertAlmostEqual(projection[3, 2], -1.0)

    def test_off_axis_is_asymmetric_when_eye_shifted(self) -> None:
        pa = np.array([-0.2655, -0.1495, 0.0], dtype=np.float64)
        pb = np.array([0.2655, -0.1495, 0.0], dtype=np.float64)
        pc = np.array([-0.2655, 0.1495, 0.0], dtype=np.float64)
        centered, _ = compute_off_axis_projection(
            pa,
            pb,
            pc,
            np.array([0.0, 0.0, 0.6], dtype=np.float64),
            0.01,
            8.0,
        )
        shifted, _ = compute_off_axis_projection(
            pa,
            pb,
            pc,
            np.array([0.08, 0.0, 0.6], dtype=np.float64),
            0.01,
            8.0,
        )
        self.assertFalse(np.allclose(centered, shifted))

    def test_off_axis_matrix_bounds(self) -> None:
        matrix = off_axis_projection(-0.2, 0.12, -0.09, 0.11, 0.01, 5.0)
        self.assertEqual(matrix.shape, (4, 4))
        self.assertLess(matrix[0, 2], 0.0)

    def test_depth_offset_changes_projection(self) -> None:
        pa = np.array([-0.2655, -0.1495, 0.0], dtype=np.float64)
        pb = np.array([0.2655, -0.1495, 0.0], dtype=np.float64)
        pc = np.array([-0.2655, 0.1495, 0.0], dtype=np.float64)
        pe = np.array([0.0, 0.0, 0.6], dtype=np.float64)
        base, _ = compute_off_axis_projection(pa, pb, pc, pe, 0.01, 8.0)
        closer, _ = compute_off_axis_projection(
            pa, pb, pc, pe, 0.01, 8.0, depth_offset_m=-0.05
        )
        self.assertFalse(np.allclose(base, closer))


if __name__ == "__main__":
    unittest.main()
