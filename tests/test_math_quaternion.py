"""Unit tests for quaternion math."""

from __future__ import annotations

import math
import unittest

import numpy as np

from src.math.quaternion import (
    from_axis_angle,
    from_rotation_matrix,
    identity_quat,
    multiply,
    normalize,
    rotate_vector,
    to_rotation_matrix,
)


class TestQuaternion(unittest.TestCase):
    """Quaternion conversions and rotation."""

    def test_identity_rotation(self) -> None:
        vector = np.array([1.0, 0.0, 0.0])
        rotated = rotate_vector(identity_quat(), vector)
        np.testing.assert_allclose(rotated, vector, atol=1e-9)

    def test_axis_angle_z_90(self) -> None:
        quat = from_axis_angle(np.array([0.0, 0.0, 1.0]), math.pi * 0.5)
        rotated = rotate_vector(quat, np.array([1.0, 0.0, 0.0]))
        np.testing.assert_allclose(rotated, [0.0, 1.0, 0.0], atol=1e-6)

    def test_matrix_round_trip(self) -> None:
        quat = normalize(from_axis_angle(np.array([0.2, 0.7, 0.1]), 0.8))
        matrix = to_rotation_matrix(quat)
        recovered = from_rotation_matrix(matrix)
        recovered_matrix = to_rotation_matrix(recovered)
        np.testing.assert_allclose(recovered_matrix, matrix, atol=1e-6)

    def test_multiply_with_conjugate(self) -> None:
        quat = normalize(from_axis_angle(np.array([1.0, 2.0, 3.0]), 1.1))
        product = multiply(quat, np.array([quat[0], -quat[1], -quat[2], -quat[3]]))
        np.testing.assert_allclose(product[0], 1.0, atol=1e-6)
        np.testing.assert_allclose(product[1:], 0.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
