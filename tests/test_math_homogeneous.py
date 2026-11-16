"""Unit tests for 4x4 homogeneous matrix utilities."""

from __future__ import annotations

import unittest

import numpy as np

from src.math.homogeneous import (
    identity4,
    invert,
    multiply,
    transform_point,
    translation_matrix,
)


class TestHomogeneous(unittest.TestCase):
    """4x4 matrix operations."""

    def test_identity_transform(self) -> None:
        point = np.array([1.0, 2.0, 3.0])
        result = transform_point(identity4(), point)
        np.testing.assert_allclose(result, point, atol=1e-9)

    def test_translation(self) -> None:
        matrix = translation_matrix(10.0, -5.0, 2.0)
        result = transform_point(matrix, np.zeros(3))
        np.testing.assert_allclose(result, [10.0, -5.0, 2.0], atol=1e-9)

    def test_multiply_with_inverse(self) -> None:
        matrix = translation_matrix(3.0, 4.0, 5.0)
        product = multiply(matrix, invert(matrix))
        np.testing.assert_allclose(product, identity4(), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
