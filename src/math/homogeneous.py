"""4x4 homogeneous matrix utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

Mat4 = npt.NDArray[np.float64]
Vec3 = npt.NDArray[np.float64]
Vec4 = npt.NDArray[np.float64]


def identity4() -> Mat4:
    """Return a 4x4 identity matrix."""
    return np.eye(4, dtype=np.float64)


def translation_matrix(x: float, y: float, z: float) -> Mat4:
    """Build a 4x4 translation matrix."""
    matrix = identity4()
    matrix[0, 3] = x
    matrix[1, 3] = y
    matrix[2, 3] = z
    return matrix


def scale_matrix(sx: float, sy: float, sz: float) -> Mat4:
    """Build a 4x4 scale matrix."""
    matrix = identity4()
    matrix[0, 0] = sx
    matrix[1, 1] = sy
    matrix[2, 2] = sz
    return matrix


def multiply(a: Mat4, b: Mat4) -> Mat4:
    """Multiply two 4x4 matrices (a @ b)."""
    return a @ b


def transform_point(matrix: Mat4, point: Vec3) -> Vec3:
    """Transform a 3D point by a 4x4 matrix."""
    homogeneous = np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)
    result = matrix @ homogeneous
    w = result[3]
    if abs(w) < 1e-12:
        return result[:3]
    return result[:3] / w


def transform_direction(matrix: Mat4, direction: Vec3) -> Vec3:
    """Transform a direction vector, ignoring translation."""
    homogeneous = np.array(
        [direction[0], direction[1], direction[2], 0.0],
        dtype=np.float64,
    )
    result = matrix @ homogeneous
    return result[:3]


def invert(matrix: Mat4) -> Mat4:
    """Invert a 4x4 matrix."""
    return np.linalg.inv(matrix)


def to_column_major_flat(matrix: Mat4) -> npt.NDArray[np.float32]:
    """Flatten matrix in column-major order for OpenGL."""
    return np.asarray(matrix, dtype=np.float32).T.reshape(-1, order="F")
