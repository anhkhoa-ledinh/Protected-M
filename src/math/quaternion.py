"""Quaternion utilities for head pose rotation."""

from __future__ import annotations

import math

import numpy as np
import numpy.typing as npt

from src.math.homogeneous import Mat4, Vec3

Quat = npt.NDArray[np.float64]


def identity_quat() -> Quat:
    """Return identity quaternion [w, x, y, z]."""
    return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


def normalize(quaternion: Quat) -> Quat:
    """Normalize quaternion to unit length."""
    norm = np.linalg.norm(quaternion)
    if norm < 1e-12:
        return identity_quat()
    return quaternion / norm


def multiply(a: Quat, b: Quat) -> Quat:
    """Hamilton product of two quaternions [w, x, y, z]."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dtype=np.float64,
    )


def conjugate(quaternion: Quat) -> Quat:
    """Return quaternion conjugate."""
    w, x, y, z = quaternion
    return np.array([w, -x, -y, -z], dtype=np.float64)


def from_axis_angle(axis: Vec3, angle_rad: float) -> Quat:
    """Create quaternion from axis-angle representation."""
    axis_norm = np.linalg.norm(axis)
    if axis_norm < 1e-12:
        return identity_quat()

    unit_axis = axis / axis_norm
    half = angle_rad * 0.5
    sin_half = math.sin(half)
    return np.array(
        [math.cos(half), unit_axis[0] * sin_half, unit_axis[1] * sin_half, unit_axis[2] * sin_half],
        dtype=np.float64,
    )


def to_rotation_matrix(quaternion: Quat) -> Mat4:
    """Convert unit quaternion to a 4x4 rotation matrix."""
    q = normalize(quaternion)
    w, x, y, z = q

    matrix = np.eye(4, dtype=np.float64)
    matrix[0, 0] = 1.0 - 2.0 * (y * y + z * z)
    matrix[0, 1] = 2.0 * (x * y - z * w)
    matrix[0, 2] = 2.0 * (x * z + y * w)
    matrix[1, 0] = 2.0 * (x * y + z * w)
    matrix[1, 1] = 1.0 - 2.0 * (x * x + z * z)
    matrix[1, 2] = 2.0 * (y * z - x * w)
    matrix[2, 0] = 2.0 * (x * z - y * w)
    matrix[2, 1] = 2.0 * (y * z + x * w)
    matrix[2, 2] = 1.0 - 2.0 * (x * x + y * y)
    return matrix


def from_rotation_matrix(matrix: Mat4) -> Quat:
    """Convert 3x3 rotation block of a matrix to quaternion."""
    rot = matrix[:3, :3]
    trace = float(np.trace(rot))

    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    return normalize(np.array([w, x, y, z], dtype=np.float64))


def rotate_vector(quaternion: Quat, vector: Vec3) -> Vec3:
    """Rotate a 3D vector by quaternion."""
    q = normalize(quaternion)
    vector_quat = np.array([0.0, vector[0], vector[1], vector[2]], dtype=np.float64)
    rotated = multiply(multiply(q, vector_quat), conjugate(q))
    return rotated[1:]
