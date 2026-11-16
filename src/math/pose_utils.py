"""Pose composition utilities for head tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.math.homogeneous import Mat4, Vec3, multiply, translation_matrix
from src.math.quaternion import Quat, from_rotation_matrix, normalize, to_rotation_matrix


@dataclass(frozen=True)
class Pose6DoF:
    """Six-degree-of-freedom head pose."""

    translation_mm: Vec3
    rotation: Quat
    timestamp: float = 0.0

    def as_matrix(self) -> Mat4:
        """Convert pose to a 4x4 transform matrix."""
        rotation_matrix = to_rotation_matrix(self.rotation)
        translation = translation_matrix(
            self.translation_mm[0],
            self.translation_mm[1],
            self.translation_mm[2],
        )
        return multiply(translation, rotation_matrix)


def compose_poses(parent: Pose6DoF, child: Pose6DoF) -> Pose6DoF:
    """Compose two poses: parent @ child."""
    combined = multiply(parent.as_matrix(), child.as_matrix())
    return Pose6DoF(
        translation_mm=combined[:3, 3],
        rotation=from_rotation_matrix(combined),
        timestamp=child.timestamp,
    )


def invert_pose(pose: Pose6DoF) -> Pose6DoF:
    """Invert a 6DoF pose."""
    inverse = np.linalg.inv(pose.as_matrix())
    return Pose6DoF(
        translation_mm=inverse[:3, 3],
        rotation=from_rotation_matrix(inverse),
        timestamp=pose.timestamp,
    )


def interpolate_pose(a: Pose6DoF, b: Pose6DoF, alpha: float) -> Pose6DoF:
    """Linear translation + spherical quaternion interpolation."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    translation = (1.0 - alpha) * a.translation_mm + alpha * b.translation_mm

    q_a = normalize(a.rotation)
    q_b = normalize(b.rotation)
    dot = float(np.dot(q_a, q_b))
    if dot < 0.0:
        q_b = -q_b
        dot = -dot

    if dot > 0.9995:
        rotation = normalize((1.0 - alpha) * q_a + alpha * q_b)
    else:
        theta = np.arccos(np.clip(dot, -1.0, 1.0))
        sin_theta = np.sin(theta)
        w1 = np.sin((1.0 - alpha) * theta) / sin_theta
        w2 = np.sin(alpha * theta) / sin_theta
        rotation = normalize(w1 * q_a + w2 * q_b)

    timestamp = (1.0 - alpha) * a.timestamp + alpha * b.timestamp
    return Pose6DoF(translation_mm=translation, rotation=rotation, timestamp=timestamp)


def identity_pose(timestamp: float = 0.0) -> Pose6DoF:
    """Return identity pose at origin."""
    return Pose6DoF(
        translation_mm=np.zeros(3, dtype=np.float64),
        rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
        timestamp=timestamp,
    )
