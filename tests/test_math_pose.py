"""Unit tests for pose estimation utilities."""

from __future__ import annotations

import unittest

import numpy as np

from src.math.pose_utils import Pose6DoF, compose_poses, identity_pose, interpolate_pose, invert_pose
from src.math.quaternion import from_axis_angle


class TestPoseUtils(unittest.TestCase):
    """6DoF pose composition and interpolation."""

    def test_identity_compose(self) -> None:
        pose = identity_pose()
        composed = compose_poses(pose, pose)
        np.testing.assert_allclose(
            composed.translation_mm,
            pose.translation_mm,
            atol=1e-9,
        )

    def test_invert_pose(self) -> None:
        pose = Pose6DoF(
            translation_mm=np.array([10.0, 20.0, 30.0]),
            rotation=from_axis_angle(np.array([0.0, 1.0, 0.0]), 0.5),
            timestamp=1.0,
        )
        restored = compose_poses(pose, invert_pose(pose))
        np.testing.assert_allclose(restored.translation_mm, 0.0, atol=1e-6)

    def test_interpolate_midpoint(self) -> None:
        a = Pose6DoF(
            translation_mm=np.array([0.0, 0.0, 0.0]),
            rotation=from_axis_angle(np.array([0.0, 0.0, 1.0]), 0.0),
            timestamp=0.0,
        )
        b = Pose6DoF(
            translation_mm=np.array([100.0, 0.0, 0.0]),
            rotation=from_axis_angle(np.array([0.0, 0.0, 1.0]), np.pi * 0.5),
            timestamp=1.0,
        )
        mid = interpolate_pose(a, b, 0.5)
        np.testing.assert_allclose(mid.translation_mm, [50.0, 0.0, 0.0], atol=1e-6)
        self.assertAlmostEqual(mid.timestamp, 0.5)


if __name__ == "__main__":
    unittest.main()
