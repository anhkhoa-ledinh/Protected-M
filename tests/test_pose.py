"""Unit tests for solvePnP head pose estimation."""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from src.math.pose_utils import Pose6DoF
from src.math.quaternion import from_rotation_matrix, to_rotation_matrix
from src.pose.calibration import CameraIntrinsics
from src.pose.eye_position import EyePositionEstimator
from src.pose.head_pose import (
    HeadPoseEstimator,
    HeadPoseResult,
    R_CV_TO_HCPD,
    _euler_from_rotation,
    _opencv_rot_to_hcpd,
    _opencv_to_hcpd_translation,
)
from src.tracking.landmarks import FACE_MODEL_MM, LandmarkSet


def _head_result_at(translation_mm: np.ndarray) -> HeadPoseResult:
    """A detected HeadPoseResult carrying ``translation_mm`` and no rotation."""
    translation_mm = np.asarray(translation_mm, dtype=np.float64)
    return HeadPoseResult(
        detected=True,
        pose=Pose6DoF(
            translation_mm=translation_mm,
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
        ),
        pitch_deg=0.0,
        yaw_deg=0.0,
        roll_deg=0.0,
        translation_mm=translation_mm,
        rotation_matrix=np.eye(3),
    )


def _iris_at(u: float, v: float) -> LandmarkSet:
    """Both iris centres on one pixel, so the midpoint is exactly (u, v)."""
    return LandmarkSet(
        image_points=np.zeros((6, 2)),
        left_iris=np.array([float(u), float(v)]),
        right_iris=np.array([float(u), float(v)]),
    )


class TestHeadPose(unittest.TestCase):
    """Head pose estimation from synthetic landmarks."""

    def test_estimate_with_centered_face(self) -> None:
        intrinsics = CameraIntrinsics.from_image_size(640, 480)
        estimator = HeadPoseEstimator(
            intrinsics=intrinsics,
            reference_distance_mm=600.0,
        )
        cx, cy = intrinsics.cx, intrinsics.cy
        image_points = np.array(
            [
                [cx, cy],
                [cx, cy + 80],
                [cx - 60, cy - 30],
                [cx + 60, cy - 30],
                [cx - 40, cy + 40],
                [cx + 40, cy + 40],
            ],
            dtype=np.float64,
        )
        landmarks = LandmarkSet(
            image_points=image_points,
            left_iris=np.array([cx - 30, cy - 30]),
            right_iris=np.array([cx + 30, cy - 30]),
        )
        result = estimator.estimate(landmarks, timestamp=1.0)
        self.assertTrue(result.detected)
        self.assertAlmostEqual(result.translation_mm[2], 600.0, delta=150.0)

    def test_eye_z_varies_with_raw_depth_after_baseline(self) -> None:
        """Per-frame reference/depth scaling must not cancel Z variation."""
        intrinsics = CameraIntrinsics.from_image_size(640, 480)
        estimator = HeadPoseEstimator(
            intrinsics=intrinsics,
            reference_distance_mm=600.0,
            webcam_offset_mm=np.array([0.0, 103.0, 5.0]),
        )
        estimator._reference_depth_raw = 2650.0

        near = _opencv_to_hcpd_translation(
            np.array([0.0, 0.0, 600.0 * (2500.0 / 2650.0)]),
            estimator._webcam_offset,
        )
        far = _opencv_to_hcpd_translation(
            np.array([0.0, 0.0, 600.0 * (2800.0 / 2650.0)]),
            estimator._webcam_offset,
        )

        self.assertNotAlmostEqual(near[2], far[2], places=1)
        self.assertLess(near[2], 605.0)
        self.assertGreater(far[2], 605.0)

    def test_monocular_scale_uses_fixed_baseline_not_per_frame_depth(self) -> None:
        estimator = HeadPoseEstimator(
            intrinsics=CameraIntrinsics.from_image_size(640, 480),
            reference_distance_mm=600.0,
        )
        estimator._reference_depth_raw = 2500.0
        self.assertAlmostEqual(estimator._monocular_scale(), 600.0 / 2500.0)

    def test_face_model_has_six_points(self) -> None:
        self.assertEqual(FACE_MODEL_MM.shape, (6, 3))


if __name__ == "__main__":
    unittest.main()


class BasisChangeInvariantTest(unittest.TestCase):
    """
    Guards on R_CV_TO_HCPD itself, not on one worked example.

    The bug this replaces was a reflection, diag(1,-1,1), sitting where a rigid
    motion belongs. A determinant check catches that whole class: anyone who
    writes another axis flip trips this without needing to understand the optics.
    """

    def test_determinant_is_plus_one(self) -> None:
        """det == +1 rejects every reflection."""
        self.assertAlmostEqual(float(np.linalg.det(R_CV_TO_HCPD)), 1.0, places=12)

    def test_length_is_preserved(self) -> None:
        """
        det == +1 alone still admits a wrong scale, e.g. diag(-2,-2,0.25).
        Redundant for a diag constant; load-bearing the day M becomes a general
        matrix for an off-axis camera mount.
        """
        for v in ([1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
                  [123.0, -45.0, 600.0]):
            arr = np.asarray(v, dtype=np.float64)
            self.assertAlmostEqual(
                float(np.linalg.norm(R_CV_TO_HCPD @ arr)),
                float(np.linalg.norm(arr)),
                places=12,
            )

    def test_is_orthogonal(self) -> None:
        np.testing.assert_allclose(
            R_CV_TO_HCPD @ R_CV_TO_HCPD.T, np.eye(3), atol=1e-12
        )


class EulerMatchesQuaternionTest(unittest.TestCase):
    """
    The Euler triple and the quaternion must describe the SAME rotation.

    They did not: `estimate` built the quaternion from the HCPD matrix and the
    angles from the raw OpenCV one. Asserting "pitch has the right sign" would
    only pin one worked example, so this rebuilds the matrix from the angles and
    compares it to the matrix the quaternion decodes to — that catches any future
    basis, order or degree/radian slip in either half.
    """

    @staticmethod
    def _rebuild(pitch_deg: float, yaw_deg: float, roll_deg: float) -> np.ndarray:
        """Intrinsic Z-Y-X: R = Rz(roll) @ Ry(yaw) @ Rx(pitch)."""
        p, y, r = (np.radians(a) for a in (pitch_deg, yaw_deg, roll_deg))
        rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
        ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
        rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
        return rz @ ry @ rx

    def test_angles_reconstruct_the_quaternion(self) -> None:
        rng = np.random.default_rng(7)
        rotations = [np.eye(3)]
        for _ in range(24):
            rotations.append(cv2.Rodrigues(rng.normal(size=3) * 0.6)[0])

        for rotation_cv in rotations:
            hcpd = _opencv_rot_to_hcpd(rotation_cv)
            pitch, yaw, roll = _euler_from_rotation(hcpd[:3, :3])
            quaternion = from_rotation_matrix(hcpd)
            np.testing.assert_allclose(
                self._rebuild(pitch, yaw, roll),
                to_rotation_matrix(quaternion)[:3, :3],
                atol=1e-9,
            )

    def test_angles_are_degrees_not_radians(self) -> None:
        """A 30 deg yaw must read ~30, not ~0.52."""
        yaw_rad = np.radians(30.0)
        rotation_cv = np.array(
            [
                [np.cos(yaw_rad), 0.0, np.sin(yaw_rad)],
                [0.0, 1.0, 0.0],
                [-np.sin(yaw_rad), 0.0, np.cos(yaw_rad)],
            ]
        )
        _, yaw, _ = _euler_from_rotation(_opencv_rot_to_hcpd(rotation_cv)[:3, :3])
        self.assertAlmostEqual(abs(yaw), 30.0, places=9)


class ParallaxSignTest(unittest.TestCase):
    """
    The viewer's right must map to world +X.

    Confirmed against physical tape marks: the tracking benchmark's ruler
    results record that predictions needed -t_x to line up, which is an
    end-to-end measurement and therefore also rules out driver-side mirroring.
    """

    def test_positive_image_x_gives_negative_world_x(self) -> None:
        zero = np.zeros(3)
        world = _opencv_to_hcpd_translation(np.array([100.0, 0.0, 600.0]), zero)
        self.assertLess(world[0], 0.0)

    def test_depth_stays_positive(self) -> None:
        zero = np.zeros(3)
        world = _opencv_to_hcpd_translation(np.array([0.0, 0.0, 600.0]), zero)
        self.assertAlmostEqual(world[2], 600.0)

    def test_image_down_gives_world_up(self) -> None:
        zero = np.zeros(3)
        world = _opencv_to_hcpd_translation(np.array([0.0, 100.0, 600.0]), zero)
        self.assertLess(world[1], 0.0)

    def test_iris_right_of_centre_maps_to_negative_world_x(self) -> None:
        """
        The refinement must pull the same way the base translation does.

        It used to disagree in X, so it amplified the parallax inversion instead
        of fighting it. Expressed on the iris ESTIMATE rather than on the blended
        output: the output is bracketed by construction now, so asserting a
        direction on it would pass for the wrong reason.
        """
        intr = CameraIntrinsics.from_image_size(1280, 720)
        est = EyePositionEstimator(fx=intr.fx, cx=intr.cx, cy=intr.cy)

        base = _opencv_to_hcpd_translation(np.array([100.0, 0.0, 600.0]), np.zeros(3))
        self.assertLess(float(base[0]), 0.0)  # tx > 0 -> world x < 0
        head = _head_result_at(base)
        # Iris right of image centre must land on the same side as base.
        lm = _iris_at(intr.cx + 100.0, intr.cy)
        refined = est.estimate(head, lm, timestamp=0.0)
        iris_x = -100.0 * float(base[2]) / intr.fx
        self.assertLess(iris_x, 0.0)
        # And the result sits between the two, not beyond either.
        self.assertGreaterEqual(float(refined.translation_mm[0]), min(base[0], iris_x))
        self.assertLessEqual(float(refined.translation_mm[0]), max(base[0], iris_x))


class IrisBlendTest(unittest.TestCase):
    """
    The iris refinement is an INTERPOLATION, not an addition (audit #1, #2).

    ``eye += 0.12 * iris_offset`` added 12% of one absolute position to 100% of
    another, so it was a 1.12x gain on X (+12.1 mm at a true 100 mm offset) plus
    a constant ``-0.12 * camera_offset_y`` = -12.4 mm on Y, because the iris term
    was lens-centred while the translation was screen-centred.

    The test that used to live here asserted ``refined_x < base_x`` -- the
    overshoot signature itself -- which is why the defect survived. These pin the
    two properties an interpolation has and an addition cannot.
    """

    WEBCAM_OFFSET_MM = np.array([0.0, 103.0, 5.0])

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics.from_image_size(1280, 720)
        self.estimator = EyePositionEstimator(
            fx=self.intrinsics.fx,
            cx=self.intrinsics.cx,
            cy=self.intrinsics.cy,
            fy=self.intrinsics.fy,
            webcam_offset_mm=self.WEBCAM_OFFSET_MM,
        )

    def _project_eye(self, eye_world_mm) -> tuple[np.ndarray, float, float]:
        """Return (translation_mm, u, v) for a viewer truly at ``eye_world_mm``."""
        eye_world_mm = np.asarray(eye_world_mm, dtype=np.float64)
        eye_cam = eye_world_mm - self.WEBCAM_OFFSET_MM
        tvec = R_CV_TO_HCPD.T @ eye_cam
        u = self.intrinsics.cx + self.intrinsics.fx * tvec[0] / tvec[2]
        v = self.intrinsics.cy + self.intrinsics.fy * tvec[1] / tvec[2]
        translation = _opencv_to_hcpd_translation(tvec, self.WEBCAM_OFFSET_MM)
        return translation, float(u), float(v)

    def test_no_correction_when_iris_lands_on_the_pnp_ray(self) -> None:
        """
        THE invariant. If the iris projects exactly where the PnP anchor says it
        should, the two estimates agree and the correction must be zero. An
        addition can never be zero, which is exactly how the gain hid.
        """
        for eye in ([0, 0, 600], [100, 0, 600], [200, 0, 600], [0, 100, 600],
                    [0, -100, 600], [100, 100, 600], [-150, -80, 450],
                    [180, -120, 900]):
            translation, u, v = self._project_eye(eye)
            refined = self.estimator.estimate(
                _head_result_at(translation), _iris_at(u, v), timestamp=0.0
            )
            np.testing.assert_allclose(
                refined.translation_mm[:2], np.asarray(eye, float)[:2], atol=1e-9
            )

    def test_correction_is_exactly_the_blend_fraction(self) -> None:
        """Moves IRIS_BLEND of the way toward the iris estimate, never past it."""
        translation, u0, v0 = self._project_eye([100.0, 0.0, 600.0])
        depth_cam = float(translation[2]) - self.WEBCAM_OFFSET_MM[2]
        for du in (-200.0, -60.0, 60.0, 200.0):
            u = u0 + du
            refined = self.estimator.estimate(
                _head_result_at(translation), _iris_at(u, v0), timestamp=0.0
            )
            iris_x = (
                -(u - self.intrinsics.cx) * depth_cam / self.intrinsics.fx
                + self.WEBCAM_OFFSET_MM[0]
            )
            moved = float(refined.translation_mm[0]) - float(translation[0])
            self.assertAlmostEqual(
                moved / (iris_x - float(translation[0])),
                EyePositionEstimator.IRIS_BLEND,
                places=12,
            )

    def test_vertical_offset_uses_fy(self) -> None:
        """
        dy must divide by fy. Equal to fx for from_image_size, so this forces
        them apart -- otherwise the bug is invisible until a real calibration.
        """
        fx, fy = 1044.4, 900.0
        estimator = EyePositionEstimator(
            fx=fx, cx=640.0, cy=360.0, fy=fy, webcam_offset_mm=np.zeros(3)
        )
        translation = np.array([0.0, 0.0, 600.0])
        refined = estimator.estimate(
            _head_result_at(translation), _iris_at(640.0, 360.0 + 90.0), timestamp=0.0
        )
        expected_iris_y = -90.0 * 600.0 / fy
        self.assertAlmostEqual(
            float(refined.translation_mm[1]),
            EyePositionEstimator.IRIS_BLEND * expected_iris_y,
            places=9,
        )
        # And that it is NOT the fx answer, so the assertion has teeth.
        self.assertNotAlmostEqual(
            float(refined.translation_mm[1]),
            EyePositionEstimator.IRIS_BLEND * (-90.0 * 600.0 / fx),
            places=3,
        )

    def test_z_is_passed_through_untouched(self) -> None:
        translation, u, v = self._project_eye([120.0, -40.0, 700.0])
        refined = self.estimator.estimate(
            _head_result_at(translation), _iris_at(u + 150.0, v - 90.0), timestamp=0.0
        )
        self.assertAlmostEqual(
            float(refined.translation_mm[2]), float(translation[2]), places=12
        )


class GarbageSolveCounterTest(unittest.TestCase):
    """
    Nghiệm rác: tz dương nhưng khoảng cách vô lý.

    Lớp này lọt qua cổng tz <= 0 và bị bộ clamp trong tracking_worker che mất.
    Bộ đếm CHỈ QUAN SÁT — nó không được chặn khung nào, vì trên toàn bộ dữ
    liệu có được SQPNP sinh 0/1200 và 0/451 nghiệm rác, tức chưa có gì để
    chặn. Xem docs/document.md §4.6.
    """

    def setUp(self) -> None:
        self.intrinsics = CameraIntrinsics.from_image_size(1280, 720)

    def _landmarks_at(self, distance_mm: float) -> LandmarkSet:
        """Chiếu model mặt ở một khoảng cách biết trước để sinh landmark 2D."""
        import cv2

        points, _ = cv2.projectPoints(
            FACE_MODEL_MM,
            np.zeros((3, 1)),
            np.array([[0.0], [0.0], [float(distance_mm)]]),
            self.intrinsics.matrix,
            self.intrinsics.dist_coeffs,
        )
        points = points.reshape(-1, 2)
        return LandmarkSet(
            image_points=points, left_iris=points[2], right_iris=points[3]
        )

    def test_sane_distance_does_not_count(self) -> None:
        estimator = HeadPoseEstimator(intrinsics=self.intrinsics, face_scale=1.0)
        estimator.estimate(self._landmarks_at(600.0), timestamp=0.0)
        self.assertEqual(estimator.garbage_solve_count, 0)

    def test_far_garbage_is_counted(self) -> None:
        # 3000 mm nằm ngoài cận trên 1500 mm của DIST_SANE_MM.
        estimator = HeadPoseEstimator(intrinsics=self.intrinsics, face_scale=1.0)
        estimator.estimate(self._landmarks_at(3000.0), timestamp=0.0)
        self.assertEqual(estimator.garbage_solve_count, 1)

    def test_counting_does_not_reject_the_frame(self) -> None:
        # Quan sát, không chặn: khung vẫn phải đi qua với detected=True.
        estimator = HeadPoseEstimator(intrinsics=self.intrinsics, face_scale=1.0)
        result = estimator.estimate(self._landmarks_at(3000.0), timestamp=0.0)
        self.assertTrue(result.detected)

    def test_close_frames_at_142mm_are_not_garbage(self) -> None:
        # Hồi quy cụ thể: cận dưới cũ 200 mm đánh dấu 4.8% khung THẬT của
        # người dùng là rác. Bản ghi chạm 142 mm và đó là khoảng cách thật.
        estimator = HeadPoseEstimator(intrinsics=self.intrinsics, face_scale=1.0)
        estimator.estimate(self._landmarks_at(142.0), timestamp=0.0)
        self.assertEqual(estimator.garbage_solve_count, 0)
