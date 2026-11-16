"""
Kiểm đường đi khi solvePnP hỏng nhưng MediaPipe vẫn thấy mặt.

Bối cảnh (đo ngày 2026-09-19): `confidence` của FaceTracker luôn bằng đúng 1.0
vì model face mesh không xuất presence theo từng landmark, nên nhánh giữ pose
theo confidence trong TrackingWorker chưa từng chạy. Cổng chết đó che một
đường đi hỏng thật: khi tz <= 0, head_pose trả HeadPoseResult(detected=False)
kèm pose identity neo ở reference_distance, và trước đây không ai kiểm
head.detected — pose identity đi thẳng xuống, mắt NHẢY về 600 mm và cú nhảy
được ghi luôn vào _last_confident_eye.

Test dựng instance bằng object.__new__ rồi gán đúng những thuộc tính mà
_estimate_pose đụng tới. Cố ý không gọi __init__ thật: nó nạp model MediaPipe
2.5 MB, không cần cho phép kiểm này.
"""

from __future__ import annotations

import threading
import unittest

import numpy as np

from src.engine.tracking_worker import TrackingWorker
from src.math.pose_utils import Pose6DoF
from src.pose.head_pose import HeadPoseResult
from src.tracking.face_tracker import FaceTrackingResult
from src.tracking.landmarks import LandmarkSet

HELD_EYE_MM = np.array([37.0, -12.0, 415.0], dtype=np.float64)
REFERENCE_DISTANCE_MM = 600.0


class _PacketStub:
    """Đứng thay CapturePacket — _estimate_pose chỉ đọc .image."""

    image = np.zeros((720, 1280, 3), dtype=np.uint8)


def _landmarks() -> LandmarkSet:
    """Landmark hợp lệ bất kỳ — phép giải pose mới là thứ bị ép hỏng."""
    return LandmarkSet(
        image_points=np.arange(12, dtype=np.float64).reshape(6, 2),
        left_iris=np.array([610.0, 350.0]),
        right_iris=np.array([670.0, 350.0]),
    )


class _TrackerStub:
    """Đứng thay FaceTracker: luôn báo thấy mặt với confidence 1.0."""

    is_available = True

    def process(self, image, process_scale: float = 1.0) -> FaceTrackingResult:
        return FaceTrackingResult(
            detected=True,
            landmarks=_landmarks(),
            confidence=1.0,
            image_width=1280,
            image_height=720,
        )


class _HeadPoseStub:
    """Đứng thay HeadPoseEstimator: tái hiện đúng nhánh tz <= 0."""

    def __init__(self, detected: bool) -> None:
        self._detected = detected

    def estimate(self, landmarks, timestamp: float) -> HeadPoseResult:
        identity = Pose6DoF(
            translation_mm=np.array([0.0, 0.0, REFERENCE_DISTANCE_MM]),
            rotation=np.array([1.0, 0.0, 0.0, 0.0]),
            timestamp=timestamp,
        )
        return HeadPoseResult(
            detected=self._detected,
            pose=identity,
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            translation_mm=identity.translation_mm,
            rotation_matrix=np.eye(3),
        )


class _EyeStub:
    """EyePositionEstimator thật trả head_pose.pose khi detected=False."""

    def estimate(self, head_pose, landmarks, timestamp: float) -> Pose6DoF:
        return head_pose.pose


def _worker(*, solve_succeeds: bool) -> TrackingWorker:
    """Dựng worker tối thiểu, bỏ qua __init__ nặng."""
    worker = object.__new__(TrackingWorker)
    worker._face_tracker = _TrackerStub()
    worker._head_pose = _HeadPoseStub(detected=solve_succeeds)
    worker._eye_estimator = _EyeStub()
    worker._downscale_enabled = False
    worker._tracking_scale = 1.0
    worker._preview_lock = threading.Lock()
    worker._preview_snapshot = None
    worker._face_detected = True
    worker._hold_frames_remaining = 8
    worker._confidence_threshold = 0.25
    worker._hold_max_frames = 8
    worker._last_confident_eye = HELD_EYE_MM.copy()
    worker._last_confident_rotation = np.array([1.0, 0.0, 0.0, 0.0])
    worker._eye_mm = HELD_EYE_MM.copy()
    worker._rotation = np.array([1.0, 0.0, 0.0, 0.0])
    # _hold_or_simulate đọc _last_head.detected để chọn giữ hay mô phỏng.
    worker._last_head = _HeadPoseStub(detected=True).estimate(None, 0.0)
    worker._intrinsics = type("I", (), {"width": 1280})()
    return worker


class FailedSolveHoldsLastPoseTest(unittest.TestCase):
    """solvePnP hỏng phải giữ pose cũ, không được nhảy về reference_distance."""

    def test_failed_solve_does_not_snap_to_reference_distance(self) -> None:
        worker = _worker(solve_succeeds=False)
        pose = worker._estimate_pose(packet=_PacketStub(), timestamp=1.0)
        # Đây là hồi quy cụ thể: trước bản sửa, Z trả về đúng 600.0 mm.
        self.assertNotAlmostEqual(
            float(pose.translation_mm[2]), REFERENCE_DISTANCE_MM, places=3
        )
        np.testing.assert_allclose(pose.translation_mm, HELD_EYE_MM)

    def test_failed_solve_does_not_poison_the_fallback(self) -> None:
        worker = _worker(solve_succeeds=False)
        worker._estimate_pose(packet=_PacketStub(), timestamp=1.0)
        # _last_confident_eye phải giữ nguyên giá trị tốt trước đó.
        np.testing.assert_allclose(worker._last_confident_eye, HELD_EYE_MM)

    def test_failed_solve_reports_face_lost_on_the_hud(self) -> None:
        worker = _worker(solve_succeeds=False)
        worker._estimate_pose(packet=_PacketStub(), timestamp=1.0)
        self.assertFalse(worker._face_detected)

    def test_failed_solve_still_draws_the_skeleton(self) -> None:
        # Landmark vẫn hợp lệ; giấu skeleton sẽ ngụ ý MediaPipe mất mặt.
        worker = _worker(solve_succeeds=False)
        worker._estimate_pose(packet=_PacketStub(), timestamp=1.0)
        self.assertTrue(worker._preview_snapshot.detected)
        self.assertIsNotNone(worker._preview_snapshot.pnp_points)

    def test_failed_solve_keeps_last_head_intact(self) -> None:
        # Ghi đè _last_head bằng kết quả hỏng sẽ làm khung sau rơi xuống pose
        # mô phỏng thay vì giữ pose thật.
        worker = _worker(solve_succeeds=False)
        worker._estimate_pose(packet=_PacketStub(), timestamp=1.0)
        self.assertTrue(worker._last_head.detected)

    def test_successful_solve_still_passes_through(self) -> None:
        worker = _worker(solve_succeeds=True)
        pose = worker._estimate_pose(packet=_PacketStub(), timestamp=1.0)
        self.assertAlmostEqual(
            float(pose.translation_mm[2]), REFERENCE_DISTANCE_MM, places=3
        )
        self.assertTrue(worker._face_detected)


if __name__ == "__main__":
    unittest.main()
