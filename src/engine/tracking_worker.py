"""Background tracking worker with MediaPipe and solvePnP."""

from __future__ import annotations

import logging
import threading
import time

import numpy as np

from src.config.schema import AppConfig
from src.engine.camera_worker import CameraWorker
from src.engine.pose_bus import PoseBus
from src.filters import PoseFilterPipeline
from src.math.pose_utils import Pose6DoF
from src.math.quaternion import from_axis_angle
from src.pose.calibration import resolve_face_scale, resolve_intrinsics
from src.pose.eye_position import EyePositionEstimator
from src.pose.head_pose import HeadPoseEstimator, HeadPoseResult, z_diagnostics
from src.renderer.webcam_preview import TrackingPreviewSnapshot
from src.tracking import FaceTracker

logger = logging.getLogger(__name__)


class TrackingWorker:
    """
    Thread 2: MediaPipe face/iris tracking, solvePnP, filtering.

    Consumes frames from CameraWorker internally — never displays them.
    Falls back to keyboard nudge when face is lost (dev aid only).
    """

    #: Mặc định ở mức class để bản dựng bằng ``__new__`` (test) vẫn chạy —
    #: cùng lý do với ``GlfwWindow._demo``.
    _preview_enabled = True

    def __init__(
        self,
        pose_bus: PoseBus,
        filter_pipeline: PoseFilterPipeline,
        camera_worker: CameraWorker,
        config: AppConfig,
        webcam_offset_mm: np.ndarray | None = None,
    ) -> None:
        self._pose_bus = pose_bus
        self._filters = filter_pipeline
        self._camera = camera_worker
        self._config = config

        cam = config.camera
        mount = config.webcam_mount
        opt = config.optimization
        self._intrinsics = resolve_intrinsics(cam.width, cam.height)

        # Prefer the offset from screen.json (physical measurement) over the
        # generic webcam_mount config.  webcam_offset_mm is in HCPD/world mm:
        # X right, Y up, Z toward viewer.
        if webcam_offset_mm is not None:
            effective_offset = np.asarray(webcam_offset_mm, dtype=np.float64)
        else:
            effective_offset = np.array(
                [mount.offset_x_mm, mount.offset_y_mm, mount.offset_z_mm],
                dtype=np.float64,
            )

        # PHASE B: use the stored per-user face scale when present; absent or
        # unusable calibration keeps the legacy _monocular_scale path.
        #
        # resolve_face_scale, not load_face_calibration: the stored scale is only
        # valid for the fx it was measured with, and it now has to answer to
        # whatever resolve_intrinsics just produced.
        face_scale = resolve_face_scale(self._intrinsics)
        self._head_pose = HeadPoseEstimator(
            intrinsics=self._intrinsics,
            reference_distance_mm=config.user.viewing_distance_mm,
            webcam_offset_mm=effective_offset,
            fast_solve_pnp=opt.fast_solve_pnp,
            face_scale=face_scale,
        )
        self._eye_estimator = EyePositionEstimator(
            ipd_mm=config.user.ipd_mm,
            fx=self._intrinsics.fx,
            cx=self._intrinsics.cx,
            cy=self._intrinsics.cy,
            fy=self._intrinsics.fy,
            # SAME offset the head pose uses. The iris estimate is lens-centred
            # until this is applied, and blending a lens-centred value into a
            # screen-centred one is a constant bias of -IRIS_BLEND * offset.
            webcam_offset_mm=effective_offset,
        )
        track = config.tracking
        self._face_tracker = FaceTracker(
            min_detection_confidence=track.min_detection_confidence,
            min_tracking_confidence=track.min_tracking_confidence,
            model_path=track.model_path,
        )

        self._eye_mm = np.array(
            [0.0, 0.0, config.user.viewing_distance_mm],
            dtype=np.float64,
        )
        self._rotation = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self._last_head = HeadPoseResult(
            detected=False,
            pose=Pose6DoF(
                translation_mm=self._eye_mm.copy(),
                rotation=self._rotation.copy(),
                timestamp=0.0,
            ),
            pitch_deg=0.0,
            yaw_deg=0.0,
            roll_deg=0.0,
            translation_mm=self._eye_mm.copy(),
            rotation_matrix=np.eye(3),
        )
        self._face_detected = False
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._input_lock = threading.Lock()
        self._input_delta = np.zeros(3, dtype=np.float64)
        self._tracking_latency_ms = 0.0
        self._track_max_fps = opt.track_max_fps
        self._tracking_scale = 1.0
        self._downscale_enabled = opt.tracking_downscale

        # Low-confidence hold: when MediaPipe reports poor presence
        # (steep angle, partial occlusion), hold the last stable eye position
        # for up to _hold_max_frames rather than feeding bad data to the filter.
        self._confidence_threshold = 0.25
        self._hold_max_frames = 8
        self._hold_frames_remaining = 0
        self._last_confident_eye: np.ndarray | None = None
        self._last_confident_rotation: np.ndarray | None = None
        self._last_frame_index = -1
        self._preview_lock = threading.Lock()
        self._preview_snapshot = TrackingPreviewSnapshot()
        #: TASK_ui_v3_1107: giao diện v3 bỏ hẳn PiP webcam (F4), nên ảnh chụp
        #: landmark không ai dùng tới. Dựng nó vẫn tốn ba lượt ``.copy()``
        #: mảng điểm MỖI khung tracking — tắt hẳn cho rẻ.
        self._preview_enabled = True

    def set_preview_enabled(self, enabled: bool) -> None:
        """Bật/tắt việc dựng ảnh chụp landmark cho PiP."""
        self._preview_enabled = bool(enabled)

    def _store_preview(self, snapshot) -> None:
        if not self._preview_enabled:
            return
        with self._preview_lock:
            self._preview_snapshot = snapshot

    @property
    def tracking_latency_ms(self) -> float:
        return self._tracking_latency_ms

    @property
    def face_detected(self) -> bool:
        return self._face_detected

    @property
    def tracker_available(self) -> bool:
        return self._face_tracker.is_available

    def preview_snapshot(self) -> TrackingPreviewSnapshot:
        """Thread-safe copy of latest landmark overlay data for the PiP HUD."""
        with self._preview_lock:
            snap = self._preview_snapshot
            pnp = snap.pnp_points.copy() if snap.pnp_points is not None else None
            left = snap.left_iris.copy() if snap.left_iris is not None else None
            right = snap.right_iris.copy() if snap.right_iris is not None else None
            return TrackingPreviewSnapshot(
                detected=snap.detected,
                confidence=snap.confidence,
                image_width=snap.image_width,
                image_height=snap.image_height,
                pnp_points=pnp,
                left_iris=left,
                right_iris=right,
            )

    def set_tracking_scale(self, scale: float) -> None:
        """Set MediaPipe input downscale factor from adaptive quality."""
        self._tracking_scale = float(np.clip(scale, 0.25, 1.0))

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop,
            name="TrackingWorker",
            daemon=True,
        )
        self._thread.start()
        mode = "MediaPipe" if self._face_tracker.is_available else "simulation"
        logger.info("Tracking worker started (%s)", mode)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._face_tracker.close()

    def apply_input(self, move_x: float, move_y: float, move_z: float, dt: float) -> None:
        """Keyboard nudge when face lost or for fine-tuning (dev aid)."""
        speed_mm_s = 180.0
        with self._input_lock:
            self._input_delta[0] += move_x * speed_mm_s * dt
            self._input_delta[1] += move_y * speed_mm_s * dt
            self._input_delta[2] += move_z * speed_mm_s * dt

    def head_angles_deg(self) -> tuple[float, float, float]:
        """Return pitch, yaw, roll from last head pose estimate."""
        return (
            self._last_head.pitch_deg,
            self._last_head.yaw_deg,
            self._last_head.roll_deg,
        )

    def _loop(self) -> None:
        logger.debug("TrackingWorker loop entered")
        try:
            while not self._stop.is_set():
                try:
                    self._loop_once()
                except Exception:
                    logger.exception("Tracking iteration failed; continuing")
                    time.sleep(0.01)
        except Exception:
            logger.exception("TrackingWorker loop crashed")
        finally:
            logger.debug("TrackingWorker loop exited")

    def _loop_once(self) -> None:
        start = time.perf_counter()
        now = start

        packet = self._camera.pop_frame()
        # A repeated frame index means no fresh measurement: the tracking loop
        # runs up to track_max_fps while the camera delivers at its own rate,
        # so roughly half of these iterations re-publish the previous pose.
        # Use the frame's own capture timestamp, not the loop time, so the
        # predictor differentiates over the real capture interval.
        new_measurement = packet is not None and (
            packet.frame_index != self._last_frame_index
        )
        if packet is not None:
            now = float(packet.timestamp)
            self._last_frame_index = packet.frame_index
        raw_pose = self._estimate_pose(packet, now)

        with self._input_lock:
            if np.any(self._input_delta != 0.0):
                raw_pose = Pose6DoF(
                    translation_mm=raw_pose.translation_mm + self._input_delta,
                    rotation=raw_pose.rotation,
                    timestamp=raw_pose.timestamp,
                )
                self._input_delta.fill(0.0)

        translation = np.asarray(
            raw_pose.translation_mm, dtype=np.float64
        ).reshape(3).copy()
        if not np.all(np.isfinite(translation)):
            translation = self._eye_mm.copy()
        translation[0] = float(np.clip(translation[0], -2000.0, 2000.0))
        translation[1] = float(np.clip(translation[1], -2000.0, 2000.0))
        translation[2] = float(np.clip(translation[2], 250.0, 4000.0))
        z_diagnostics.set(z_after_clamp=float(translation[2]))
        raw_pose = Pose6DoF(
            translation_mm=translation,
            rotation=raw_pose.rotation,
            timestamp=raw_pose.timestamp,
        )

        filtered = self._filters.update(raw_pose, new_measurement=new_measurement)
        self._eye_mm = filtered.translation_mm
        self._rotation = filtered.rotation
        self._pose_bus.publish(filtered)
        z_diagnostics.set(z_published=float(filtered.translation_mm[2]))
        z_diagnostics.commit()

        self._tracking_latency_ms = (time.perf_counter() - start) * 1000.0
        if self._track_max_fps > 0.0:
            budget_s = 1.0 / self._track_max_fps
            elapsed_s = self._tracking_latency_ms / 1000.0
            time.sleep(max(0.0, budget_s - elapsed_s))

    def _estimate_pose(self, packet, timestamp: float) -> Pose6DoF:
        if packet is None or not self._face_tracker.is_available:
            return self._simulated_pose(timestamp)

        scale = self._tracking_scale if self._downscale_enabled else 1.0
        tracking = self._face_tracker.process(packet.image, process_scale=scale)
        if not tracking.detected or tracking.landmarks is None:
            self._face_detected = False
            self._hold_frames_remaining = max(0, self._hold_frames_remaining - 1)
            if self._preview_enabled:
                self._store_preview(TrackingPreviewSnapshot(
                    detected=False,
                    image_width=tracking.image_width,
                    image_height=tracking.image_height,
                ))
            return self._hold_or_simulate(timestamp)

        if tracking.image_width != self._intrinsics.width:
            self._intrinsics = resolve_intrinsics(
                tracking.image_width,
                tracking.image_height,
            )
            # solvePnP's K must follow too. It did not: only the iris estimator
            # was updated, so the whole session kept solving against intrinsics
            # built for a resolution the camera never delivered.
            self._head_pose.update_intrinsics(
                self._intrinsics,
                face_scale=resolve_face_scale(self._intrinsics),
            )
            self._eye_estimator.update_intrinsics(
                self._intrinsics.fx,
                self._intrinsics.cx,
                self._intrinsics.cy,
                fy=self._intrinsics.fy,
            )

        head = self._head_pose.estimate(tracking.landmarks, timestamp)
        if not head.detected:
            # MediaPipe thấy mặt nhưng solvePnP không cho nghiệm dùng được
            # (tz <= 0, tức nghiệm lật ra sau camera). head_pose trả về một pose
            # identity neo ở reference_distance cho trường hợp này, và comment
            # của nó hứa rằng "caller's hold-last-pose path takes over" — nhưng
            # trước đây caller KHÔNG hề kiểm head.detected, nên pose identity đi
            # thẳng xuống dưới: mắt NHẢY về 600 mm, rồi vì confidence luôn bằng
            # 1.0 nên nhánh dưới còn ghi luôn cú nhảy đó vào _last_confident_eye,
            # đầu độc giá trị dự phòng cho các khung sau.
            #
            # Giữ pose đã publish gần nhất, đúng như lời hứa. Xử lý giống hệt
            # nhánh MediaPipe trượt ở trên để hai đường hỏng có cùng hành vi.
            #
            # KHÔNG gán self._last_head = head ở đây: _hold_or_simulate đọc
            # _last_head.detected để quyết định giữ hay chuyển sang pose mô
            # phỏng. Ghi đè bằng một kết quả hỏng sẽ làm khung kế tiếp rơi
            # xuống mô phỏng thay vì giữ.
            self._face_detected = False
            self._hold_frames_remaining = max(0, self._hold_frames_remaining - 1)
            lm_failed = tracking.landmarks
            if self._preview_enabled:
                # Landmark vẫn hợp lệ — chỉ có phép giải pose hỏng. Vẫn vẽ
                # skeleton lên PiP, còn HUD thì báo LOST qua _face_detected.
                self._store_preview(TrackingPreviewSnapshot(
                    detected=True,
                    confidence=tracking.confidence,
                    image_width=tracking.image_width,
                    image_height=tracking.image_height,
                    pnp_points=lm_failed.image_points.copy(),
                    left_iris=lm_failed.left_iris.copy(),
                    right_iris=lm_failed.right_iris.copy(),
                ))
            return self._hold_or_simulate(timestamp)

        eye_pose = self._eye_estimator.estimate(head, tracking.landmarks, timestamp)
        # Set after estimate(): HeadPoseEstimator.estimate starts the diag row.
        z_diagnostics.set(presence=float(tracking.confidence))
        self._face_detected = True
        self._last_head = head

        lm = tracking.landmarks
        if self._preview_enabled:
            self._store_preview(TrackingPreviewSnapshot(
                detected=True,
                confidence=tracking.confidence,
                image_width=tracking.image_width,
                image_height=tracking.image_height,
                pnp_points=lm.image_points.copy(),
                left_iris=lm.left_iris.copy(),
                right_iris=lm.right_iris.copy(),
            ))

        # Low-confidence hold: steep angles / partial occlusion produce
        # unreliable solvePnP results.  Keep the last confident reading
        # for a short window rather than snapping to a bad pose.
        if tracking.confidence >= self._confidence_threshold:
            self._last_confident_eye = eye_pose.translation_mm.copy()
            self._last_confident_rotation = eye_pose.rotation.copy()
            self._hold_frames_remaining = self._hold_max_frames
            return eye_pose

        # Confidence below threshold — hold last stable pose if available.
        if self._last_confident_eye is not None and self._hold_frames_remaining > 0:
            self._hold_frames_remaining -= 1
            logger.debug(
                "Low confidence %.2f — holding last pose (%d frames left)",
                tracking.confidence, self._hold_frames_remaining,
            )
            return Pose6DoF(
                translation_mm=self._last_confident_eye.copy(),
                rotation=self._last_confident_rotation.copy(),  # type: ignore[arg-type]
                timestamp=timestamp,
            )

        # Confidence low and no hold budget left: pass through but log warning.
        logger.warning(
            "Low face presence score %.2f — pose may be unreliable", tracking.confidence
        )
        return eye_pose

    def _hold_or_simulate(self, timestamp: float) -> Pose6DoF:
        if self._last_head.detected:
            return Pose6DoF(
                translation_mm=self._eye_mm.copy(),
                rotation=self._rotation.copy(),
                timestamp=timestamp,
            )
        return self._simulated_pose(timestamp)

    def _simulated_pose(self, timestamp: float) -> Pose6DoF:
        """Fallback when no camera frame or MediaPipe unavailable."""
        yaw = self._eye_mm[0] / 600.0
        rotation = from_axis_angle(
            np.array([0.0, 1.0, 0.0], dtype=np.float64),
            yaw * 0.15,
        )
        return Pose6DoF(
            translation_mm=self._eye_mm.copy(),
            rotation=rotation,
            timestamp=timestamp,
        )
