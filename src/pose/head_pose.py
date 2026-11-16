"""Convert solvePnP output to 6DoF head pose."""

from __future__ import annotations

import atexit
import csv
import json
import logging
import math
import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from src.math.pose_utils import Pose6DoF
from src.math.quaternion import from_rotation_matrix
from src.pose.calibration import CameraIntrinsics
from src.pose.solve_pnp import solve_head_pose
from src.tracking.landmarks import FACE_MODEL_MM, LandmarkSet

logger = logging.getLogger(__name__)


#: The one basis change from the OpenCV camera frame to HCPD world.
#:
#: OpenCV camera : X right in image, Y down, Z forward into the scene
#: HCPD world    : X right (viewer's right), Y up, Z toward the viewer
#:
#: The camera sits on the screen looking back at the viewer, so Z_cam already
#: points along world +Z. With a right-handed basis (X x Y = Z) that forces
#: X_cam = Y_cam x Z_cam = (0,-1,0) x (0,0,1) = (-1,0,0), i.e. camera X runs
#: along world -X. Hence diag(-1, -1, 1), determinant +1.
#:
#: This replaces three different conventions that used to coexist: translation
#: applied diag(1,-1,1) (determinant -1, a reflection, not a rigid motion),
#: rotation applied diag(1,-1,-1) (a rotation, but about a different axis), and
#: the iris refinement in eye_position.py negated Y but not X. Confirmed against
#: physical ruler marks in the tracking benchmark, whose results header records
#: that predictions needed -t_x to line up with the tape: that measurement is
#: end-to-end, so it also rules out any driver-level mirroring of the frame.
R_CV_TO_HCPD: npt.NDArray[np.float64] = np.diag([-1.0, -1.0, 1.0])


#: Khoảng cách đầu→camera coi là hợp lý, mm. Ngoài dải này mà ``tz`` VẪN DƯƠNG
#: thì là **nghiệm rác**: cổng ``tz <= 0`` không bắt được lớp này, và bộ clamp
#: trong tracking_worker che mất nó.
#:
#: Cận dưới là 120 chứ không phải 200. Bản ghi 1200 khung thật chạm **142 mm**,
#: và đó là khoảng cách THẬT chứ không phải nghiệm hỏng: 58 khung đó gom thành
#: 5 đoạn liên tục, bề rộng hai mắt trên ảnh đạt 625 px — lớn nhất toàn chuỗi —
#: và tam giác đồng dạng ``fx·E/625`` cho 150 mm, khớp solvePnP trong 6%; tương
#: quan ``|t|`` với nghịch đảo bề rộng mắt là 0.91. Để 200 thì báo động giả
#: 4.8% trên chính dữ liệu người dùng.
#:
#: 120 mm suy từ hình học: dưới mức đó khuôn mặt không còn lọt khung ở FOV 63°.
#: **Con số đó thừa hưởng một giả định** — FOV 63° là ƯỚC LƯỢNG của
#: ``CameraIntrinsics.from_image_size``, không phải hiệu chuẩn camera thật. Có
#: hiệu chuẩn thật thì phải tính lại.
#:
#: Định nghĩa ở ĐÂY và scripts/compare_solvers.py import về, để chỉ có MỘT
#: nguồn sự thật. Hai bản sao của cùng một ngưỡng là đúng loại bẫy mà dự án này
#: đã dính nhiều lần (xem PHẦN V).
DIST_SANE_MM: tuple[float, float] = (120.0, 1500.0)

#: Cứ mỗi ngần này lần thì ghi một dòng log nghiệm rác. CHỈ ghi khi kích hoạt —
#: không ghi mỗi khung. Lớp lỗi này hiếm (0/1200 và 0/451 trên dữ liệu có
#: được), nên chi phí khi không có gì xảy ra phải bằng 0.
GARBAGE_LOG_EVERY: int = 20


# ---------------------------------------------------------------------------
# PHASE A — Z-axis diagnostics.  READ + LOG ONLY: nothing here alters the pose
# that reaches PoseBus.  Gated behind assets/config/default.json:
#
#     "debug": { "z_diagnostics": false }
#
# and defaults to OFF.  When OFF every hook below is an early-return, so the
# tracking hot path pays only an attribute read.
#
# Row lifecycle (one row per DETECTED frame):
#   HeadPoseEstimator.estimate      begin_frame() + tvec_z_raw, depth_raw,
#                                   reference_depth_raw, z_after_scale,
#                                   reproj_rmse_px
#   EyePositionEstimator.estimate   z_after_iris
#   TrackingWorker._estimate_pose   presence
#   TrackingWorker._loop_once       z_after_clamp, z_published, commit()
#
# NOTE for whoever reads the CSV: `z_after_scale` is logged as the literal
# output of _monocular_scale, BEFORE _opencv_to_hcpd_translation adds
# webcam_offset_mm.  With the current screen.json that offset adds +5.0 mm on
# Z, so a constant +5.0 mm step between `z_after_scale` and `z_after_iris` is
# expected and is NOT a bug.
# ---------------------------------------------------------------------------

Z_DIAG_COLUMNS = (
    "frame_idx",
    "tvec_z_raw",
    "depth_raw",
    "reference_depth_raw",
    "z_after_scale",
    "z_after_iris",
    "z_after_clamp",
    "z_published",
    "reproj_rmse_px",
    "presence",
)


def _read_z_diagnostics_flag() -> bool:
    """Resolve ``debug.z_diagnostics`` from default.json, then user.json."""
    root = Path(__file__).resolve().parents[2]
    enabled = False
    for name in ("default.json", "user.json"):
        path = root / "assets" / "config" / name
        if not path.exists():
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("Cannot read %s for z_diagnostics flag: %s", name, exc)
            continue
        debug = payload.get("debug")
        if isinstance(debug, dict) and "z_diagnostics" in debug:
            enabled = bool(debug["z_diagnostics"])
    return enabled


class _ZDiagnostics:
    """Per-frame CSV recorder for the Z chain. Disabled unless configured on."""

    def __init__(self) -> None:
        self._enabled: bool | None = None
        self._lock = threading.Lock()
        self._handle = None
        self._writer: csv.DictWriter | None = None
        self._row: dict | None = None
        self._frame_idx = 0

    @property
    def enabled(self) -> bool:
        if self._enabled is None:
            with self._lock:
                if self._enabled is None:
                    self._enabled = _read_z_diagnostics_flag()
                    if self._enabled:
                        self._open()
        return bool(self._enabled)

    def _open(self) -> None:
        root = Path(__file__).resolve().parents[2]
        path = root / "artifacts" / "z_diag.csv"
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._handle = path.open("w", encoding="utf-8", newline="")
        except OSError as exc:
            logger.error("z_diagnostics: cannot open %s (%s) — disabling", path, exc)
            self._enabled = False
            return
        self._writer = csv.DictWriter(self._handle, fieldnames=list(Z_DIAG_COLUMNS))
        self._writer.writeheader()
        self._handle.flush()
        atexit.register(self.close)
        logger.info("z_diagnostics ENABLED — writing %s", path)

    def begin_frame(self) -> None:
        if not self.enabled:
            return
        self._row = {name: "" for name in Z_DIAG_COLUMNS}

    def set(self, **values: float) -> None:
        if not self.enabled or self._row is None:
            return
        for key, value in values.items():
            if key in self._row:
                self._row[key] = value

    def commit(self) -> None:
        """Write the pending row. No-op when no frame was started."""
        if not self.enabled or self._row is None or self._writer is None:
            return
        row = self._row
        self._row = None
        row["frame_idx"] = self._frame_idx
        self._frame_idx += 1
        try:
            self._writer.writerow(row)
            if self._handle is not None:
                self._handle.flush()
        except (OSError, ValueError) as exc:
            logger.error("z_diagnostics: row write failed (%s) — disabling", exc)
            self._enabled = False

    def close(self) -> None:
        with self._lock:
            if self._handle is not None:
                try:
                    self._handle.close()
                except OSError:
                    pass
                self._handle = None
                self._writer = None


#: Process-wide recorder shared by head_pose / eye_position / tracking_worker.
z_diagnostics = _ZDiagnostics()


def _reprojection_rmse_px(
    model_points_mm: npt.NDArray[np.float64],
    image_points_px: npt.NDArray[np.float64],
    intrinsics: CameraIntrinsics,
    rvec: npt.NDArray[np.float64],
    tvec: npt.NDArray[np.float64],
) -> float:
    """RMS pixel distance between projected model points and the landmarks."""
    projected, _ = cv2.projectPoints(
        model_points_mm.astype(np.float64),
        np.asarray(rvec, dtype=np.float64).reshape(3, 1),
        np.asarray(tvec, dtype=np.float64).reshape(3, 1),
        intrinsics.matrix,
        intrinsics.dist_coeffs,
    )
    diff = projected.reshape(-1, 2) - image_points_px.reshape(-1, 2)
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=1))))


@dataclass(frozen=True)
class HeadPoseResult:
    """
    Raw and processed head pose from one frame.

    Frames, because mixing them up here is this module's recurring bug:

    - ``pose.rotation``, ``pitch_deg``/``yaw_deg``/``roll_deg`` and
      ``translation_mm`` are all **HCPD** (X right, Y up, Z toward viewer).
    - ``rotation_matrix`` is the **raw OpenCV** ``cv2.Rodrigues`` output, kept
      unconverted on purpose as a diagnostic. It has no consumers; convert with
      ``_opencv_rot_to_hcpd`` before comparing it to anything above.
    """

    detected: bool
    pose: Pose6DoF
    pitch_deg: float
    yaw_deg: float
    roll_deg: float
    translation_mm: npt.NDArray[np.float64]
    rotation_matrix: npt.NDArray[np.float64]


class HeadPoseEstimator:
    """
    Estimate 6DoF head pose from face landmarks using solvePnP.

    Outputs pose in head-coupled display coordinates:
    +X right, +Y up, +Z toward the viewer (mm, screen-centered).
    """

    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        reference_distance_mm: float = 600.0,
        webcam_offset_mm: npt.NDArray[np.float64] | None = None,
        fast_solve_pnp: bool = True,
        face_scale: float | None = None,
    ) -> None:
        self._intrinsics = intrinsics
        self._reference_distance = reference_distance_mm
        self._reference_depth_raw: float | None = None
        self._garbage_solves = 0
        self._fast_solve = fast_solve_pnp
        self._webcam_offset = (
            webcam_offset_mm
            if webcam_offset_mm is not None
            else np.zeros(3, dtype=np.float64)
        )
        # PHASE B: a calibrated face_scale sizes the millimetre face model to
        # this user, so solvePnP returns tvec in true millimetres and the
        # legacy _monocular_scale must be bypassed (see estimate()).
        self._face_scale = float(face_scale) if face_scale is not None else None
        if self._face_scale is not None:
            self._model = FACE_MODEL_MM * self._face_scale
            logger.info("Z path: CALIBRATED (face_scale=%.4f)", self._face_scale)
        else:
            self._model = FACE_MODEL_MM.copy()
            logger.info("Z path: LEGACY (_monocular_scale)")

    def estimate(
        self,
        landmarks: LandmarkSet,
        timestamp: float,
    ) -> HeadPoseResult:
        """Compute head pose from 2D landmarks."""
        z_diagnostics.begin_frame()
        success, rvec, tvec = solve_head_pose(
            self._model,
            landmarks.image_points,
            self._intrinsics,
            fast=self._fast_solve,
        )
        if not success:
            identity = Pose6DoF(
                translation_mm=np.array([0.0, 0.0, self._reference_distance]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=timestamp,
            )
            return HeadPoseResult(
                detected=False,
                pose=identity,
                pitch_deg=0.0,
                yaw_deg=0.0,
                roll_deg=0.0,
                translation_mm=identity.translation_mm,
                rotation_matrix=np.eye(3),
            )

        rotation_matrix, _ = cv2.Rodrigues(rvec)
        translation_cv = tvec.astype(np.float64).ravel()
        if translation_cv[2] <= 0.0:
            # The face cannot be behind the camera. A non-positive depth means
            # the PnP solve flipped; report it rather than rectifying it, so the
            # caller's hold-last-pose path takes over.
            logger.warning(
                "solvePnP returned non-positive depth (%.1f mm) — treating as "
                "a failed solve", translation_cv[2],
            )
            identity = Pose6DoF(
                translation_mm=np.array([0.0, 0.0, self._reference_distance]),
                rotation=np.array([1.0, 0.0, 0.0, 0.0]),
                timestamp=timestamp,
            )
            return HeadPoseResult(
                detected=False, pose=identity, pitch_deg=0.0, yaw_deg=0.0,
                roll_deg=0.0, translation_mm=identity.translation_mm,
                rotation_matrix=np.eye(3),
            )
        depth_raw = float(translation_cv[2])
        self._count_garbage_solve(translation_cv)

        if self._face_scale is not None:
            # CALIBRATED: the model is in true millimetres sized to this user,
            # so tvec is already in true millimetres.  _monocular_scale is
            # skipped entirely — both mechanisms correct the SAME error and
            # stacking them would apply the correction twice.
            scale = 1.0
            xy = translation_cv[:2]
            z_hcpd = depth_raw
        else:
            self._ensure_depth_baseline(depth_raw)
            scale = self._monocular_scale()

            # Scale X/Y only; Z must follow live tvec depth (per-frame ratio),
            # not reference/depth_raw which would pin Z to reference_distance
            # every frame.
            xy = translation_cv[:2] * scale
            if self._reference_depth_raw is not None and self._reference_depth_raw > 1e-3:
                z_hcpd = self._reference_distance * (
                    depth_raw / self._reference_depth_raw
                )
            else:
                z_hcpd = depth_raw * scale
        scaled = np.array([xy[0], xy[1], z_hcpd], dtype=np.float64)

        if z_diagnostics.enabled:
            z_diagnostics.set(
                tvec_z_raw=float(translation_cv[2]),
                depth_raw=depth_raw,
                reference_depth_raw=(
                    float(self._reference_depth_raw)
                    if self._reference_depth_raw is not None
                    else ""
                ),
                z_after_scale=float(z_hcpd),
                reproj_rmse_px=_reprojection_rmse_px(
                    self._model,
                    landmarks.image_points,
                    self._intrinsics,
                    rvec,
                    tvec,
                ),
            )

        hcpd_pre_offset = _opencv_to_hcpd_translation(scaled, np.zeros(3))
        translation_mm = _opencv_to_hcpd_translation(scaled, self._webcam_offset)
        # ONE basis change, then everything downstream reads the same frame.
        # The Euler triple used to come from `rotation_matrix` (raw OpenCV)
        # while `rotation` came from the converted one, so a single
        # HeadPoseResult reported the same rotation in two different bases.
        # Conjugating by diag(-1,-1,1) maps a rotation about n to one about
        # M @ n, so PITCH (about X) and YAW (about Y) flipped sign while ROLL
        # (about Z) did not: the F1 overlay showed "looking up" for a head
        # tilted down, and "turned left" for one turned right. Measured on
        # R_cv = Rz(5)Ry(30)Rx(-12): old = (-12, +30, 5), new = (+12, -30, 5).
        #
        # Same class of defect as the old translation/rotation/iris split
        # described next to R_CV_TO_HCPD, and it survived for the same reason:
        # the angles only feed the F1 overlay, never the projection.
        rotation_hcpd = _opencv_rot_to_hcpd(rotation_matrix)
        rotation = from_rotation_matrix(rotation_hcpd)
        pitch, yaw, roll = _euler_from_rotation(rotation_hcpd[:3, :3])

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "solvePnP tvec_cam=[%.1f, %.1f, %.1f] mm  depth_raw=%.1f  "
                "depth_baseline=%.1f  scale=%.4f  "
                "scaled=[%.1f, %.1f, %.1f]  hcpd_pre_offset=[%.1f, %.1f, %.1f]  "
                "eye_world=[%.1f, %.1f, %.1f] mm  cam_offset=[%.1f, %.1f, %.1f] mm  "
                "pitch=%.1f yaw=%.1f roll=%.1f deg",
                translation_cv[0], translation_cv[1], translation_cv[2],
                depth_raw,
                self._reference_depth_raw or 0.0,
                scale,
                scaled[0], scaled[1], scaled[2],
                hcpd_pre_offset[0], hcpd_pre_offset[1], hcpd_pre_offset[2],
                translation_mm[0], translation_mm[1], translation_mm[2],
                self._webcam_offset[0], self._webcam_offset[1], self._webcam_offset[2],
                pitch, yaw, roll,
            )

        pose = Pose6DoF(
            translation_mm=translation_mm,
            rotation=rotation,
            timestamp=timestamp,
        )
        return HeadPoseResult(
            detected=True,
            pose=pose,
            pitch_deg=pitch,
            yaw_deg=yaw,
            roll_deg=roll,
            translation_mm=translation_mm,
            rotation_matrix=rotation_matrix,
        )

    def update_intrinsics(
        self,
        intrinsics: CameraIntrinsics,
        face_scale: float | None = None,
    ) -> None:
        """
        Adopt new intrinsics mid-session, and everything that depends on fx.

        The camera may not honour the requested resolution, so ``TrackingWorker``
        re-resolves intrinsics from the first real frame. That used to update the
        iris estimator only: solvePnP kept solving against the STALE K forever,
        which is the one matrix that decides depth and lateral direction.

        Three things depend on fx and are all handled here, because updating one
        without the others is how this stays broken:

        - ``self._intrinsics`` — what solvePnP is actually given.
        - ``face_scale`` — tvec.z scales with fx, so the caller must re-resolve
          it (``resolve_face_scale``) and pass the result.
        - ``_reference_depth_raw`` — the legacy baseline. It is a raw tvec.z
          captured under the OLD fx, and the legacy formula divides by it, so an
          fx change invalidates it outright. Cleared, not rescaled: re-baselining
          on the next frame is exact, rescaling would inherit the same ~1%
          residual documented in resolve_face_scale for no reason.
        """
        old_fx = float(self._intrinsics.fx)
        self._intrinsics = intrinsics
        new_fx = float(intrinsics.fx)

        self._face_scale = float(face_scale) if face_scale is not None else None
        if self._face_scale is not None:
            self._model = FACE_MODEL_MM * self._face_scale
        else:
            self._model = FACE_MODEL_MM.copy()

        if abs(new_fx - old_fx) > 1e-6 and self._reference_depth_raw is not None:
            logger.info(
                "fx changed %.1f -> %.1f — clearing the monocular depth baseline "
                "(%.1f mm); it will re-capture on the next frame",
                old_fx, new_fx, self._reference_depth_raw,
            )
            self._reference_depth_raw = None

        logger.info(
            "Intrinsics updated: %dx%d fx=%.1f fy=%.1f — Z path: %s",
            intrinsics.width, intrinsics.height, new_fx, intrinsics.fy,
            "CALIBRATED (face_scale=%.4f)" % self._face_scale
            if self._face_scale is not None
            else "LEGACY (_monocular_scale)",
        )

    @property
    def garbage_solve_count(self) -> int:
        """Số nghiệm có ``tz`` dương nhưng khoảng cách ngoài ``DIST_SANE_MM``."""
        return self._garbage_solves

    def _count_garbage_solve(self, translation_cv: npt.NDArray[np.float64]) -> None:
        """
        Đếm nghiệm rác. CHỈ QUAN SÁT — không đổi pose, không chặn khung.

        Lớp này lọt qua cổng ``tz <= 0`` vì độ sâu vẫn dương, rồi bị bộ clamp
        trong tracking_worker kéo về dải hợp lệ nên biến mất không dấu vết. Dự
        án anh em từng thấy một khung ``dist = 451.488.954 mm`` lọt qua mà
        **cơ chế sinh ra nó tới giờ vẫn chưa giải thích được**.

        Không chặn ở đây là có chủ đích: trên toàn bộ dữ liệu có được, SQPNP
        sinh 0/1200 và 0/451 nghiệm rác, nên chưa có gì để chặn. Đếm thì rẻ và
        biến lần hỏng sau thành dữ liệu; chặn thì là trả phí cho một lớp lỗi
        chưa từng quan sát được — đúng lý do cổng nhất quán hình học bị bỏ
        (xem §4.6 trong docs/document.md).
        """
        distance = float(np.linalg.norm(translation_cv))
        if DIST_SANE_MM[0] <= distance <= DIST_SANE_MM[1]:
            return

        self._garbage_solves += 1
        if self._garbage_solves == 1 or self._garbage_solves % GARBAGE_LOG_EVERY == 0:
            logger.warning(
                "Garbage solve #%d: tz=%.1f mm positive but |t|=%.1f mm is outside "
                "the sane range %s — pose passed through unchanged, the clamp in "
                "tracking_worker will bound it",
                self._garbage_solves, float(translation_cv[2]), distance,
                DIST_SANE_MM,
            )

    def _ensure_depth_baseline(self, depth_raw: float) -> None:
        """Capture solvePnP depth on first frame to fix monocular scale ambiguity."""
        if self._reference_depth_raw is None and depth_raw > 1e-3:
            self._reference_depth_raw = depth_raw
            logger.info(
                "Monocular depth baseline: %.1f mm (maps to %.1f mm viewing distance)",
                depth_raw,
                self._reference_distance,
            )

    def _monocular_scale(self) -> float:
        """
        Uniform scale from the first-frame depth baseline.

        Using per-frame ``reference / depth`` cancels all Z variation because
        ``abs(tvec_z * reference / tvec_z) == reference`` every frame.
        """
        if self._reference_depth_raw is None or self._reference_depth_raw < 1e-3:
            return 1.0
        return self._reference_distance / self._reference_depth_raw


def _opencv_rot_to_hcpd(rotation: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """
    Convert OpenCV rotation to HCPD/OpenGL-compatible 4x4 matrix.

    Same basis change as the translation: ``R_world = M @ R_cv @ M.T``. This
    used to apply diag(1,-1,-1) while the translation applied diag(1,-1,1),
    so the two disagreed. Rotation only feeds the F1 overlay here, which is why
    the mismatch went unnoticed for so long.
    """
    converted = R_CV_TO_HCPD @ rotation @ R_CV_TO_HCPD.T
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = converted
    return matrix


def _opencv_to_hcpd_translation(
    tvec: npt.NDArray[np.float64],
    webcam_offset: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    Convert solvePnP camera-space tvec to screen-centre world space (mm).

    Uses :data:`R_CV_TO_HCPD`; see that constant for the derivation.

    The previous form was ``[tx, -ty, abs(tz)]``. Two defects: it left X
    unnegated, which inverted parallax, and ``abs`` silently rectified a
    negative depth instead of rejecting it. Measured over 1077 frames of a real
    session, ``tz <= 0`` never occurred, so the abs was covering nothing;
    ``estimate`` now treats a non-positive depth as a failed solve.
    """
    hcpd = R_CV_TO_HCPD @ np.asarray(tvec, dtype=np.float64).reshape(3)
    return hcpd + webcam_offset


def _euler_from_rotation(rotation: npt.NDArray[np.float64]) -> tuple[float, float, float]:
    """
    Extract pitch, yaw, roll in degrees from a 3x3 rotation matrix.

    Pass the **HCPD** matrix (``_opencv_rot_to_hcpd``), not the raw OpenCV one:
    the caller's quaternion is in HCPD, and feeding this two different bases is
    what made the F1 overlay disagree with the pose it was labelling.

    Decomposition is intrinsic Z-Y-X, i.e. ``R = Rz(roll) @ Ry(yaw) @ Rx(pitch)``
    about the HCPD axes (X right, Y up, Z toward the viewer). The gimbal-lock
    branch (``sy < 1e-6``, i.e. yaw at +-90 deg) pins roll to 0 and folds the
    whole remaining rotation into pitch — that is the standard degenerate
    convention, not a recovery.

    DISPLAY ONLY. Kooima builds its view matrix from the screen basis and the
    eye *position*; head orientation never enters the projection. The absolute
    sign of each angle is inherited from whatever axes ``FACE_MODEL_MM`` is
    drawn in, which has never been verified against a physical protractor --
    only the *consistency* with ``rotation`` is guaranteed here.
    """
    sy = math.sqrt(float(rotation[0, 0] ** 2 + rotation[1, 0] ** 2))
    singular = sy < 1e-6
    if not singular:
        pitch = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        pitch = math.atan2(float(-rotation[1, 2]), float(rotation[1, 1]))
        yaw = math.atan2(float(-rotation[2, 0]), sy)
        roll = 0.0
    return math.degrees(pitch), math.degrees(yaw), math.degrees(roll)
