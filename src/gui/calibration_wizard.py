"""Interactive screen and webcam calibration wizard."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from src.config import AppConfig, save_config
from src.config.schema import ScreenSettings, WebcamMountSettings
from src.pose.calibration import (
    CALIBRATION_FRAMES,
    CALIBRATION_MAX_RMSE_PX,
    CALIBRATION_MIN_PRESENCE,
    DEFAULT_CALIBRATION_DISTANCE_MM,
    FaceCalibration,
    FaceCalibrationError,
    compute_face_scale,
    load_face_calibration,
    resolve_intrinsics,
    save_face_calibration,
)

logger = logging.getLogger(__name__)

WINDOW_NAME = "HCPD Calibration Wizard"

REVIEW_STEP = 4


class CalibrationWizard:
    """
    Step-by-step wizard for screen, webcam mount, and face-scale calibration.

    Steps:
    1. Screen physical size (mm) and resolution (px)
    2. Monitor selection index
    3. Webcam offset relative to screen center (mm)
    4. Per-user face scale (live capture at a measured distance)
    5. Review and save to user.json
    """

    def __init__(self, config: AppConfig, output_path: Path) -> None:
        self._config = config
        self._output_path = output_path
        self._screen = ScreenSettings(**config.screen.__dict__)
        self._mount = WebcamMountSettings(**config.webcam_mount.__dict__)
        self._step = 0
        self._field_index = 0
        self._fields: list[tuple[str, str, float | int]] = []
        # Face-scale state. An existing calibration is loaded so that saving
        # from this wizard never silently drops it.
        self._face_calibration: FaceCalibration | None = load_face_calibration(
            output_path
        )
        self._calibration_distance_mm: float = (
            self._face_calibration.distance_mm
            if self._face_calibration is not None
            else DEFAULT_CALIBRATION_DISTANCE_MM
        )
        self._calibration_error: str | None = None

    def run(self) -> AppConfig:
        """Run the wizard and persist results."""
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        self._refresh_fields()

        while self._step <= REVIEW_STEP:
            canvas = self._draw_step()
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(0) & 0xFF

            if key in (27, ord("q")):
                break
            if key in (13, ord("\r"), ord("n")):
                self._next_step()
            elif key == ord("p"):
                self._previous_step()
            elif key in (ord("+"), ord("=")):
                self._adjust_field(+1)
            elif key in (ord("-"), ord("_")):
                self._adjust_field(-1)
            elif key == ord("c") and self._step == 3:
                self._capture_face_scale()
            elif key == ord("s") and self._step == REVIEW_STEP:
                self._save()

        cv2.destroyAllWindows()
        return self._build_config()

    def _capture_face_scale(self) -> None:
        """
        Capture N frames at the measured distance and derive ``face_scale``.

        Uses the UNCALIBRATED millimetre face model on purpose: face_scale is
        exactly the correction between that generic model and this user.
        """
        from src.camera.capture import CameraCapture, CaptureConfig
        from src.pose.head_pose import _reprojection_rmse_px
        from src.pose.solve_pnp import solve_head_pose
        from src.tracking import FaceTracker
        from src.tracking.landmarks import FACE_MODEL_MM

        self._calibration_error = None
        cam = self._config.camera
        capture = CameraCapture(
            CaptureConfig(
                device_id=cam.device_id,
                width=cam.width,
                height=cam.height,
                fps=cam.fps,
                buffer_size=cam.buffer_size,
                fourcc=cam.fourcc,
            )
        )
        tracker = FaceTracker(
            min_detection_confidence=self._config.tracking.min_detection_confidence,
            min_tracking_confidence=self._config.tracking.min_tracking_confidence,
            model_path=self._config.tracking.model_path,
        )

        tz_samples: list[float] = []
        rmse_seen: list[float] = []
        rejected = 0
        try:
            if not tracker.is_available:
                raise FaceCalibrationError(
                    "MediaPipe face tracking unavailable — cannot calibrate."
                )
            capture.open()
            # Must match what TrackingWorker resolves, or the face_scale this
            # run produces is measured against an fx the app never uses.
            intrinsics = resolve_intrinsics(cam.width, cam.height)

            for index in range(CALIBRATION_FRAMES):
                packet = capture.read()
                if packet is None:
                    rejected += 1
                    continue
                if packet.width != intrinsics.width:
                    intrinsics = resolve_intrinsics(packet.width, packet.height)

                result = tracker.process(packet.image)
                if not result.detected or result.landmarks is None:
                    rejected += 1
                elif result.confidence < CALIBRATION_MIN_PRESENCE:
                    rejected += 1
                else:
                    ok, rvec, tvec = solve_head_pose(
                        FACE_MODEL_MM,
                        result.landmarks.image_points,
                        intrinsics,
                        fast=True,
                    )
                    rmse = (
                        _reprojection_rmse_px(
                            FACE_MODEL_MM,
                            result.landmarks.image_points,
                            intrinsics,
                            rvec,
                            tvec,
                        )
                        if ok
                        else float("inf")
                    )
                    if ok:
                        rmse_seen.append(rmse)
                    if ok and rmse <= CALIBRATION_MAX_RMSE_PX:
                        tz_samples.append(abs(float(tvec[2])))
                    else:
                        rejected += 1

                cv2.imshow(
                    WINDOW_NAME,
                    self._draw_capture_progress(index + 1, len(tz_samples), rejected),
                )
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    raise FaceCalibrationError("Calibration cancelled by user.")

            self._face_calibration = compute_face_scale(
                tz_samples,
                self._calibration_distance_mm,
                intrinsics_fx=intrinsics.fx,
            )
            logger.info(
                "Face scale calibrated: %.4f (%d/%d valid frames, "
                "median|tz|=%.1f mm, std=%.1f mm)",
                self._face_calibration.face_scale,
                self._face_calibration.valid_frames,
                CALIBRATION_FRAMES,
                self._face_calibration.tz_median_mm,
                self._face_calibration.tz_std_mm,
            )
        except FaceCalibrationError as exc:
            # Never fall back to a default — surface the failure and keep any
            # previously stored calibration untouched.  Include the observed
            # reprojection error so a dead-end failure is actionable: if the
            # median RMSE sits above CALIBRATION_MAX_RMSE_PX, the generic face
            # model simply does not fit this face that closely and the
            # threshold, not the user, is what needs adjusting.
            detail = str(exc)
            if rmse_seen:
                arr = np.asarray(rmse_seen, dtype=np.float64)
                detail += (
                    f" | reproj_rmse_px over {arr.size} detected frames: "
                    f"min={arr.min():.1f} median={float(np.median(arr)):.1f} "
                    f"max={arr.max():.1f} vs threshold "
                    f"{CALIBRATION_MAX_RMSE_PX:.1f}"
                )
            self._calibration_error = detail
            logger.error("Face scale calibration failed: %s", detail)
        except Exception as exc:  # camera / tracker failures
            self._calibration_error = f"Capture failed: {exc}"
            logger.exception("Face scale calibration aborted")
        finally:
            capture.close()
            tracker.close()

    def _draw_capture_progress(
        self, frame_index: int, valid: int, rejected: int
    ) -> np.ndarray:
        canvas = np.zeros((480, 760, 3), dtype=np.uint8)
        self._put_text(canvas, "Capturing face scale...", (20, 40), 0.8, (0, 255, 180))
        lines = [
            f"Hold still at {self._calibration_distance_mm:.0f} mm from the screen.",
            "",
            f"Frame     : {frame_index}/{CALIBRATION_FRAMES}",
            f"Accepted  : {valid}",
            f"Rejected  : {rejected}",
            "",
            "Q/Esc cancel",
        ]
        y = 100
        for line in lines:
            self._put_text(canvas, line, (30, y), 0.6, (220, 220, 220))
            y += 34
        return canvas

    def _refresh_fields(self) -> None:
        if self._step == 0:
            self._fields = [
                ("width_mm", "Screen width (mm)", self._screen.width_mm),
                ("height_mm", "Screen height (mm)", self._screen.height_mm),
                ("width_px", "Screen width (px)", self._screen.width_px),
                ("height_px", "Screen height (px)", self._screen.height_px),
            ]
        elif self._step == 1:
            self._fields = [
                ("monitor_index", "Monitor index", self._screen.monitor_index),
            ]
        elif self._step == 2:
            self._fields = [
                ("offset_x_mm", "Webcam X offset (mm)", self._mount.offset_x_mm),
                ("offset_y_mm", "Webcam Y offset (mm)", self._mount.offset_y_mm),
                ("offset_z_mm", "Webcam Z offset (mm)", self._mount.offset_z_mm),
            ]
        elif self._step == 3:
            self._fields = [
                (
                    "calibration_distance_mm",
                    "Measured eye->screen distance (mm)",
                    self._calibration_distance_mm,
                ),
            ]
        else:
            self._fields = []

        self._field_index = min(self._field_index, max(0, len(self._fields) - 1))

    def _draw_step(self) -> np.ndarray:
        canvas = np.zeros((480, 760, 3), dtype=np.uint8)
        titles = [
            "Step 1/5 — Screen size",
            "Step 2/5 — Monitor selection",
            "Step 3/5 — Webcam position",
            "Step 4/5 — Face scale calibration",
            "Step 5/5 — Review & save",
        ]
        self._put_text(canvas, titles[self._step], (20, 40), 0.8, (0, 255, 180))

        if self._step == 3:
            self._draw_face_scale_step(canvas)
        elif self._step < 3:
            y = 90
            for index, (key, label, value) in enumerate(self._fields):
                prefix = ">" if index == self._field_index else " "
                color = (0, 220, 255) if index == self._field_index else (200, 200, 200)
                if isinstance(value, float):
                    line = f"{prefix} {label}: {value:.1f}"
                else:
                    line = f"{prefix} {label}: {value}"
                self._put_text(canvas, line, (30, y), 0.65, color)
                y += 36

            help_lines = [
                "+/- adjust value",
                "N/Enter next step",
                "P previous step",
                "Q quit",
            ]
            y = 360
            for line in help_lines:
                self._put_text(canvas, line, (30, y), 0.55, (140, 140, 140))
                y += 28
        else:
            summary = self._build_config()
            lines = [
                f"Screen: {summary.screen.width_mm:.1f} x {summary.screen.height_mm:.1f} mm",
                f"Pixels: {summary.screen.width_px} x {summary.screen.height_px}",
                f"Monitor: {summary.screen.monitor_index}",
                (
                    "Webcam offset (mm): "
                    f"{summary.webcam_mount.offset_x_mm:.1f}, "
                    f"{summary.webcam_mount.offset_y_mm:.1f}, "
                    f"{summary.webcam_mount.offset_z_mm:.1f}"
                ),
                (
                    f"Face scale: {self._face_calibration.face_scale:.4f} "
                    f"(D={self._face_calibration.distance_mm:.0f} mm, "
                    f"{self._face_calibration.valid_frames} frames)"
                    if self._face_calibration is not None
                    else "Face scale: NOT CALIBRATED — legacy Z path"
                ),
                "",
                "Press S to save user.json",
                "Press N to finish without saving",
            ]
            y = 90
            for line in lines:
                self._put_text(canvas, line, (30, y), 0.6, (220, 220, 220))
                y += 34

        return canvas

    def _draw_face_scale_step(self, canvas: np.ndarray) -> None:
        """Render the face-scale step: distance field, status, help."""
        y = 90
        for index, (_, label, value) in enumerate(self._fields):
            prefix = ">" if index == self._field_index else " "
            color = (0, 220, 255) if index == self._field_index else (200, 200, 200)
            self._put_text(canvas, f"{prefix} {label}: {value:.0f}", (30, y), 0.65, color)
            y += 40

        if self._calibration_error is not None:
            self._put_text(canvas, "FAILED — nothing saved:", (30, y), 0.58, (80, 120, 255))
            y += 28
            for chunk in _wrap(self._calibration_error, 68):
                self._put_text(canvas, chunk, (30, y), 0.5, (120, 160, 255))
                y += 24
        elif self._face_calibration is not None:
            cal = self._face_calibration
            self._put_text(
                canvas,
                f"face_scale = {cal.face_scale:.4f}",
                (30, y),
                0.65,
                (0, 255, 180),
            )
            y += 30
            self._put_text(
                canvas,
                f"median|tz|={cal.tz_median_mm:.1f} mm  std={cal.tz_std_mm:.1f} mm  "
                f"frames={cal.valid_frames}",
                (30, y),
                0.5,
                (200, 200, 200),
            )
            y += 30
        else:
            self._put_text(
                canvas, "Not calibrated — legacy Z path in use.", (30, y), 0.55, (200, 200, 200)
            )
            y += 30

        help_lines = [
            "Measure eyes -> screen with a tape, set the value above.",
            "+/- adjust distance (10 mm)",
            f"C run capture ({CALIBRATION_FRAMES} frames)",
            "N/Enter next step   P previous   Q quit",
        ]
        y = max(y + 16, 350)
        for line in help_lines:
            self._put_text(canvas, line, (30, y), 0.52, (140, 140, 140))
            y += 26

    def _adjust_field(self, direction: int) -> None:
        if not self._fields:
            return

        key, _, value = self._fields[self._field_index]
        if key == "calibration_distance_mm":
            step = 10.0
            new_value = float(value) + direction * step
        elif isinstance(value, float):
            step = 1.0 if "mm" in key else 10.0
            new_value = float(value) + direction * step
        else:
            new_value = int(value) + direction

        self._apply_field(key, new_value)
        self._refresh_fields()

    def _apply_field(self, key: str, value: float | int) -> None:
        if key == "calibration_distance_mm":
            self._calibration_distance_mm = max(100.0, float(value))
        elif hasattr(self._screen, key):
            setattr(self._screen, key, type(getattr(self._screen, key))(value))
        elif hasattr(self._mount, key):
            setattr(self._mount, key, type(getattr(self._mount, key))(value))

    def _next_step(self) -> None:
        if self._step < REVIEW_STEP:
            self._step += 1
            self._field_index = 0
            self._refresh_fields()

    def _previous_step(self) -> None:
        if self._step > 0:
            self._step -= 1
            self._field_index = 0
            self._refresh_fields()

    def _build_config(self) -> AppConfig:
        config = AppConfig.from_dict(self._config.to_dict())
        config.screen = self._screen
        config.webcam_mount = self._mount
        return config

    def _save(self) -> None:
        config = self._build_config()
        save_config(config, self._output_path)
        # save_config writes AppConfig.to_dict() wholesale, which drops the
        # face_calibration block. Re-merge it afterwards.
        if self._face_calibration is not None:
            save_face_calibration(self._face_calibration, self._output_path)
        logger.info("Calibration saved to %s", self._output_path)
        self._step += 1

    @staticmethod
    def _put_text(
        image: np.ndarray,
        text: str,
        origin: tuple[int, int],
        scale: float,
        color: tuple[int, int, int],
    ) -> None:
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            text,
            origin,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            color,
            1,
            cv2.LINE_AA,
        )


def _wrap(text: str, width: int) -> list[str]:
    """Greedy word wrap for multi-line error text on the canvas."""
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        if len(candidate) > width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def run_calibration_wizard(
    config: AppConfig,
    output_path: Path,
) -> AppConfig:
    """Launch calibration wizard and return updated configuration."""
    wizard = CalibrationWizard(config, output_path)
    return wizard.run()
