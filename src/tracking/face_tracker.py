"""MediaPipe face landmark tracking with iris (Tasks API + Solutions fallback)."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

from src.tracking.landmarks import (
    LEFT_IRIS_CENTER,
    PNP_LANDMARKS,
    RIGHT_IRIS_CENTER,
    LandmarkSet,
    normalized_to_pixels,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FaceTrackingResult:
    """Output of one face tracking frame."""

    detected: bool
    landmarks: LandmarkSet | None
    confidence: float
    image_width: int
    image_height: int


class FaceTracker:
    """
    Face landmark tracker with iris support.

    Prefers the modern MediaPipe Tasks API (``FaceLandmarker``) and falls
    back to the legacy Solutions API when available. Webcam frames are
    processed internally — never displayed.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        max_num_faces: int = 1,
        model_path: str | None = None,
    ) -> None:
        self._min_detection = min_detection_confidence
        self._min_tracking = min_tracking_confidence
        self._max_faces = max_num_faces
        self._model_path = model_path

        self._backend = "none"
        self._landmarker = None  # Tasks API
        self._face_mesh = None  # Solutions API
        self._available = False
        self._last_timestamp_ms = 0

        if not self._init_tasks_api():
            self._init_solutions_api()

    @property
    def is_available(self) -> bool:
        return self._available

    @property
    def backend(self) -> str:
        return self._backend

    def close(self) -> None:
        """Release MediaPipe resources."""
        if self._landmarker is not None:
            self._landmarker.close()
            self._landmarker = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None

    def process(
        self,
        bgr_image: npt.NDArray[np.uint8],
        process_scale: float = 1.0,
    ) -> FaceTrackingResult:
        """Detect face landmarks and iris centers in a BGR frame."""
        height, width = bgr_image.shape[:2]
        if not self._available:
            return FaceTrackingResult(False, None, 0.0, width, height)

        scale = float(np.clip(process_scale, 0.25, 1.0))
        if scale < 0.99:
            proc = cv2.resize(
                bgr_image,
                (int(width * scale), int(height * scale)),
                interpolation=cv2.INTER_LINEAR,
            )
            proc_w, proc_h = proc.shape[1], proc.shape[0]
        else:
            proc = bgr_image
            proc_w, proc_h = width, height

        rgb = cv2.cvtColor(proc, cv2.COLOR_BGR2RGB)

        if self._backend == "tasks":
            all_pts, confidence = self._detect_tasks(rgb)
        else:
            all_pts, confidence = self._detect_solutions(rgb)

        if all_pts is None:
            return FaceTrackingResult(False, None, 0.0, width, height)

        pixel_landmarks = normalized_to_pixels(all_pts[:, :2], proc_w, proc_h)
        if scale < 0.99:
            pixel_landmarks /= scale

        landmarks = self._build_landmark_set(pixel_landmarks, all_pts)
        return FaceTrackingResult(True, landmarks, confidence, width, height)

    def _build_landmark_set(
        self,
        pixel_landmarks: npt.NDArray[np.float64],
        all_pts: npt.NDArray[np.float64],
    ) -> LandmarkSet:
        pnp_points = pixel_landmarks[list(PNP_LANDMARKS), :2]
        count = pixel_landmarks.shape[0]
        if count > RIGHT_IRIS_CENTER:
            left_iris = pixel_landmarks[LEFT_IRIS_CENTER, :2]
            right_iris = pixel_landmarks[RIGHT_IRIS_CENTER, :2]
        else:
            left_iris = pixel_landmarks[33, :2]
            right_iris = pixel_landmarks[263, :2]
        return LandmarkSet(
            image_points=pnp_points,
            left_iris=left_iris,
            right_iris=right_iris,
            all_landmarks=all_pts,
        )

    def _detect_tasks(
        self,
        rgb: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float64] | None, float]:
        """Return (landmarks_array, presence_score). Score is in [0, 1]."""
        import mediapipe as mp

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        timestamp_ms = int(time.perf_counter() * 1000.0)
        if timestamp_ms <= self._last_timestamp_ms:
            timestamp_ms = self._last_timestamp_ms + 1
        self._last_timestamp_ms = timestamp_ms

        result = self._landmarker.detect_for_video(mp_image, timestamp_ms)
        if not result.face_landmarks:
            return None, 0.0
        face = result.face_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in face], dtype=np.float64)
        # Use mean presence of the 6 solvePnP landmarks as a confidence proxy.
        presence_scores = [
            lm.presence for lm in face
            if hasattr(lm, "presence") and lm.presence is not None
        ]
        confidence = float(np.mean(presence_scores)) if presence_scores else 1.0
        return pts, confidence

    def _detect_solutions(
        self,
        rgb: npt.NDArray[np.uint8],
    ) -> tuple[npt.NDArray[np.float64] | None, float]:
        """Return (landmarks_array, confidence). Solutions API has no per-landmark score."""
        rgb.flags.writeable = False
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            return None, 0.0
        face = result.multi_face_landmarks[0]
        pts = np.array([[lm.x, lm.y, lm.z] for lm in face.landmark], dtype=np.float64)
        return pts, 1.0

    def _init_tasks_api(self) -> bool:
        model_path = self._resolve_model_path()
        if model_path is None:
            logger.warning("Face landmark model not found — skipping Tasks API")
            return False

        try:
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision

            base_options = mp_python.BaseOptions(
                model_asset_path=str(model_path)
            )
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=vision.RunningMode.VIDEO,
                num_faces=self._max_faces,
                min_face_detection_confidence=self._min_detection,
                min_face_presence_confidence=self._min_tracking,
                min_tracking_confidence=self._min_tracking,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self._landmarker = vision.FaceLandmarker.create_from_options(options)
            self._last_timestamp_ms = 0
            self._backend = "tasks"
            self._available = True
            logger.info("MediaPipe Tasks FaceLandmarker initialized (%s)", model_path)
            return True
        except ImportError:
            logger.info("MediaPipe Tasks API unavailable — trying Solutions API")
            return False
        except Exception as exc:
            logger.error("Tasks API init failed: %s", exc)
            return False

    def _init_solutions_api(self) -> None:
        try:
            import mediapipe as mp

            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=self._max_faces,
                refine_landmarks=True,
                min_detection_confidence=self._min_detection,
                min_tracking_confidence=self._min_tracking,
            )
            self._backend = "solutions"
            self._available = True
            logger.info("MediaPipe Solutions FaceMesh initialized with iris")
        except (ImportError, AttributeError):
            logger.warning(
                "MediaPipe tracking unavailable (no Tasks model, no Solutions API). "
                "Install model or 'pip install mediapipe' with solutions support."
            )
        except Exception as exc:
            logger.error("Failed to initialize MediaPipe Solutions: %s", exc)

    def _resolve_model_path(self) -> Path | None:
        candidates: list[Path] = []
        if self._model_path:
            candidates.append(Path(self._model_path))
        project_root = Path(__file__).resolve().parents[2]
        candidates.append(project_root / "assets" / "models" / "face_landmarker.task")
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return None
