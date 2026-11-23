"""Synchronous webcam capture with configurable resolution and timing."""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from src.camera.timing import FrameTimer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class CaptureConfig:
    """Webcam capture settings."""

    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    buffer_size: int = 1
    fourcc: str = "MJPG"

    def to_dict(self) -> dict[str, int | str]:
        """Serialize configuration for JSON persistence."""
        return {
            "device_id": self.device_id,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
            "buffer_size": self.buffer_size,
            "fourcc": self.fourcc,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CaptureConfig:
        """Build configuration from a dictionary."""
        return cls(
            device_id=int(data.get("device_id", 0)),
            width=int(data.get("width", 1280)),
            height=int(data.get("height", 720)),
            fps=int(data.get("fps", 60)),
            buffer_size=int(data.get("buffer_size", 1)),
            fourcc=str(data.get("fourcc", "MJPG")),
        )


@dataclass(frozen=True)
class FramePacket:
    """A captured frame with timing metadata."""

    image: npt.NDArray[np.uint8]
    frame_index: int
    timestamp: float
    capture_latency_ms: float

    @property
    def height(self) -> int:
        return int(self.image.shape[0])

    @property
    def width(self) -> int:
        return int(self.image.shape[1])

    @property
    def channels(self) -> int:
        return int(self.image.shape[2]) if self.image.ndim == 3 else 1


class CameraCapture:
    """Blocking webcam capture backed by OpenCV VideoCapture."""

    def __init__(self, config: CaptureConfig | None = None) -> None:
        self._config = config or CaptureConfig()
        self._cap: cv2.VideoCapture | None = None
        self._frame_index = 0
        self._timer = FrameTimer()
        self._is_open = False

    @property
    def config(self) -> CaptureConfig:
        return self._config

    @property
    def is_open(self) -> bool:
        return self._is_open and self._cap is not None and self._cap.isOpened()

    @property
    def frame_index(self) -> int:
        return self._frame_index

    @property
    def timer(self) -> FrameTimer:
        return self._timer

    def open(self) -> None:
        """Open the webcam and apply capture properties."""
        if self.is_open:
            return

        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(self._config.device_id, backend)
        if not cap.isOpened():
            cap.release()
            cap = cv2.VideoCapture(self._config.device_id)
        if not cap.isOpened():
            raise RuntimeError(
                f"Failed to open camera device {self._config.device_id}"
            )

        fourcc = cv2.VideoWriter_fourcc(*self._config.fourcc)
        if sys.platform != "win32":
            # V4L2: size and fps are matched against the pixel format that is
            # active when they are set. Setting the format last leaves the
            # camera on raw YUYV, which tops out near 10 fps at 1280x720.
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, float(self._config.buffer_size))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, float(self._config.width))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, float(self._config.height))
        cap.set(cv2.CAP_PROP_FPS, float(self._config.fps))

        if sys.platform == "win32":
            cap.set(cv2.CAP_PROP_FOURCC, fourcc)

        self._cap = cap
        self._is_open = True
        self._frame_index = 0
        self._timer.reset()

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        logger.info(
            "Camera opened: device=%d resolution=%dx%d fps=%.1f",
            self._config.device_id,
            actual_w,
            actual_h,
            actual_fps,
        )

    def close(self) -> None:
        """Release the webcam resource."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._is_open = False

    def read(self) -> FramePacket | None:
        """Capture one frame. Returns None if acquisition fails."""
        if not self.is_open or self._cap is None:
            raise RuntimeError("Camera is not open. Call open() first.")

        start = FrameTimer.now()
        ok, frame = self._cap.read()
        capture_latency_ms = self._timer.elapsed_ms(start)

        if not ok or frame is None:
            logger.warning("Frame read failed at index %d", self._frame_index)
            return None

        timestamp = FrameTimer.now()
        self._frame_index += 1
        self._timer.tick(capture_latency_ms=capture_latency_ms)

        return FramePacket(
            image=frame,
            frame_index=self._frame_index,
            timestamp=timestamp,
            capture_latency_ms=capture_latency_ms,
        )

    def grab_latest(self, max_discard: int = 4) -> FramePacket | None:
        """Drain stale buffered frames and return the most recent one."""
        latest: FramePacket | None = None
        for _ in range(max_discard + 1):
            packet = self.read()
            if packet is None:
                break
            latest = packet
        return latest

    def get_backend_info(self) -> dict[str, float | int | str]:
        """Query actual capture properties from the driver."""
        if not self.is_open or self._cap is None:
            return {}

        return {
            "device_id": self._config.device_id,
            "width": int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(self._cap.get(cv2.CAP_PROP_FPS)),
            "backend": str(self._cap.getBackendName()),
            "buffer_size": int(self._cap.get(cv2.CAP_PROP_BUFFERSIZE)),
        }

    def __enter__(self) -> CameraCapture:
        self.open()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.close()
