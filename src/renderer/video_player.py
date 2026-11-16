"""OpenCV-backed video decoder for the in-scene video plane."""

from __future__ import annotations

import logging
import os
from pathlib import Path

from src.content.cv_io import open_video_capture
import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


class VideoPlayer:
    """
    Decode video frames for GPU upload.

    Falls back to a procedural test pattern when the file is missing.
    """

    def __init__(self, path: Path, loop: bool = True) -> None:
        self._loop = loop
        self._path = path
        self._cap: cv2.VideoCapture | None = None
        self._procedural = False
        self._width = 1280
        self._height = 720
        self._fps = 30.0
        self._aspect = 16.0 / 9.0
        self._last_rgb: npt.NDArray[np.uint8] | None = None
        self._accum_s = 0.0
        self._frame_duration_s = 1.0 / 30.0
        self._proc_t = 0.0
        # PB-059 (3): đếm khung đã giải mã, chỉ để ĐO, mặc định TẮT. Cùng lối
        # móc sau biến môi trường như PB-058 — không đặt biến thì không có thêm
        # một dòng log hay một phép tính nào.
        self._stats = os.environ.get("HCPD_VIDEO_STATS") == "1"
        self._decoded = 0
        self._open(path)

    @property
    def decoded_frames(self) -> int:
        """Số khung đã giải mã kể từ khi mở file (chỉ dùng cho script đo)."""
        return self._decoded

    @property
    def is_procedural(self) -> bool:
        return self._procedural

    def replace(self, path: Path, loop: bool | None = None) -> None:
        """Close current decoder and open a new video file."""
        self.close()
        self._procedural = False
        self._last_rgb = None
        self._accum_s = 0.0
        self._proc_t = 0.0
        if loop is not None:
            self._loop = loop
        self._path = path
        self._open(path)
        if self._procedural:
            raise ValueError(f"Failed to open video: {path.name}")

    @property
    def aspect_ratio(self) -> float:
        return self._aspect

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None

    def prime_first_frame(self) -> npt.NDArray[np.uint8] | None:
        """Decode and return the first displayable frame (eager, no dt required)."""
        if self._procedural:
            self._last_rgb = self._procedural_frame()
            return self._last_rgb
        if self._last_rgb is not None:
            return self._last_rgb
        frame = self._read_next()
        if frame is not None:
            self._last_rgb = frame
            self._accum_s = 0.0
        else:
            logger.error("Video decode failed — no frames in %s", self._path)
        return self._last_rgb

    def tick(self, dt_s: float) -> npt.NDArray[np.uint8] | None:
        """Advance playback and return the latest RGB frame (or None)."""
        if self._procedural:
            self._proc_t += dt_s
            self._last_rgb = self._procedural_frame()
            return self._last_rgb

        if self._last_rgb is None:
            return self.prime_first_frame()

        self._accum_s += dt_s
        if self._accum_s < self._frame_duration_s:
            return self._last_rgb

        while self._accum_s >= self._frame_duration_s:
            self._accum_s -= self._frame_duration_s
            frame = self._read_next()
            if frame is None:
                return self._last_rgb
            self._last_rgb = frame
            if self._stats:
                self._decoded += 1
                if self._decoded % 300 == 0:
                    logger.info("Video stats: %d frames decoded from %s",
                                self._decoded, self._path.name)

        return self._last_rgb

    def _open(self, path: Path) -> None:
        if not path.exists():
            logger.warning(
                "Video not found at %s — using procedural test pattern", path
            )
            self._procedural = True
            return

        cap = open_video_capture(path)
        if cap is None:
            logger.warning("Failed to open %s — using procedural test pattern", path)
            self._procedural = True
            return

        self._cap = cap
        self._width = max(1, int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self._height = max(1, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self._fps = fps if fps > 1.0 else 30.0
        self._aspect = self._width / float(self._height)
        self._frame_duration_s = 1.0 / self._fps
        logger.info(
            "Video opened: %s (%dx%d @ %.1f fps)",
            path.name, self._width, self._height, self._fps,
        )
        if self.prime_first_frame() is None:
            logger.error(
                "Video opened but first frame decode failed: %s", path.resolve()
            )
            self.close()
            self._procedural = True

    def _read_next(self) -> npt.NDArray[np.uint8] | None:
        if self._cap is None:
            return None
        ok, bgr = self._cap.read()
        if ok and bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if not self._loop:
            return self._last_rgb

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, bgr = self._cap.read()
        if ok and bgr is not None:
            return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return self._last_rgb

    def _procedural_frame(self) -> npt.NDArray[np.uint8]:
        """Animated gradient bars when no video file is available."""
        w, h = self._width, self._height
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        t = self._proc_t
        for y in range(h):
            frame[y, :, 0] = np.clip(40 + 30 * np.sin(t + y * 0.02), 0, 255)
            frame[y, :, 1] = np.clip(35 + 25 * np.sin(t * 1.3), 0, 255)
            frame[y, :, 2] = np.clip(45 + 20 * np.sin(t * 0.7 + y * 0.01), 0, 255)
        bar_x = int((0.5 + 0.4 * np.sin(t * 2.0)) * (w - 80))
        cv2.rectangle(frame, (bar_x, h // 4), (bar_x + 80, 3 * h // 4), (220, 220, 230), -1)
        cv2.putText(
            frame, "Place video at assets/video/demo.mp4",
            (20, h - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 210), 1, cv2.LINE_AA,
        )
        return frame
