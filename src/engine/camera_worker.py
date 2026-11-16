"""Background camera capture worker (no display)."""

from __future__ import annotations

import logging
import threading
import time

from src.camera import AsyncCameraCapture, CaptureConfig

logger = logging.getLogger(__name__)


class CameraWorker:
    """
    Thread 1: continuous webcam capture.

    Frames are consumed internally by the tracking worker in Phase 3.
    The render thread never waits on capture.
    """

    def __init__(self, config: CaptureConfig, queue_size: int = 2) -> None:
        self._capture = AsyncCameraCapture(config, queue_size=queue_size)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._latest_latency_ms = 0.0

    @property
    def capture_latency_ms(self) -> float:
        return self._latest_latency_ms

    def start(self) -> None:
        """Start capture thread."""
        if self._thread is not None:
            return
        self._stop.clear()
        self._capture.start()
        self._thread = threading.Thread(
            target=self._loop,
            name="CameraWorker",
            daemon=True,
        )
        self._thread.start()
        logger.info("Camera worker started")

    def stop(self) -> None:
        """Stop capture thread."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        self._capture.stop()

    def pop_frame(self):
        """Return latest frame for tracking (Phase 3)."""
        return self._capture.pop_latest()

    def peek_frame(self):
        """Return latest frame without removing it (for preview HUD)."""
        return self._capture.get_latest()

    def _loop(self) -> None:
        logger.debug("CameraWorker loop entered")
        try:
            while not self._stop.is_set():
                stats = self._capture.stats()
                if stats.timing is not None:
                    self._latest_latency_ms = stats.timing.last_capture_latency_ms
                time.sleep(0.001)
        except Exception:
            logger.exception("CameraWorker loop crashed")
        finally:
            logger.debug("CameraWorker loop exited")
