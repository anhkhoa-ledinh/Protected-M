"""Asynchronous webcam capture on a dedicated background thread."""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

from src.camera.capture import CameraCapture, CaptureConfig, FramePacket
from src.camera.timing import FrameTimer, TimingStats

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AsyncCaptureStats:
    """Runtime statistics for the async capture worker."""

    frames_produced: int = 0
    frames_dropped: int = 0
    queue_depth: int = 0
    is_running: bool = False
    timing: TimingStats | None = None

    def to_dict(self) -> dict[str, float | int | bool | None]:
        timing = self.timing.to_dict() if self.timing else None
        return {
            "frames_produced": self.frames_produced,
            "frames_dropped": self.frames_dropped,
            "queue_depth": self.queue_depth,
            "is_running": self.is_running,
            "timing": timing,
        }


class AsyncCameraCapture:
    """
    Non-blocking frame producer running capture on a background thread.

    Uses a bounded deque so the consumer always receives recent frames
    without blocking the render or tracking threads.
    """

    def __init__(
        self,
        config: CaptureConfig | None = None,
        queue_size: int = 2,
        on_frame: Callable[[FramePacket], None] | None = None,
    ) -> None:
        self._config = config or CaptureConfig()
        self._queue_size = max(1, queue_size)
        self._on_frame = on_frame

        self._camera = CameraCapture(self._config)
        self._queue: deque[FramePacket] = deque(maxlen=self._queue_size)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._frames_produced = 0
        self._frames_dropped = 0

    @property
    def config(self) -> CaptureConfig:
        return self._config

    @property
    def is_running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def start(self) -> None:
        """Open the camera and begin background capture."""
        if self.is_running:
            return

        self._camera.open()
        self._stop_event.clear()
        self._frames_produced = 0
        self._frames_dropped = 0

        self._thread = threading.Thread(
            target=self._capture_loop,
            name="AsyncCameraCapture",
            daemon=True,
        )
        self._thread.start()
        logger.info("Async capture started (queue_size=%d)", self._queue_size)

    def stop(self, timeout: float = 2.0) -> None:
        """Stop the capture thread and release the camera."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
        self._camera.close()

        with self._lock:
            self._queue.clear()

        logger.info("Async capture stopped")

    def get_latest(self) -> FramePacket | None:
        """Return the newest frame without blocking."""
        with self._lock:
            if not self._queue:
                return None
            return self._queue[-1]

    def pop_latest(self) -> FramePacket | None:
        """Remove and return the newest frame, discarding older queued frames."""
        with self._lock:
            if not self._queue:
                return None
            packet = self._queue.pop()
            self._queue.clear()
            return packet

    def wait_for_frame(self, timeout: float = 1.0) -> FramePacket | None:
        """Block until a frame arrives or timeout expires."""
        deadline = FrameTimer.now() + timeout
        while FrameTimer.now() < deadline:
            packet = self.pop_latest()
            if packet is not None:
                return packet
            time.sleep(0.001)
        return None

    def stats(self) -> AsyncCaptureStats:
        """Current producer statistics."""
        with self._lock:
            depth = len(self._queue)

        return AsyncCaptureStats(
            frames_produced=self._frames_produced,
            frames_dropped=self._frames_dropped,
            queue_depth=depth,
            is_running=self.is_running,
            timing=self._camera.timer.stats,
        )

    def _capture_loop(self) -> None:
        logger.debug("AsyncCameraCapture loop entered")
        try:
            while not self._stop_event.is_set():
                try:
                    packet = self._camera.read()
                except Exception:
                    logger.exception("Camera read failed; retrying")
                    time.sleep(0.05)
                    continue
                if packet is None:
                    time.sleep(0.001)
                    continue

                dropped = 0
                with self._lock:
                    if len(self._queue) == self._queue.maxlen:
                        dropped = len(self._queue)
                        self._queue.clear()
                    self._queue.append(packet)

                self._frames_produced += 1
                self._frames_dropped += dropped

                if self._on_frame is not None:
                    self._on_frame(packet)
        except Exception:
            logger.exception("AsyncCameraCapture loop crashed")
        finally:
            logger.debug("AsyncCameraCapture loop exited")

    def __enter__(self) -> AsyncCameraCapture:
        self.start()
        return self

    def __exit__(self, exc_type: object, exc_val: object, exc_tb: object) -> None:
        self.stop()
