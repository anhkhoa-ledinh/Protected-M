"""Background worker for monocular depth estimation."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import numpy as np
import numpy.typing as npt

from src.content.depth_estimator import DepthEstimator

logger = logging.getLogger(__name__)


class DepthWorker:
    """
    Async depth estimation — does not block the render loop.

    Depth is computed once per media path (image load or video first frame).
    TODO: per-frame video depth for true temporal 2.5D/3D (high CPU/GPU cost).
    """

    def __init__(self, model_name: str, backend: str = "torch") -> None:
        self._estimator = DepthEstimator(model_name, backend)
        self._thread: threading.Thread | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._pending: tuple[str, npt.NDArray[np.uint8]] | None = None
        self._ready: dict[str, npt.NDArray[np.float32]] = {}
        self._failed = False
        self._in_flight: str | None = None
        self._last_inference_ms: float = 0.0
        self._last_inference_key: str | None = None

    @property
    def failed(self) -> bool:
        return self._failed

    @property
    def last_inference_ms(self) -> float:
        return self._last_inference_ms

    def last_inference_ms_for(self, media_key: str) -> float:
        if self._last_inference_key == media_key:
            return self._last_inference_ms
        return 0.0

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, name="DepthWorker", daemon=True)
        self._thread.start()
        logger.info("Depth worker started (model=%s)", self._estimator._model_name)

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=3.0)
            self._thread = None

    def submit(self, media_key: str, rgb: npt.NDArray[np.uint8]) -> None:
        """Queue RGB frame for depth estimation if not already cached."""
        if self._failed:
            return
        with self._lock:
            if media_key in self._ready:
                return
            if self._in_flight == media_key:
                return
            self._pending = (media_key, np.ascontiguousarray(rgb, dtype=np.uint8))

    def pop_ready(self, media_key: str) -> npt.NDArray[np.float32] | None:
        """Return completed depth map for ``media_key``, if any."""
        with self._lock:
            return self._ready.pop(media_key, None)

    def has_ready(self, media_key: str) -> bool:
        with self._lock:
            return media_key in self._ready

    @staticmethod
    def media_key(path: Path | None) -> str:
        return str(path.resolve()) if path is not None else ""

    def _loop(self) -> None:
        logger.debug("DepthWorker loop entered")
        try:
            while not self._stop.is_set():
                job = self._take_pending()
                if job is None:
                    time.sleep(0.01)
                    continue
                media_key, rgb = job
                try:
                    if not self._estimator.available:
                        logger.error(
                            "Depth estimation disabled: %s",
                            self._estimator.load_error or "model load failed",
                        )
                        self._failed = True
                        continue
                    logger.info(
                        "Estimating depth for %s (%dx%d) — video uses first frame only; "
                        "TODO: per-frame depth for temporal parallax",
                        Path(media_key).name,
                        rgb.shape[1],
                        rgb.shape[0],
                    )
                    t0 = time.perf_counter()
                    depth = self._estimator.estimate(rgb)
                    inference_ms = (time.perf_counter() - t0) * 1000.0
                    with self._lock:
                        self._ready[media_key] = depth
                        self._in_flight = None
                        self._last_inference_ms = inference_ms
                        self._last_inference_key = media_key
                    logger.info(
                        "Depth map ready for %s (%dx%d) inference=%.1fms",
                        Path(media_key).name,
                        depth.shape[1],
                        depth.shape[0],
                        inference_ms,
                    )
                except Exception:
                    logger.exception("Depth estimation failed for %s", media_key)
                    self._failed = True
                    with self._lock:
                        self._in_flight = None
        except Exception:
            logger.exception("DepthWorker loop crashed")
        finally:
            logger.debug("DepthWorker loop exited")

    def _take_pending(self) -> tuple[str, npt.NDArray[np.uint8]] | None:
        with self._lock:
            if self._pending is None:
                return None
            job = self._pending
            self._pending = None
            self._in_flight = job[0]
            return job
