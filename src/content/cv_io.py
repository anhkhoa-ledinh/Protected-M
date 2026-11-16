"""OpenCV I/O helpers with Unicode path support on Windows."""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


def imread_bgr(path: Path | str) -> npt.NDArray[np.uint8] | None:
    """
    Load a BGR image from disk.

    ``cv2.imread`` fails on Windows when the path contains non-ASCII characters
    (e.g. Vietnamese folder names). ``np.fromfile`` + ``cv2.imdecode`` works.
    """
    file_path = Path(path)
    if not file_path.is_file():
        return None

    try:
        data = np.fromfile(str(file_path), dtype=np.uint8)
    except OSError as exc:
        logger.error("Cannot read image bytes from %s: %s", file_path, exc)
        return None

    if data.size == 0:
        return None

    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        logger.error("cv2.imdecode failed for %s", file_path)
    return image


def open_video_capture(path: Path | str) -> cv2.VideoCapture | None:
    """Open a video file, trying Unicode-safe URI form on Windows."""
    file_path = Path(path)
    if not file_path.is_file():
        return None

    candidates = [str(file_path.resolve())]
    try:
        candidates.append(file_path.resolve().as_uri())
    except ValueError:
        pass

    for source in candidates:
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        if cap.isOpened():
            return cap
        cap.release()

    cap = cv2.VideoCapture(str(file_path))
    if cap.isOpened():
        return cap
    cap.release()
    return None
