"""OpenCV I/O helpers with Unicode path support on Windows.

Photos are NOT read here: they go through ``media_gate.import_photo``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)


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
