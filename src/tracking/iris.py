"""Iris landmark extraction utilities."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from src.tracking.landmarks import LandmarkSet


def iris_midpoint(landmarks: LandmarkSet) -> npt.NDArray[np.float64]:
    """Return the midpoint between left and right iris centers in pixels."""
    return (landmarks.left_iris + landmarks.right_iris) * 0.5


def inter_pupillary_distance_px(landmarks: LandmarkSet) -> float:
    """Pixel distance between iris centers."""
    delta = landmarks.left_iris - landmarks.right_iris
    return float(np.linalg.norm(delta))
