"""MediaPipe face landmark indices for pose estimation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

# 6 canonical landmarks for solvePnP (MediaPipe Face Mesh indices)
PNP_LANDMARKS = (
    1,    # nose tip
    152,  # chin
    33,   # left eye outer corner
    263,  # right eye outer corner
    61,   # left mouth corner
    291,  # right mouth corner
)

LEFT_IRIS_CENTER = 468
RIGHT_IRIS_CENTER = 473
LEFT_EYE_INNER = 133
LEFT_EYE_OUTER = 33
RIGHT_EYE_INNER = 362
RIGHT_EYE_OUTER = 263

# Generic 3D face model as originally shipped, in ARBITRARY units.
#
# Despite the historical name this array is NOT in millimetres.  Measured
# against the live path on 2026-08-20: solvePnP with this model reported
# tvec.z = 2824 mm for a viewer sitting at a true 600 mm, i.e. the array is
# ~4.7x oversized.  Kept verbatim as the provenance of FACE_MODEL_MM below.
FACE_MODEL_RAW: npt.NDArray[np.float64] = np.array(
    [
        [0.0, 0.0, 0.0],           # nose tip
        [0.0, -330.0, -65.0],      # chin
        [-225.0, 170.0, -135.0],   # left eye outer
        [225.0, 170.0, -135.0],    # right eye outer
        [-150.0, -150.0, -125.0],  # left mouth
        [150.0, -150.0, -125.0],   # right mouth
    ],
    dtype=np.float64,
)

# Anthropometric anchor converting FACE_MODEL_RAW into true millimetres.
#
# The model's outer eye-corner span is 450 units; adult ectocanthion-to-
# ectocanthion breadth averages ~90 mm, giving 5.0 units per millimetre.
# With this divisor the generic model measures 90.0 mm eye-to-eye and
# 67.3 mm nose-to-chin, both squarely in the anthropometric range.  That
# places a per-user ``face_scale`` near 1.0 and makes the [0.6, 1.6]
# calibration sanity gate meaningful (see src/pose/calibration.py).
#
# NOTE: the legacy _monocular_scale path is mathematically INVARIANT to this
# rescale — it divides depth_raw by reference_depth_raw, and both shrink by
# the same factor — so uncalibrated behaviour is unchanged.
FACE_MODEL_UNITS_PER_MM: float = 5.0

# Generic 3D face model in TRUE millimeters (anthropometric average).
FACE_MODEL_MM: npt.NDArray[np.float64] = FACE_MODEL_RAW / FACE_MODEL_UNITS_PER_MM


@dataclass(frozen=True)
class LandmarkSet:
    """2D image landmarks and optional iris centers."""

    image_points: npt.NDArray[np.float64]
    left_iris: npt.NDArray[np.float64]
    right_iris: npt.NDArray[np.float64]
    all_landmarks: npt.NDArray[np.float64] | None = None

    @property
    def has_iris(self) -> bool:
        return bool(np.any(self.left_iris) or np.any(self.right_iris))


def normalized_to_pixels(
    landmarks: npt.NDArray[np.float64],
    width: int,
    height: int,
) -> npt.NDArray[np.float64]:
    """Convert normalized [0,1] coords to pixel coordinates."""
    points = landmarks.copy()
    points[:, 0] *= float(width)
    points[:, 1] *= float(height)
    return points
