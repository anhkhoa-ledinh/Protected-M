"""Pose-level filter adapters built from scalar filters."""

from __future__ import annotations

import numpy as np

from src.filters.ema import EmaConfig, EmaFilter
from src.filters.kalman import KalmanConfig, KalmanFilter1D
from src.filters.one_euro import OneEuroConfig, OneEuroFilter
from src.math.pose_utils import Pose6DoF
from src.math.quaternion import normalize


class PoseSmoother:
    """Apply independent smoothing to translation and rotation channels."""

    def __init__(
        self,
        filter_type: str,
        one_euro: OneEuroConfig | None = None,
        kalman: KalmanConfig | None = None,
        ema: EmaConfig | None = None,
    ) -> None:
        self._type = filter_type
        self._one_euro_cfg = one_euro or OneEuroConfig()
        self._kalman_cfg = kalman or KalmanConfig()
        self._ema_cfg = ema or EmaConfig()
        self._filters = self._build_filters(filter_type)

    @property
    def filter_type(self) -> str:
        return self._type

    def set_type(self, filter_type: str) -> None:
        """Switch smoothing algorithm at runtime."""
        self._type = filter_type
        self._filters = self._build_filters(filter_type)

    def reset(self) -> None:
        """Reset all internal filter states."""
        for channel in self._filters:
            channel.reset()

    def smooth(self, pose: Pose6DoF) -> Pose6DoF:
        """Return temporally smoothed pose."""
        if self._type == "none":
            return pose

        translation = np.zeros(3, dtype=np.float64)
        for axis in range(3):
            translation[axis] = self._filters[axis].filter(
                float(pose.translation_mm[axis]),
                pose.timestamp,
            )

        rotation = np.zeros(4, dtype=np.float64)
        for index in range(4):
            rotation[index] = self._filters[3 + index].filter(
                float(pose.rotation[index]),
                pose.timestamp,
            )
        rotation = normalize(rotation)

        return Pose6DoF(
            translation_mm=translation,
            rotation=rotation,
            timestamp=pose.timestamp,
        )

    def _build_filters(self, filter_type: str) -> list:
        if filter_type == "one_euro":
            factory = lambda: OneEuroFilter(self._one_euro_cfg)
        elif filter_type == "kalman":
            factory = lambda: KalmanFilter1D(self._kalman_cfg)
        elif filter_type == "ema":
            factory = lambda: EmaFilter(self._ema_cfg)
        elif filter_type == "none":
            return []
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return [factory() for _ in range(7)]
