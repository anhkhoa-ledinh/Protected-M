"""Kalman filter for 1D constant-velocity tracking."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KalmanConfig:
    """Process and measurement noise tuning."""

    process_noise: float = 1.0
    measurement_noise: float = 4.0


class KalmanFilter1D:
    """
    Constant-velocity Kalman filter for a single scalar channel.

    State vector: [position, velocity].
    """

    def __init__(self, config: KalmanConfig | None = None) -> None:
        self._config = config or KalmanConfig()
        self._x = np.zeros(2, dtype=np.float64)
        self._p = np.eye(2, dtype=np.float64) * 100.0
        self._initialized = False

    def reset(self) -> None:
        """Reset state and covariance."""
        self._x.fill(0.0)
        self._p = np.eye(2, dtype=np.float64) * 100.0
        self._initialized = False

    def filter(self, measurement: float, timestamp: float) -> float:
        """Predict and update with a new measurement."""
        if not self._initialized:
            self._x[0] = measurement
            self._x[1] = 0.0
            self._initialized = True
            self._last_t = timestamp
            return measurement

        dt = max(timestamp - self._last_t, 1e-6)
        self._last_t = timestamp

        f = np.array([[1.0, dt], [0.0, 1.0]], dtype=np.float64)
        q = self._config.process_noise * np.array(
            [[dt**4 / 4.0, dt**3 / 2.0], [dt**3 / 2.0, dt**2]],
            dtype=np.float64,
        )
        h = np.array([[1.0, 0.0]], dtype=np.float64)
        r = np.array([[self._config.measurement_noise]], dtype=np.float64)

        self._x = f @ self._x
        self._p = f @ self._p @ f.T + q

        y = measurement - (h @ self._x)[0]
        s = h @ self._p @ h.T + r
        k = self._p @ h.T @ np.linalg.inv(s)
        self._x = self._x + (k @ np.array([[y]])).ravel()
        self._p = (np.eye(2) - k @ h) @ self._p

        return float(self._x[0])
