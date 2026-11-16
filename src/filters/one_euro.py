"""One Euro Filter for low-latency signal smoothing."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class OneEuroConfig:
    """Tuning parameters for the 1€ filter."""

    min_cutoff: float = 0.15
    beta: float = 0.05
    d_cutoff: float = 1.0


class OneEuroFilter:
    """
    Adaptive low-pass filter (Casiez et al., 2012).

    Reduces jitter while minimizing lag during fast motion.
    """

    def __init__(self, config: OneEuroConfig | None = None) -> None:
        self._config = config or OneEuroConfig()
        self._x_prev: float | None = None
        self._dx_prev: float = 0.0
        self._t_prev: float | None = None

    def reset(self) -> None:
        """Clear filter state."""
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    def filter(self, value: float, timestamp: float) -> float:
        """Filter a scalar sample at the given timestamp (seconds)."""
        if self._x_prev is None or self._t_prev is None:
            self._x_prev = value
            self._t_prev = timestamp
            return value

        dt = max(timestamp - self._t_prev, 1e-6)
        dx = (value - self._x_prev) / dt
        edx = self._smooth(dx, self._dx_prev, dt, self._config.d_cutoff)
        cutoff = self._config.min_cutoff + self._config.beta * abs(edx)
        filtered = self._smooth(value, self._x_prev, dt, cutoff)

        self._x_prev = filtered
        self._dx_prev = edx
        self._t_prev = timestamp
        return filtered

    @staticmethod
    def _smooth(
        value: float,
        previous: float,
        dt: float,
        cutoff: float,
    ) -> float:
        tau = 1.0 / (2.0 * math.pi * max(cutoff, 1e-6))
        alpha = 1.0 / (1.0 + tau / dt)
        return alpha * value + (1.0 - alpha) * previous
