"""Exponential moving average filter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EmaConfig:
    """EMA smoothing factor in (0, 1]."""

    alpha: float = 0.35


class EmaFilter:
    """Simple exponential moving average for scalar signals."""

    def __init__(self, config: EmaConfig | None = None) -> None:
        self._config = config or EmaConfig()
        self._value: float | None = None

    def reset(self) -> None:
        """Clear filter state."""
        self._value = None

    def filter(self, value: float, timestamp: float = 0.0) -> float:
        """Apply EMA smoothing. Timestamp is unused but kept for API parity."""
        _ = timestamp
        if self._value is None:
            self._value = value
            return value

        alpha = self._config.alpha
        self._value = alpha * value + (1.0 - alpha) * self._value
        return self._value
