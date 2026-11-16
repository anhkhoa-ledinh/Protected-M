"""Adaptive quality scaling based on measured frame rate."""

from __future__ import annotations

from dataclasses import dataclass

from src.config.schema import AdaptiveQualitySettings


@dataclass(frozen=True)
class QualityState:
    """Current adaptive quality tier and scale factors."""

    level_index: int
    quality_percent: int
    render_scale: float
    tracking_scale: float
    target_fps: float
    measured_fps: float


class AdaptiveQualityController:
    """
    Reduce tracking/render quality when FPS falls below target.

    Steps through configured quality levels (100 → 75 → 50 → 35).
    """

    def __init__(self, settings: AdaptiveQualitySettings | None = None) -> None:
        self._settings = settings or AdaptiveQualitySettings()
        self._level_index = 0
        self._stable_frames = 0

    @property
    def settings(self) -> AdaptiveQualitySettings:
        return self._settings

    def update(self, measured_fps: float) -> QualityState:
        """Update quality tier from latest FPS measurement."""
        if not self._settings.enabled:
            return self._state(measured_fps)

        levels = self._settings.quality_levels or [100]
        target = self._settings.target_fps
        minimum = self._settings.min_fps

        if measured_fps < minimum and self._level_index < len(levels) - 1:
            self._level_index += 1
            self._stable_frames = 0
        elif measured_fps >= target and self._level_index > 0:
            self._stable_frames += 1
            if self._stable_frames >= 60:
                self._level_index -= 1
                self._stable_frames = 0
        else:
            self._stable_frames = 0

        return self._state(measured_fps)

    def reset(self) -> None:
        """Return to highest quality tier."""
        self._level_index = 0
        self._stable_frames = 0

    def _state(self, measured_fps: float) -> QualityState:
        levels = self._settings.quality_levels or [100]
        index = min(self._level_index, len(levels) - 1)
        quality = levels[index]
        scale = quality / 100.0
        return QualityState(
            level_index=index,
            quality_percent=quality,
            render_scale=scale,
            tracking_scale=scale,
            target_fps=self._settings.target_fps,
            measured_fps=measured_fps,
        )
