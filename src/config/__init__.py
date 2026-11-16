"""Configuration schema and persistence."""

from src.config.loader import load_config, merge_user_config, save_config
from src.config.schema import (
    AdaptiveQualitySettings,
    AppConfig,
    CameraSettings,
    FilterSettings,
    RendererSettings,
    ScreenSettings,
    TrackingSettings,
    UserSettings,
    WebcamMountSettings,
)

__all__ = [
    "AdaptiveQualitySettings",
    "AppConfig",
    "CameraSettings",
    "FilterSettings",
    "RendererSettings",
    "ScreenSettings",
    "TrackingSettings",
    "UserSettings",
    "WebcamMountSettings",
    "load_config",
    "merge_user_config",
    "save_config",
]
