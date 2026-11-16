"""Supported media formats for the content plane."""

from __future__ import annotations

from pathlib import Path

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}
SUPPORTED_EXTENSIONS = IMAGE_EXTENSIONS | VIDEO_EXTENSIONS

FILE_DIALOG_FILTER = [
    ("Media files", "*.jpg *.jpeg *.png *.webp *.mp4 *.mov *.avi *.mkv"),
    ("Images", "*.jpg *.jpeg *.png *.webp"),
    ("Videos", "*.mp4 *.mov *.avi *.mkv"),
    ("All files", "*.*"),
]


def detect_content_type(path: Path | str) -> str | None:
    """Return ``image``, ``video``, or None if unsupported."""
    suffix = Path(path).suffix.lower()
    if suffix in IMAGE_EXTENSIONS:
        return "image"
    if suffix in VIDEO_EXTENSIONS:
        return "video"
    return None


def is_supported_media(path: Path | str) -> bool:
    return detect_content_type(path) is not None
