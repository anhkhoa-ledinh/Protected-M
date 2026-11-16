"""Tests for runtime media format detection and persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from src.config.schema import ContentSettings
from src.content.formats import detect_content_type, is_supported_media
from src.content.persistence import media_path_for_config, save_content_settings


class TestContentFormats(unittest.TestCase):
    def test_detect_image_extensions(self) -> None:
        for ext in (".jpg", ".jpeg", ".PNG", ".webp"):
            self.assertEqual(detect_content_type(f"photo{ext}"), "image")

    def test_detect_video_extensions(self) -> None:
        for ext in (".mp4", ".mov", ".avi", ".mkv"):
            self.assertEqual(detect_content_type(f"clip{ext}"), "video")

    def test_unsupported_extension(self) -> None:
        self.assertIsNone(detect_content_type("notes.txt"))
        self.assertFalse(is_supported_media("archive.zip"))

    def test_resolved_media_path_prefers_content_path(self) -> None:
        root = Path("/project")
        content = ContentSettings(
            content_path="assets/images/poster.png",
            video_path="assets/video/demo.mp4",
        )
        self.assertEqual(
            content.resolved_media_path(root),
            root / "assets/images/poster.png",
        )

    def test_save_content_settings_writes_relative_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "assets/config/content.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text("{}", encoding="utf-8")
            media = root / "media" / "clip.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"\x00")

            content = ContentSettings()
            save_content_settings(config_path, content, media, "video", root)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["content_path"], "media/clip.mp4")
            self.assertEqual(payload["content_type"], "video")

    def test_media_path_for_config_keeps_absolute_outside_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "project"
            root.mkdir()
            external = Path(tmp) / "outside.mp4"
            external.write_bytes(b"\x00")
            stored = media_path_for_config(external, root)
            self.assertEqual(stored, str(external.resolve()))


if __name__ == "__main__":
    unittest.main()
