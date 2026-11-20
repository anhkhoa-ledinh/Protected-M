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

    def test_save_content_settings_only_changes_path_and_type_keys(self) -> None:
        """TASK_perf_textures_1119 Fix 6: sandbox mode's RUNTIME D2
        physical-window override lives on the SAME ``ContentSettings``
        object this function reads from (``plane_position_m``,
        ``plane_width_m``) -- it must never reach the file. Every other
        key already in content.json must survive byte-for-byte."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "assets/config/content.json"
            config_path.parent.mkdir(parents=True)
            original = {
                "enabled": True,
                "mode": "image",
                "content_path": "media/old.png",
                "content_type": "image",
                "loop_video": True,
                "plane_position_m": [0.0, 0.0, -0.75],
                "plane_width_m": 1.28,
                "plane_height_m": 0.72,
                "depth_backend": "onnx",
            }
            config_path.write_text(json.dumps(original), encoding="utf-8")
            media = root / "media" / "new.png"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"\x00")

            # Simulates the D2 override already having mutated the runtime
            # ContentSettings BEFORE this function is called.
            content = ContentSettings(
                plane_position_m=(0.0, 0.0, 0.0), plane_width_m=0.344,
                mode="video", enabled=False, loop_video=False,
            )
            save_content_settings(config_path, content, media, "image", root)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["content_path"], "media/new.png")
            self.assertEqual(payload["content_type"], "image")
            self.assertNotIn("video_path", payload)   # only written for video
            # Every OTHER key: untouched, including the ones the mutated
            # ``content`` object disagrees with.
            for key in ("enabled", "mode", "loop_video", "plane_position_m",
                       "plane_width_m", "plane_height_m", "depth_backend"):
                self.assertEqual(payload[key], original[key], key)

    def test_save_content_settings_video_also_writes_video_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config_path = root / "assets/config/content.json"
            config_path.parent.mkdir(parents=True)
            config_path.write_text(json.dumps({"video_path": "media/old.mp4"}),
                                   encoding="utf-8")
            media = root / "media" / "clip.mp4"
            media.parent.mkdir(parents=True)
            media.write_bytes(b"\x00")

            save_content_settings(config_path, ContentSettings(), media, "video", root)

            payload = json.loads(config_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["video_path"], "media/clip.mp4")

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
