"""Tests for video window content configuration."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from src.config.schema import ContentSettings
from src.scene.primitives import create_window_frame_bezel
from src.scene.video_scene import build_video_window_scene


class TestVideoContent(unittest.TestCase):
    def test_plane_defaults_behind_screen(self) -> None:
        content = ContentSettings()
        self.assertLess(content.plane_position_m[2], 0.0)
        self.assertGreaterEqual(content.plane_position_m[2], -1.0)

    def test_frame_is_in_front_of_content_plane(self) -> None:
        content = ContentSettings()
        self.assertGreater(content.frame_z_m, content.plane_position_m[2])

    def test_depth_layers_order(self) -> None:
        content = ContentSettings()
        self.assertGreater(content.plane_position_m[2], content.mid_layer_z_m)
        self.assertGreater(content.mid_layer_z_m, content.backdrop_z_m)

    def test_window_frame_bezel_vertex_count(self) -> None:
        mesh = create_window_frame_bezel(1.28, 0.72, 0.05, -0.58)
        self.assertEqual(mesh.vertex_count, 24)

    @patch("src.scene.video_scene.create_mesh", return_value=None)
    def test_video_scene_has_depth_layers(self, _mock_mesh) -> None:
        root = build_video_window_scene(ContentSettings())
        names = [child.name for child in root.children]
        self.assertEqual(names, ["backdrop", "mid_panel_left", "mid_panel_right", "window_frame"])


if __name__ == "__main__":
    unittest.main()
