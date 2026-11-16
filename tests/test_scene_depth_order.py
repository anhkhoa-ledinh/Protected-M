"""
Khung viền, tấm giữa và tấm nền phải luôn nằm **sau lớp ảnh xa nhất**.

Lỗi thật ngày 31/10: người dùng chọn `layer_z_far_m` = −0,60 ở buổi A/B, lớp
xa chui ra sau khung viền đục ở −0,58, và khung viền che mất ảnh — trên màn
hiện ra một chữ U đen. Khung viền vẽ trước và **có ghi depth**, còn các lớp ảnh
cố ý **không ghi depth**, nên mặt đục nào lọt vào giữa chồng lớp cũng che được
ảnh.
"""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from src.config.schema import ContentSettings
from src.scene.video_scene import (SCENE_CLEARANCE_M, behind_content_z_m,
                                   build_video_window_scene,
                                   farthest_content_z_m)

ROOT = Path(__file__).resolve().parents[1]


def _demo_content(**overrides) -> ContentSettings:
    """`content.demo.json` thật, cộng vài khoá ghi đè."""
    data = json.loads((ROOT / "assets/config/content.demo.json")
                      .read_text(encoding="utf-8"))
    return replace(ContentSettings.from_dict(data), **overrides)


def _build_scene_no_gl(content):
    """
    Dựng cảnh mà **không nạp lên GPU**: thay ``create_mesh`` bằng hàm đồng
    nhất, nên ``node.mesh`` chính là ``MeshData`` còn đọc được toạ độ đỉnh.
    Test phải chạy được khi không có ngữ cảnh OpenGL.
    """
    with mock.patch("src.scene.video_scene.create_mesh",
                    side_effect=lambda data: data):
        return build_video_window_scene(content)


def _node_z_range(root, name):
    """(z nhỏ nhất, z lớn nhất) của một node trong cảnh, đọc từ đỉnh thật."""
    for child in root.children:
        if child.name == name:
            verts = child.mesh.vertices.reshape(-1, 6)
            return float(verts[:, 2].min()), float(verts[:, 2].max())
    raise AssertionError(f"khong thay node {name}")


class TestFarthestContent(unittest.TestCase):

    def test_counts_plane_z_and_layer_z(self) -> None:
        content = _demo_content(layer_z_far_m=-0.60)
        plane_z = float(content.plane_position_m[2])
        self.assertAlmostEqual(farthest_content_z_m(content), plane_z - 0.60)

    def test_without_layers_it_is_the_plane_itself(self) -> None:
        content = _demo_content(layer_clustering_enabled=False,
                                layer_z_far_m=-0.60)
        self.assertAlmostEqual(farthest_content_z_m(content),
                               float(content.plane_position_m[2]))


class TestBehindContent(unittest.TestCase):

    def test_deep_enough_surfaces_are_left_alone(self) -> None:
        """Cấu hình cũ (−0,35) không được đổi một chút nào."""
        content = _demo_content(layer_z_far_m=-0.35)
        for z in (content.frame_z_m, content.mid_layer_z_m,
                  content.backdrop_z_m):
            self.assertEqual(behind_content_z_m(z, content), float(z))

    def test_shallow_surface_is_pushed_back(self) -> None:
        content = _demo_content(layer_z_far_m=-0.60)
        got = behind_content_z_m(content.frame_z_m, content)
        self.assertAlmostEqual(got, farthest_content_z_m(content)
                               - SCENE_CLEARANCE_M)
        self.assertLess(got, farthest_content_z_m(content))


class TestSceneGeometry(unittest.TestCase):
    """Đọc thẳng toạ độ đỉnh của cảnh dựng ra, không tin mỗi công thức."""

    def _assert_all_behind(self, content) -> None:
        root = _build_scene_no_gl(content)
        floor = farthest_content_z_m(content)
        for name in ("window_frame", "mid_panel_left", "mid_panel_right",
                     "backdrop"):
            z_min, z_max = _node_z_range(root, name)
            # Đỉnh lưu float32 nên dung sai 1e-5 m (0,01 mm), không phải 1e-9.
            self.assertLessEqual(
                z_max, floor - SCENE_CLEARANCE_M + 1e-5,
                msg=f"{name} lan vao truoc lop xa nhat ({z_max} > {floor})")

    def test_user_pick_minus_060(self) -> None:
        self._assert_all_behind(_demo_content(layer_z_far_m=-0.60))

    def test_old_value_minus_035(self) -> None:
        self._assert_all_behind(_demo_content(layer_z_far_m=-0.35))

    def test_far_beyond_the_backdrop(self) -> None:
        """Lớp xa đẩy tới −2,9 m thì ngay cả tấm nền −2,8 cũng phải lùi."""
        self._assert_all_behind(_demo_content(layer_z_far_m=-2.90))

    def test_committed_demo_config_is_consistent(self) -> None:
        """Cấu hình demo đang commit tự nó không được dính lỗi chữ U đen."""
        self._assert_all_behind(_demo_content())


if __name__ == "__main__":
    unittest.main()
