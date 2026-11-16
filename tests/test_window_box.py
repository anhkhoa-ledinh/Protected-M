"""PB-034: khoá ``surround`` và hình học hộp (miệng hộp = hình chữ nhật màn)."""

from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from src.config.schema import ContentSettings
from src.math.projection import compute_off_axis_projection
from src.scene.primitives import create_open_box
from src.scene.video_scene import SCENE_CLEARANCE_M, farthest_content_z_m
from src.scene.window_box import back_wall_z_m, box_cross_section_m

ROOT = Path(__file__).resolve().parents[1]
SCREEN_W_M, SCREEN_H_M = 0.600, 0.340
VIEWPORT = (1920, 1080)
#: Danh sách mắt chốt ở `TASK_hcpd_fix_3d_game.md` Phase B3 (mm).
EYES_MM = ((0.0, 0.0, 600.0), (150.0, 0.0, 600.0), (-150.0, 0.0, 600.0),
           (0.0, 80.0, 600.0), (0.0, -80.0, 600.0))


def _project(point_m, eye_mm, width_px, height_px):
    """Điểm world (m) → pixel khung nhìn, qua đúng ma trận app dùng."""
    half_w, half_h = SCREEN_W_M * 0.5, SCREEN_H_M * 0.5
    pa = np.array([-half_w, -half_h, 0.0])
    pb = np.array([half_w, -half_h, 0.0])
    pc = np.array([-half_w, half_h, 0.0])
    eye = np.asarray(eye_mm, dtype=np.float64) / 1000.0
    projection, view = compute_off_axis_projection(pa, pb, pc, eye, 0.05, 10.0)
    clip = projection @ view @ np.array([*point_m, 1.0], dtype=np.float64)
    ndc = clip[:3] / clip[3]
    return ((ndc[0] + 1.0) * 0.5 * width_px, (1.0 - ndc[1]) * 0.5 * height_px)


class TestMouthCorners(unittest.TestCase):
    """Bốn góc miệng hộp phải trúng bốn góc khung nhìn, sai ≤ 2 px."""

    def test_corners_hit_viewport_corners(self) -> None:
        half_w, half_h = SCREEN_W_M * 0.5, SCREEN_H_M * 0.5
        width_px, height_px = VIEWPORT
        corners = {
            "duoi-trai": ((-half_w, -half_h, 0.0), (0.0, height_px)),
            "duoi-phai": ((half_w, -half_h, 0.0), (width_px, height_px)),
            "tren-trai": ((-half_w, half_h, 0.0), (0.0, 0.0)),
            "tren-phai": ((half_w, half_h, 0.0), (width_px, 0.0)),
        }
        for eye in EYES_MM:
            for name, (point, want) in corners.items():
                got = _project(point, eye, width_px, height_px)
                self.assertAlmostEqual(got[0], want[0], delta=2.0,
                                       msg=f"{name} x, mat {eye}")
                self.assertAlmostEqual(got[1], want[1], delta=2.0,
                                       msg=f"{name} y, mat {eye}")


class TestOpenBox(unittest.TestCase):
    """Hộp 5 mặt, mở về phía người xem, pháp tuyến hướng vào trong."""

    def _box(self):
        return create_open_box(SCREEN_W_M, SCREEN_H_M, 0.0, -0.40)

    def test_five_quads_no_front_face(self) -> None:
        verts = self._box().vertices.reshape(-1, 6)
        self.assertEqual(verts.shape[0], 20)        # 5 mặt × 4 đỉnh
        self.assertEqual(self._box().vertex_count, 30)
        # Không mặt nào nằm hẳn ở z = 0 với pháp tuyến hướng ra: đó là miệng.
        front = [v for v in verts if abs(v[2]) < 1e-9 and v[5] < -0.5]
        self.assertEqual(front, [])

    def test_normals_point_inwards(self) -> None:
        verts = self._box().vertices.reshape(-1, 6)
        for vertex in verts:
            position, normal = vertex[:3], vertex[3:]
            if abs(normal[1]) > 0.5:                # sàn/trần
                self.assertLess(position[1] * normal[1], 0.0)
            elif abs(normal[0]) > 0.5:              # hai vách
                self.assertLess(position[0] * normal[0], 0.0)
            else:                                   # vách sau nhìn về +z
                self.assertGreater(normal[2], 0.5)

    def test_mouth_matches_the_screen(self) -> None:
        verts = self._box().vertices.reshape(-1, 6)
        at_mouth = verts[np.abs(verts[:, 2]) < 1e-9]
        self.assertAlmostEqual(float(np.abs(at_mouth[:, 0]).max()),
                               SCREEN_W_M * 0.5)
        self.assertAlmostEqual(float(np.abs(at_mouth[:, 1]).max()),
                               SCREEN_H_M * 0.5)


class TestGeometryHelpers(unittest.TestCase):

    def test_cross_section_takes_the_larger_side(self) -> None:
        wide = box_cross_section_m(0.6, 0.34, 16.0 / 9.0)
        self.assertAlmostEqual(wide, 0.34 * 16.0 / 9.0)
        tall = box_cross_section_m(0.6, 0.34, 1.0)
        self.assertAlmostEqual(tall, 0.6)

    def test_back_wall_sits_behind_the_far_layer(self) -> None:
        """Vách sau tính **theo lớp xa nhất thật**, tức có cộng plane Z."""
        content = ContentSettings()
        self.assertAlmostEqual(
            back_wall_z_m(content),
            farthest_content_z_m(content) - SCENE_CLEARANCE_M)
        self.assertLess(back_wall_z_m(content), farthest_content_z_m(content))


class TestConfigKey(unittest.TestCase):

    def test_default_is_void(self) -> None:
        self.assertEqual(ContentSettings().surround, "void")
        self.assertEqual(ContentSettings.from_dict({}).surround, "void")

    def test_committed_demo_config_is_void(self) -> None:
        data = json.loads((ROOT / "assets/config/content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data.get("surround", "void"), "void")

    def _geom(self, layer_z_m, design_eye_mm=(-20.0, 100.0)):
        from src.renderer.content_plane import layer_geometry

        content = replace(ContentSettings(), surround="box",
                          overscan_design_eye_mm=tuple(design_eye_mm))

        def never(*_a, **_k):       # overscan không được gọi ở chế độ hộp
            raise AssertionError("box mode van goi overscan_scale")

        return content, layer_geometry(
            never, content, 0.0, layer_z_m, 0.344, 0.194,
            SCREEN_W_M * 1000.0, SCREEN_H_M * 1000.0)

    def test_box_mode_keeps_the_design_eye_offset(self) -> None:
        """
        Chế độ hộp bỏ biên ``e_max`` nhưng **vẫn dời tâm theo mắt thiết kế**.

        Bản đầu ghim tâm về 0 và ảnh rách dọc: mắt thiết kế cao 100 mm thì lớp
        xa phải dời −77,8 mm, tức 40 % chiều cao ảnh.
        """
        content, geom = self._geom(-0.35)
        design_d = float(content.overscan_design_distance_mm)
        self.assertAlmostEqual(geom.centre_x_m, 20.0 * 350.0 / design_d / 1000.0)
        self.assertAlmostEqual(geom.centre_y_m, -100.0 * 350.0 / design_d / 1000.0)

    def test_quad_covers_the_box_tube_after_the_shift(self) -> None:
        """
        Sau khi dời tâm, quad vẫn phải phủ hết tiết diện ống hộp ở mọi lớp —
        hở một dải là lộ void. Chỉ hệ số phối cảnh thôi thì lớp xa hụt 2,3 mm.
        """
        for layer_z_m in (0.0, -0.175, -0.35):
            _, geom = self._geom(layer_z_m)
            self.assertLessEqual(SCREEN_W_M * 0.5,
                                 geom.quad_w_m * 0.5 - abs(geom.centre_x_m) + 1e-9)
            self.assertLessEqual(SCREEN_H_M * 0.5,
                                 geom.quad_h_m * 0.5 - abs(geom.centre_y_m) + 1e-9)

    def test_deeper_layer_gets_a_bigger_quad(self) -> None:
        """
        Lớp càng sâu quad càng phải to, nếu không nó chiếm góc nhìn nhỏ hơn
        lớp ở mặt kính và các lớp lệch nhau — đúng lỗi của bản đầu, khi
        ``_overscan_scale`` trả 1.0 vì ``e_max`` = 0.
        """
        _, near = self._geom(0.0)
        _, far = self._geom(-0.35)
        self.assertGreater(far.quad_w_m, near.quad_w_m * 1.2)

    def test_near_layer_fits_the_mouth_exactly(self) -> None:
        """Mắt thiết kế ở giữa, lớp ở mặt kính: quad đúng bằng tiết diện hộp."""
        _, geom = self._geom(0.0, design_eye_mm=(0.0, 0.0))
        self.assertAlmostEqual(
            geom.quad_w_m, box_cross_section_m(SCREEN_W_M, SCREEN_H_M,
                                               0.344 / 0.194))

    def test_void_mode_untouched(self) -> None:
        """Mặc định ``void`` vẫn đi đúng đường cũ: gọi overscan, dời tâm như cũ."""
        from src.renderer.content_plane import (design_eye_offset_mm,
                                                layer_geometry)

        content = ContentSettings()
        calls = []

        def fake(t_mm, *args):
            calls.append(t_mm)
            return 1.7

        geom = layer_geometry(fake, content, 0.0, -0.35, 0.344, 0.194,
                              SCREEN_W_M * 1000.0, SCREEN_H_M * 1000.0)
        self.assertEqual(calls, [350.0])
        self.assertAlmostEqual(geom.scale, 1.7)
        self.assertAlmostEqual(geom.quad_w_m, 0.344 * 1.7)
        ox, oy = design_eye_offset_mm(content.overscan_design_eye_mm, 350.0,
                                      content.overscan_design_distance_mm)
        self.assertAlmostEqual(geom.centre_x_m, ox / 1000.0)
        self.assertAlmostEqual(geom.centre_y_m, oy / 1000.0)


if __name__ == "__main__":
    unittest.main()
