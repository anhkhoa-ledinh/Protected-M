"""Đặt vật ẩn giữa các lớp — PB-015 phần 1. Thuần numpy, không OpenGL."""

from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


def _bake_stub(cs, rgb, depth):
    """Bộ lớp thật, dựng tại chỗ, gói thành ``BakeResult`` cho ``_apply_bake``."""
    from src.content.bake import result_from_cluster
    from src.content.layer_clustering import cluster_into_layers

    res = cluster_into_layers(
        rgb, depth, n_clusters=int(cs.layer_count),
        z_near_m=cs.layer_z_near_m, z_far_m=cs.layer_z_far_m,
        disocclusion=None)
    return result_from_cluster(res, depth, int(cs.layer_count))

from src.content.hidden_props import (
    RAYS_CLEAR_MIN,
    LayerPlane,
    alpha_at,
    is_visible,
    layer_planes,
)
from src.content.hidden_props_search import place_props


def _plane(z_m, alpha, w=0.344, h=0.258, cx=0.0, cy=0.0, zmin=None, zmax=None):
    z0 = z_m if zmin is None else zmin
    z1 = z_m if zmax is None else zmax
    return LayerPlane(alpha, z_m, w, h, cx, cy, z0, z1)


def _opaque_patch(size=64, lo=24, hi=40):
    """Mảng alpha trong suốt, có một ô đục ở giữa."""
    a = np.zeros((size, size), np.uint8)
    a[lo:hi, lo:hi] = 255
    return a


class AlphaLookupTest(unittest.TestCase):
    """Tra alpha phải theo đúng quy ước UV của mesh: v = 0 ở MÉP TRÊN."""

    def test_outside_the_quad_reads_zero(self) -> None:
        pl = _plane(-0.35, np.full((8, 8), 255, np.uint8))
        self.assertEqual(alpha_at(pl, 1000.0, 0.0), 0.0)
        self.assertEqual(alpha_at(pl, 0.0, 1000.0), 0.0)

    def test_top_half_maps_to_the_first_rows(self) -> None:
        a = np.zeros((8, 8), np.uint8)
        a[:4, :] = 255                      # nua TREN cua anh
        pl = _plane(-0.35, a, w=0.100, h=0.100)
        self.assertEqual(alpha_at(pl, 0.0, +30.0), 1.0)
        self.assertEqual(alpha_at(pl, 0.0, -30.0), 0.0)


class VisibilityTest(unittest.TestCase):
    """
    Vật nấp sau ô đục của lớp gần, và lộ ra khi mắt lệch đủ xa.

    ``is_visible`` chỉ xét mặt phẳng phẳng tại z đã shift — relief bên trong
    lớp bị bỏ qua, đó là trần đã biết của cách làm này.
    """

    def _scene(self):
        near = _plane(-0.10, _opaque_patch(), w=0.200, h=0.200)
        return (near,)

    def test_hidden_at_the_centre_eye(self) -> None:
        self.assertFalse(
            is_visible((0.0, 0.0, 450.0), (0.0, 0.0, -200.0), 12.0, self._scene())
        )

    def test_visible_after_moving_far_enough(self) -> None:
        self.assertTrue(
            is_visible((300.0, 0.0, 450.0), (0.0, 0.0, -200.0), 12.0, self._scene())
        )

    def test_matches_a_hand_computed_ray_with_shift_and_scale(self) -> None:
        """
        Bản ghi có ``k ≠ 1`` và ``shift ≠ 0`` phải cho cùng đáp án với tia tính tay.

        Lớp gần: quad 0.400 m (k = 2 trên nội dung 0.200), tâm dời +0.050 m,
        z sau shift −0.080 m. Mắt (0,0,450), vật tại z = −0.200 m, x = 0.
        Giao điểm: s = (−80 − 450)/(−200 − 450) = 0.8154 → x = 0, y = 0.
        Điểm đó nằm ở ô đục giữa quad ⇒ bị che.
        """
        a = _opaque_patch(64, 16, 48)
        near = _plane(-0.080, a, w=0.400, h=0.400, cx=0.050, cy=0.0)
        s = (-80.0 - 450.0) / (-200.0 - 450.0)
        self.assertAlmostEqual(s, 0.8154, places=3)
        self.assertGreaterEqual(alpha_at(near, 0.0, 0.0), 0.5)
        self.assertFalse(
            is_visible((0.0, 0.0, 450.0), (0.0, 0.0, -200.0), 12.0, (near,))
        )


class PlacementTest(unittest.TestCase):
    def test_transparent_near_layer_gives_no_props(self) -> None:
        planes = (
            _plane(-0.350, np.full((32, 32), 255, np.uint8)),
            _plane(-0.000, np.zeros((32, 32), np.uint8)),
        )
        self.assertEqual(place_props(planes, 3, 12.0, (-20.0, 100.0)), [])

    def test_props_keep_clear_of_every_relief_envelope(self) -> None:
        planes = (
            _plane(-0.350, np.full((64, 64), 255, np.uint8),
                   zmin=-0.360, zmax=-0.340),
            _plane(-0.100, _opaque_patch(), zmin=-0.115, zmax=-0.085),
        )
        got = place_props(planes, 3, 12.0, (-20.0, 100.0))
        for p in got:
            z = p["pos_mm"][2] / 1000.0
            for pl in planes:
                self.assertTrue(
                    z + 0.012 <= pl.z_min_m or z - 0.012 >= pl.z_max_m,
                    f"vat o z={z:.3f} cat vao bao relief "
                    f"[{pl.z_min_m:.3f}, {pl.z_max_m:.3f}]",
                )

    def test_reported_offset_really_reveals_the_prop(self) -> None:
        planes = (
            _plane(-0.350, np.full((64, 64), 255, np.uint8)),
            _plane(-0.100, _opaque_patch(), w=0.100, h=0.100),
        )
        got = place_props(planes, 2, 12.0, (-20.0, 100.0))
        self.assertTrue(got, "khong dat duoc vat nao de kiem")
        for p in got:
            x, y, z_mm = p["pos_mm"]
            prop = (x, y, z_mm)
            near = tuple(pl for pl in planes if pl.z_m > z_mm / 1000.0)
            self.assertFalse(is_visible((-20.0, 100.0, 450.0), prop, 12.0, near))
            eye = (-20.0 + p["reveal_450_mm"], 100.0, 450.0)
            self.assertTrue(is_visible(eye, prop, 12.0, near))


class RecordBridgeTest(unittest.TestCase):
    """``layer_planes`` chỉ chuyển dạng, không tính lại gì."""

    def test_fields_come_straight_from_the_record(self) -> None:
        from src.renderer.content_plane import _LayerRenderable

        a = np.full((4, 4), 7, np.uint8)
        rec = _LayerRenderable(
            texture=0, mesh=None, z_m=-0.5, alpha=a,
            quad_w_m=0.8, quad_h_m=0.6, centre_x_m=0.01, centre_y_m=-0.02,
            z_world_m=-0.47, relief_z_min_m=-0.49, relief_z_max_m=-0.45,
        )
        (pl,) = layer_planes((rec,))
        self.assertIs(pl.alpha, a)
        self.assertEqual(
            (pl.z_m, pl.quad_w_m, pl.quad_h_m, pl.centre_x_m, pl.centre_y_m,
             pl.z_min_m, pl.z_max_m),
            (-0.47, 0.8, 0.6, 0.01, -0.02, -0.49, -0.45),
        )
        self.assertEqual(RAYS_CLEAR_MIN, 3)


class DrawPathUsesGeometryTest(unittest.TestCase):
    """
    Getter phải trả thứ ĐƯỜNG VẼ tính, không phải bản sao công thức.

    Chứng minh bằng cách thay ``layer_geometry`` rồi chạy ``_apply_bake``:
    nếu bản ghi đổi theo thì đường vẽ đúng là gọi hàm đó. Đây là loại lỗi mà
    ``MEMORY.md`` mục 7 ghi lại — script đo và renderer lệch nhau qua ``abs(t)``
    suốt một ngày vì mỗi bên giữ một bản công thức riêng.
    """

    def test_patching_layer_geometry_changes_the_records(self) -> None:
        import json
        from unittest import mock

        from src.config.schema import ContentSettings
        from src.renderer import content_plane as cp

        cs = ContentSettings.from_dict(
            json.load(open("assets/config/content.demo.json", encoding="utf-8"))
        )
        cs.adaptive_k_enabled = False
        cs.layer_count = 2
        cs.relief_enabled = False

        r = cp.ContentPlaneRenderer.__new__(cp.ContentPlaneRenderer)
        r._content, r._layers, r._mist = cs, [], None
        r._position = (0.0, 0.0, 0.0)
        r._screen_w_mm, r._screen_h_mm = 344.0, 194.0
        r._plane_width, r._plane_height = 0.344, 0.258
        r._tex_width, r._tex_height = 64, 48
        r._layer_shader, r._depth_worker, r._adaptive_k_report = object(), None, None
        r._bake_worker, r._last_bake = None, None
        r._disocclusion_runtime_enabled = False
        r._mist_runtime_enabled = False
        r._atmospheric_runtime_enabled = False
        r._relief_runtime_enabled = False
        rng = np.random.default_rng(0)
        r._last_depth_rgb = rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)
        depth = np.tile(np.linspace(0.0, 1.0, 64, dtype=np.float32), (48, 1))

        marker = cp.LayerGeometry(
            t_mm=123.0, scale=9.0, quad_w_m=7.0, quad_h_m=5.0,
            centre_x_m=0.0, centre_y_m=0.0,
        )
        dummy = type("M", (), {"STRIDE": 5, "__init__": lambda s, d: None})
        with mock.patch.object(cp, "layer_geometry", return_value=marker), \
             mock.patch.object(cp, "TexturedMesh", dummy), \
             mock.patch.object(cp.ContentPlaneRenderer, "_create_rgba_texture",
                               lambda self, a, **kw: 0):
            r._apply_bake(_bake_stub(cs, r._last_depth_rgb, depth), "t.png")

        recs = r.layer_records()
        self.assertTrue(recs, "_apply_bake khong dung duoc lop nao")
        for rec in recs:
            self.assertEqual((rec.quad_w_m, rec.quad_h_m), (7.0, 5.0))


class RestBoxTest(unittest.TestCase):
    """
    "Ẩn" phải đúng với CẢ hộp nghỉ, không chỉ một điểm mắt.

    ``run2.log`` (24 khung z ≥ 420): x p10 −32 / p90 −10, y p10 84 / p90 101 —
    ngồi yên mắt vẫn đưa võng ~22 mm ngang, ~17 mm dọc. C1 chỉ kiểm một điểm
    nên 4/14 vật lộ ra chỉ sau 20–30 mm, tức tự hiện mà người chơi không nghiêng.
    """

    def _scene(self):
        near = _plane(-0.100, _opaque_patch(), w=0.232, h=0.232)
        return (near,)

    def test_hidden_at_the_centre_but_not_at_the_box_corner(self) -> None:
        from src.content.hidden_props_search import REST_BOX_X_MM, REST_BOX_Y_MM

        near = self._scene()
        prop = (34.4, 0.0, -225.0)
        self.assertFalse(is_visible((0.0, 0.0, 450.0), prop, 12.0, near))
        self.assertTrue(
            is_visible((REST_BOX_X_MM, REST_BOX_Y_MM, 450.0), prop, 12.0, near),
            "canh hop nghi phai nhin thay — neu khong thi canh nay vo nghia",
        )

    def test_rest_box_rejects_it(self) -> None:
        from src.content.hidden_props_search import _hidden_in_box

        self.assertFalse(
            _hidden_in_box((34.4, 0.0, -225.0), 12.0, self._scene(),
                           0.0, 0.0, (450.0, 330.0))
        )


class MinRevealTest(unittest.TestCase):
    """Lộ sớm hơn ``MIN_REVEAL_MM`` thì vật tự hiện trong lúc ngồi yên."""

    def _planes(self):
        return (_plane(-0.350, np.full((16, 16), 255, np.uint8)),
                _plane(-0.100, np.full((16, 16), 255, np.uint8)))

    def _candidate_with_reveal(self, value):
        from unittest import mock

        from src.content import hidden_props_search as hs

        with mock.patch.object(hs, "_hidden_in_box", return_value=True),              mock.patch.object(hs, "_reveal_offset", return_value=value):
            return hs._candidate(self._planes(), (0.0, 0.0, -225.0), 12.0,
                                 (0.0, 0.0), 450.0, 330.0, (120.0, 60.0))

    def test_thirty_millimetres_is_rejected(self) -> None:
        self.assertIsNone(self._candidate_with_reveal(30.0))

    def test_sixty_millimetres_is_kept(self) -> None:
        got = self._candidate_with_reveal(60.0)
        self.assertIsNotNone(got)
        self.assertEqual(got["reveal_450_mm"], 60.0)
        self.assertEqual(got["reveal_axis"], "x")


class YAxisRevealTest(unittest.TestCase):
    """Không lộ được theo x thì phải thử theo y, và nói rõ trục nào."""

    def test_only_a_y_reveal_keeps_the_prop(self) -> None:
        from unittest import mock

        from src.content import hidden_props_search as hs

        def fake(planes, prop, radius, eye0, span, axis, step=10.0):
            return None if axis == 0 else -50.0

        planes = (_plane(-0.350, np.full((16, 16), 255, np.uint8)),
                  _plane(-0.100, np.full((16, 16), 255, np.uint8)))
        with mock.patch.object(hs, "_hidden_in_box", return_value=True),              mock.patch.object(hs, "_reveal_offset", side_effect=fake):
            got = hs._candidate(planes, (0.0, 0.0, -225.0), 12.0, (0.0, 0.0),
                                450.0, 330.0, (120.0, 60.0))
        self.assertIsNotNone(got)
        self.assertEqual(got["reveal_axis"], "y")
        self.assertEqual(got["reveal_450_mm"], -50.0)


class RankingTest(unittest.TestCase):
    """Ưu tiên lộ trong [50, 120] mm; ngoài khoảng đó thì gần 85 mm nhất."""

    def test_picks_the_two_in_the_preferred_band(self) -> None:
        from src.content.hidden_props_search import _rank

        cands = [
            {"pos_mm": (-100.0, 0.0, -225.0), "reveal_450_mm": 45.0},
            {"pos_mm": (0.0, 0.0, -225.0), "reveal_450_mm": 70.0},
            {"pos_mm": (100.0, 0.0, -225.0), "reveal_450_mm": 110.0},
        ]
        got = _rank(cands, 2, 12.0)
        self.assertEqual(sorted(abs(c["reveal_450_mm"]) for c in got), [70.0, 110.0])


class ImportOrderTest(unittest.TestCase):
    """
    Mỗi module phải nạp được MỘT MÌNH. C1b tách file nhưng để lại dòng tái xuất
    ``place_props`` ở cuối ``hidden_props``, thành vòng: nạp ``hidden_props_search``
    trước thì ``hidden_props`` quay lại lấy ``place_props`` của một module mới
    khởi tạo một nửa → ImportError. Cả bộ test cũ vẫn xanh vì luôn nạp
    ``hidden_props`` trước, nên lỗi chỉ hiện ở tiến trình sạch.
    """

    def _import_alone(self, module):
        return subprocess.run(
            [sys.executable, "-c", f"import {module}"],
            cwd=str(Path(__file__).resolve().parents[1]),
            capture_output=True, text=True)

    def test_each_module_imports_alone(self) -> None:
        for module in ("src.content.hidden_props_search", "src.content.hidden_props",
                       "src.content.prop_game", "src.content.prop_probe",
                       "src.content.prop_rest_eye"):
            with self.subTest(module=module):
                done = self._import_alone(module)
                self.assertEqual(done.returncode, 0, done.stderr)


if __name__ == "__main__":
    unittest.main()
