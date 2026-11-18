"""Trò tìm vật ẩn — PB-015 C2 chặng 1: đặt vật, node, phím. Không OpenGL."""

from __future__ import annotations

import unittest

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

from src.content.hidden_props import LayerPlane
from src.content.prop_game import (
    COLOR_FOUND,
    COLOR_HIDDEN,
    GameState,
    prop_items,
    start_game,
    status_line,
)


def _plane(z_m, alpha, w=0.344, h=0.258, zmin=None, zmax=None):
    return LayerPlane(alpha, z_m, w, h, 0.0, 0.0,
                      z_m if zmin is None else zmin,
                      z_m if zmax is None else zmax)


def _patch(size=64, lo=24, hi=40):
    a = np.zeros((size, size), np.uint8)
    a[lo:hi, lo:hi] = 255
    return a


def _hiding_planes():
    """Tấm nền đục + một lớp gần có ô đục — đủ để giấu được vật."""
    return (_plane(-0.350, np.full((64, 64), 255, np.uint8)),
            _plane(-0.100, _patch(), w=0.100, h=0.100))


class StartGameTest(unittest.TestCase):
    def test_opens_a_game_with_props_and_a_log_line(self) -> None:
        state = start_game(_hiding_planes(), 12.0, (-20.0, 100.0), 450.0, 5.0, "t.png")
        self.assertTrue(state.props, "khong dat duoc vat nao")
        self.assertEqual(len(state.found), len(state.props))
        self.assertEqual(len(state.visible_since_s), len(state.props))
        self.assertFalse(any(state.found))
        self.assertEqual(state.started_at_s, 5.0)
        self.assertIn("truc=", status_line(state))

    def test_an_image_that_hides_nothing_gives_zero_props(self) -> None:
        planes = (_plane(-0.350, np.full((32, 32), 255, np.uint8)),
                  _plane(-0.000, np.zeros((32, 32), np.uint8)))
        state = start_game(planes, 12.0, (-20.0, 100.0), 450.0, 0.0, "flat.png")
        self.assertEqual(state.props, [])
        self.assertIn("khong giau duoc vat nao", status_line(state))


class PropItemsTest(unittest.TestCase):
    """Renderer nhận mét; ``place_props`` trả mm. Đổi đơn vị phải đúng 1000×."""

    def test_converts_mm_to_m_and_reports_the_found_flag(self) -> None:
        state = GameState(
            image="t.png", tracking_on=True, started_at_s=0.0,
            props=[{"pos_mm": (103.2, -38.8, -290.0)},
                   {"pos_mm": (0.0, 19.4, -160.5)}],
        )
        state.found[1] = True
        items = prop_items(state, 12.0)
        self.assertEqual(len(items), 2)
        np.testing.assert_allclose(items[0][:4], (0.1032, -0.0388, -0.290, 0.012))
        self.assertEqual(items[0][4], COLOR_HIDDEN)
        np.testing.assert_allclose(items[1][:4], (0.0, 0.0194, -0.1605, 0.012))
        self.assertEqual(items[1][4], COLOR_FOUND)

    def test_a_probe_prop_keeps_its_own_colour(self) -> None:
        state = GameState(image="p.png", tracking_on=True, started_at_s=0.0,
                          props=[{"pos_mm": (0.0, 0.0, -450.0), "color": (1.0, 0.0, 0.0)}])
        self.assertEqual(prop_items(state, 12.0)[0][4], (1.0, 0.0, 0.0))

    def test_the_two_colours_differ(self) -> None:
        self.assertNotEqual(COLOR_HIDDEN, COLOR_FOUND)


class LayersRevisionTest(unittest.TestCase):
    """
    App biết "lớp đã dựng lại" qua ``layers_revision``, không so từng quad.

    Đỏ trên code trước C2: ``ContentPlaneRenderer`` không có thuộc tính này.
    """

    def test_apply_bake_bumps_the_revision(self) -> None:
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
        cs.mist_enabled = False          # mist quad cũng cần mesh thật để giải phóng

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

        # Lần gọi thứ hai đi qua _release_layers, nên mesh giả cần release().
        dummy = type("M", (), {"STRIDE": 5, "__init__": lambda s, d: None,
                               "release": lambda s: None})
        self.assertEqual(r.layers_revision, 0)
        with mock.patch.object(cp, "TexturedMesh", dummy), \
             mock.patch.object(cp.ContentPlaneRenderer, "_create_rgba_texture",
                               lambda self, a, **kw: 0):
            baked = _bake_stub(cs, r._last_depth_rgb, depth)
            r._apply_bake(baked, "t.png")
            self.assertEqual(r.layers_revision, 1)
            r._apply_bake(baked, "t.png")
        self.assertEqual(r.layers_revision, 2)


class GameKeysTest(unittest.TestCase):
    """G và T là phím sandbox của trò chơi; mặc định không bật gì."""

    def test_defaults_are_off(self) -> None:
        from src.renderer.window import InputState

        state = InputState()
        self.assertFalse(state.game_toggle)
        self.assertFalse(state.toggle_tracking)

    def test_g_and_t_set_their_flags_on_press(self) -> None:
        import glfw

        from src.renderer.window import GlfwWindow

        for key, attr in ((glfw.KEY_G, "_game_toggle_requested"),
                          (glfw.KEY_T, "_tracking_toggle_requested")):
            w = GlfwWindow.__new__(GlfwWindow)
            w._game_toggle_requested = False
            w._tracking_toggle_requested = False
            w._on_key(None, key, 0, glfw.RELEASE, 0)
            self.assertFalse(getattr(w, attr), "nha phim khong duoc tinh")
            w._on_key(None, key, 0, glfw.PRESS, 0)
            self.assertTrue(getattr(w, attr))


if __name__ == "__main__":
    unittest.main()
