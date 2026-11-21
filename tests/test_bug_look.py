r"""
TASK_bugs_look_0130 — cỡ ngẫu nhiên theo seed + màu nguỵ trang của con bọ.

Ba điều phải giữ:
1. Cùng seed (= cùng ảnh) ra đúng một danh sách ``size_k``, mọi giá trị nằm
   trong biên ``[BUG_SIZE_MIN, BUG_SIZE_MAX] × SIZE_FACTOR``.
2. Phép bắn (``pick_target``) đọc ĐÚNG ``size_k`` đã gán — không tính lại,
   không dùng hằng ``SIZE_FACTOR`` cũ: điểm còn trong hình bóng CỠ ĐẦY nhưng
   đã lọt ra ngoài hình bóng đã CO LẠI phải báo hụt.
3. Màu nguỵ trang lấy đúng màu nền thật quanh vị trí con bọ (mắt nghỉ).
"""

from __future__ import annotations

import unittest

import numpy as np

from pathlib import Path

from src.game import hit as H
from src.ui_album.bugs import (BUG_CAMO_STRENGTH, BUG_SIZE_MAX, BUG_SIZE_MIN,
                               SIZE_FACTOR, assign_bug_camo, assign_bug_sizes)

ROOT = Path(__file__).resolve().parents[1]
EYE = (0.0, 0.0, 600.0)
GLASS = (344.0, 194.0)
PROP_MM = (0.0, 0.0, -180.0)
RADIUS_MM = 12.0


def _props(n: int = 8):
    return [{"pos_mm": PROP_MM, "radius_mm": RADIUS_MM} for _ in range(n)]


class BugSizeSeedTest(unittest.TestCase):
    def test_same_seed_gives_the_same_size_list(self) -> None:
        a, b = _props(), _props()
        assign_bug_sizes(a, "anh_mau.png")
        assign_bug_sizes(b, "anh_mau.png")
        self.assertEqual([p["size_k"] for p in a], [p["size_k"] for p in b])

    def test_a_different_seed_gives_a_different_size_somewhere(self) -> None:
        a, b = _props(12), _props(12)
        assign_bug_sizes(a, "anh_mau.png")
        assign_bug_sizes(b, "anh_khac.png")
        self.assertNotEqual([p["size_k"] for p in a], [p["size_k"] for p in b])

    def test_every_size_is_within_the_bounds(self) -> None:
        props = _props(50)
        assign_bug_sizes(props, "anh_mau.png")
        lo, hi = BUG_SIZE_MIN * SIZE_FACTOR, BUG_SIZE_MAX * SIZE_FACTOR
        for prop in props:
            self.assertGreaterEqual(prop["size_k"], lo)
            self.assertLessEqual(prop["size_k"], hi)

    def test_placement_fields_are_untouched(self) -> None:
        # Chỉ thêm size_k; khong duoc dong tới radius_mm / pos_mm (dat cho,
        # is_exposed dua vao do).
        props = _props()
        before = [dict(p) for p in props]
        assign_bug_sizes(props, "anh_mau.png")
        for prop, old in zip(props, before):
            self.assertEqual(prop["pos_mm"], old["pos_mm"])
            self.assertEqual(prop["radius_mm"], old["radius_mm"])


class HitUsesThePerBugSizeTest(unittest.TestCase):
    """Bắn phải đọc đúng ``size_k`` đã gán — không phải ``SIZE_FACTOR`` cũ."""

    def test_a_point_inside_full_size_but_outside_shrunk_size_is_a_miss(self) -> None:
        from src.game.hit import SpriteMask

        mask = SpriteMask.from_rgba(np.full((64, 64, 4), 255, np.uint8))
        full_quad = H.sprite_quad(PROP_MM, RADIUS_MM, SIZE_FACTOR, EYE, GLASS)
        # Diem o 70% nua be rong cua hinh CO DAY — chac chan trong than.
        point = (full_quad.centre_x + 0.70 * full_quad.half_w, full_quad.centre_y)

        small_k = BUG_SIZE_MIN * SIZE_FACTOR
        bugs_small = {0: (PROP_MM, RADIUS_MM, small_k, mask)}
        bugs_full = {0: (PROP_MM, RADIUS_MM, SIZE_FACTOR, mask)}

        hit_full, _ = H.pick_target(bugs_full, point, EYE, (), GLASS)
        hit_small, _ = H.pick_target(bugs_small, point, EYE, (), GLASS)
        self.assertEqual(hit_full, 0, "diem phai trung tren hinh co day")
        self.assertEqual(hit_small, -1,
                         "diem phai HUT tren hinh da co lai — van dang doc "
                         "SIZE_FACTOR cu thay vi size_k rieng cua con bo")

    def test_app_aim_context_and_bug_painter_read_the_same_key(self) -> None:
        # Khoa 'size_k' la nguon DUY NHAT: doc lai y het nhau o ca hai noi.
        props = _props(3)
        assign_bug_sizes(props, "anh_mau.png")
        for prop in props:
            drawn_size = float(prop["radius_mm"]) / 1000.0 * float(prop["size_k"])
            aim_size_factor = float(prop["size_k"])
            self.assertGreater(drawn_size, 0.0)
            self.assertEqual(aim_size_factor, prop["size_k"])


class BugCamoTest(unittest.TestCase):
    def test_camo_colour_over_a_uniform_red_photo_is_red(self) -> None:
        photo = np.zeros((720, 1280, 3), np.uint8)
        photo[..., 0] = 255   # do thuan (kenh R)
        props = [{"pos_mm": PROP_MM}]
        assign_bug_camo(props, photo, EYE, GLASS)
        r, g, b = props[0]["camo_rgb"]
        self.assertAlmostEqual(r, 1.0, delta=2.0 / 255.0)
        self.assertAlmostEqual(g, 0.0, delta=2.0 / 255.0)
        self.assertAlmostEqual(b, 0.0, delta=2.0 / 255.0)

    def test_camo_colour_follows_a_different_patch_for_a_different_position(self) -> None:
        photo = np.zeros((720, 1280, 3), np.uint8)
        photo[:, :640] = (255, 0, 0)
        photo[:, 640:] = (0, 255, 0)
        left = [{"pos_mm": (-100.0, 0.0, -180.0)}]
        right = [{"pos_mm": (100.0, 0.0, -180.0)}]
        assign_bug_camo(left, photo, EYE, GLASS)
        assign_bug_camo(right, photo, EYE, GLASS)
        self.assertNotEqual(left[0]["camo_rgb"], right[0]["camo_rgb"])


class GameFeedback0530Test(unittest.TestCase):
    """
    TASK_bug_art_0530 — người dùng chơi 11/11: bọ vẫn quá to, nguỵ trang tô
    phẳng cả con thành một khối màu. Hai test này THẤT BẠI trên code trước
    bản sửa (biên cũ 0.55-0.85/0.55; công thức ``mix`` thẳng không có
    ``luma``) và xanh sau bản sửa.
    """

    def test_size_bounds_are_the_0530_values(self) -> None:
        self.assertEqual(BUG_SIZE_MIN, 0.28)
        self.assertEqual(BUG_SIZE_MAX, 0.45)
        self.assertEqual(BUG_CAMO_STRENGTH, 0.75)

    def test_shader_keeps_shading_instead_of_a_flat_colour_blob(self) -> None:
        src = (ROOT / "assets" / "shaders" / "sprite.frag").read_text(
            encoding="utf-8")
        self.assertIn("0.299", src)
        self.assertIn("0.587", src)
        self.assertIn("0.114", src)
        self.assertIn("0.55", src)
        self.assertIn("0.9", src)
        # Cong thuc CU (mix thang, khong qua luma) khong con trong file.
        self.assertNotIn("mix(texel.rgb, uCamo.rgb, uCamo.a)", src)


if __name__ == "__main__":
    unittest.main()
