"""PB-074: móc tô màu đo đạc — không đặt biến thì không đụng gì."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from src.content import debug_marks
from src.content.debug_marks import (ENV_FILL, ENV_PAD, ENV_TOP, fill_colour,
                                     pad_colour, paint_border, paint_zone,
                                     top_layer_colour)


class TestDefaultIsSilent(unittest.TestCase):
    """Mặc định: ba hàm màu trả None và hai hàm tô không đổi một byte nào."""

    def test_colours_are_none_without_env(self) -> None:
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertIsNone(pad_colour())
            self.assertIsNone(fill_colour())
            self.assertIsNone(top_layer_colour())

    def test_paint_is_a_no_op_when_colour_is_none(self) -> None:
        rgba = np.random.default_rng(0).integers(
            0, 256, (12, 10, 4)).astype(np.uint8)
        before = rgba.copy()
        paint_border(rgba, 3, None)
        paint_zone(rgba[..., :3], np.ones((12, 10), bool), None)
        np.testing.assert_array_equal(rgba, before)

    def test_bad_value_is_ignored(self) -> None:
        for raw in ("", "1,2", "a,b,c", "   "):
            with mock.patch.dict("os.environ", {ENV_PAD: raw}, clear=True):
                self.assertIsNone(pad_colour())


class TestPainting(unittest.TestCase):
    """Khi có màu: tô đúng vành / đúng vùng, không chạm chỗ khác."""

    def test_border_ring_only(self) -> None:
        rgba = np.zeros((10, 10, 4), dtype=np.uint8)
        paint_border(rgba, 2, (255, 0, 0))
        self.assertTrue((rgba[:2, :, 0] == 255).all())
        self.assertTrue((rgba[2:-2, 2:-2, 0] == 0).all())
        self.assertTrue((rgba[..., 3] == 0).all())      # alpha giữ nguyên

    def test_zone_only(self) -> None:
        rgb = np.zeros((6, 6, 3), dtype=np.uint8)
        zone = np.zeros((6, 6), dtype=bool)
        zone[1:3, 1:3] = True
        paint_zone(rgb, zone, (0, 255, 0))
        self.assertTrue((rgb[zone] == (0, 255, 0)).all())
        self.assertTrue((rgb[~zone] == 0).all())

    def test_env_is_parsed(self) -> None:
        with mock.patch.dict("os.environ",
                             {ENV_FILL: "0,0,255", ENV_TOP: " 255,255,255 "},
                             clear=True):
            self.assertEqual(fill_colour(), (0, 0, 255))
            self.assertEqual(top_layer_colour(), (255, 255, 255))


if __name__ == "__main__":
    unittest.main()


class TestPlateMark(unittest.TestCase):
    """PB-075b: tấm nền có màu đánh dấu riêng, mặc định thì không đổi gì."""

    def test_unset_means_no_mark(self) -> None:
        for name in (debug_marks.ENV_FILL, debug_marks.ENV_PLATE):
            os.environ.pop(name, None)
        self.assertIsNone(debug_marks.plate_fill_colour())

    def test_falls_back_to_the_fill_colour(self) -> None:
        """Chỉ đặt HCPD_MARK_FILL thì tấm nền dùng chung màu — y như trước."""
        os.environ.pop(debug_marks.ENV_PLATE, None)
        os.environ[debug_marks.ENV_FILL] = "0,0,255"
        try:
            self.assertEqual(debug_marks.plate_fill_colour(), (0, 0, 255))
        finally:
            os.environ.pop(debug_marks.ENV_FILL, None)

    def test_plate_colour_wins_when_set(self) -> None:
        os.environ[debug_marks.ENV_FILL] = "0,0,255"
        os.environ[debug_marks.ENV_PLATE] = "0,255,0"
        try:
            self.assertEqual(debug_marks.plate_fill_colour(), (0, 255, 0))
            self.assertEqual(debug_marks.fill_colour(), (0, 0, 255))
        finally:
            for name in (debug_marks.ENV_FILL, debug_marks.ENV_PLATE):
                os.environ.pop(name, None)
