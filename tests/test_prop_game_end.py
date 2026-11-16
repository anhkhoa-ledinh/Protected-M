"""
Kết thúc ván và điều kiện đối chứng — PB-015 C2 (sửa 16/10). Không OpenGL.

App dựng bằng ``__new__``: chỉ gán các trường mà đường kết thúc ván đọc tới,
renderer là ``SimpleNamespace`` nhận toast và danh sách vật.
"""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.config.schema import ContentSettings
from src.content.prop_game import GameState


def _app(root):
    """Ván t22 đang dở: 3 vật, mở lúc 10 s, mắt nghỉ đã đo."""
    from src.engine.app import HcpdApplication

    app = HcpdApplication.__new__(HcpdApplication)
    app._root = Path(root)
    app._renderer = SimpleNamespace(show_toast=lambda *a, **k: None,
                                    set_props=lambda items: None)
    app._content = ContentSettings()
    app._content.overscan_design_distance_mm = 450.0
    app._rest_eye, app._rest_spread = (-22.0, 64.0, 503.0), (11.0, 8.0)
    app._rest_samples = None
    app._tracking_off = False
    app._watch_until_s = None
    app._game = GameState(image="t22.png", tracking_on=True, started_at_s=10.0,
                          props=[{"pos_mm": (0.0, 0.0, -200.0)}] * 3)
    app._game_planes = ()
    return app


class AbandonedGameTest(unittest.TestCase):
    """
    Ván bỏ dở vẫn để lại đúng một dòng; tắt tracking dù một lúc thì ghi OFF.

    Đỏ trên code trước 16/10: không có ``_abandon_game`` (ván dở không ghi gì)
    và phím T bật lại thì ván bị ghi lại thành ON.
    """

    def test_g_again_writes_one_off_row_with_its_end_reason(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            app = _app(tmp)
            app._game.found[0] = True
            app._toggle_tracking()            # OFF giữa ván
            app._toggle_tracking()            # bật lại: dòng vẫn phải là OFF
            app._abandon_game(70.0, "G")
            app._abandon_game(71.0, "G")      # không còn ván: không thêm dòng
            text = (Path(tmp) / "results" / "prop_game.csv").read_text(encoding="utf-8")
        rows = list(csv.reader(text.splitlines()))
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0][-1], "end_reason")
        self.assertEqual(rows[1][1:6], ["t22.png", "1", "3", "60.0", "OFF"])
        self.assertEqual(rows[1][-1], "G")
        self.assertIsNone(app._game)


class ControlEyeTest(unittest.TestCase):
    """
    T OFF ghim mắt ở mắt nghỉ đo được — trừ lúc đang đo lại sau G.

    Đỏ trên code trước 16/10: mắt bị ghim cả trong lúc đo, nên G lần hai đo
    lại chính mắt cũ với dao động 0 mm.
    """

    def test_the_eye_is_not_pinned_while_the_rest_eye_is_measured(self) -> None:
        app = _app("unused")
        app._tracking_off = True
        np.testing.assert_allclose(app._control_eye(), (-22.0, 64.0, 503.0))
        app._rest_samples = []                # G vừa bấm: đang đo
        self.assertIsNone(app._control_eye())
        app._tracking_off, app._rest_samples = False, None
        self.assertIsNone(app._control_eye())


if __name__ == "__main__":
    unittest.main()
