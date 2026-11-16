"""Đo mắt nghỉ và vòng tìm-thấy — PB-015 C2 (bổ sung 15/10). Không OpenGL."""

from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.content.hidden_props import LayerPlane
from src.content.prop_game import (
    CSV_HEADER,
    FOUND_HOLD_S,
    GameState,
    append_csv,
    csv_row,
    update,
)
from src.content.prop_rest_eye import (
    REST_MAX_SPREAD_MM,
    REST_MIN_FRAMES,
    Z_REST_MAX_MM,
    Z_REST_MIN_MM,
    rest_eye,
    spread_mm,
    z_rest_mm,
)


def _still(n, centre=(-20.0, 100.0, 450.0), jitter=3.0):
    """Luồng mắt của người ngồi yên: lắc nhỏ quanh một tâm."""
    rng = np.random.default_rng(0)
    return [tuple(c + rng.uniform(-jitter, jitter) for c in centre) for _ in range(n)]


class RestEyeTest(unittest.TestCase):
    def test_takes_the_per_axis_median(self) -> None:
        samples = [(0.0, 10.0, 400.0), (10.0, 20.0, 500.0), (5.0, 30.0, 450.0)]
        got, why = rest_eye(samples * 10)
        self.assertIsNone(why)
        self.assertEqual(got, (5.0, 20.0, 450.0))

    def test_rejects_too_few_face_found_frames(self) -> None:
        got, why = rest_eye(_still(REST_MIN_FRAMES - 1))
        self.assertIsNone(got)
        self.assertIn(f"{REST_MIN_FRAMES - 1}", why)

    def test_rejects_a_spread_wider_than_the_limit(self) -> None:
        samples = _still(60)
        # Nửa số mẫu lệch sang phải 60 mm: p10-p90 vượt ngưỡng trên trục x.
        samples = samples[:30] + [(x + 60.0, y, z) for x, y, z in samples[30:]]
        got, why = rest_eye(samples)
        self.assertIsNone(got)
        self.assertTrue(why.startswith("x"), why)
        self.assertGreater(spread_mm(samples, 0), REST_MAX_SPREAD_MM)

    def test_a_wide_y_is_rejected_too(self) -> None:
        samples = _still(60)
        samples = samples[:30] + [(x, y + 80.0, z) for x, y, z in samples[30:]]
        got, why = rest_eye(samples)
        self.assertIsNone(got)
        self.assertTrue(why.startswith("y"), why)

    def test_z_is_clamped_to_the_working_range(self) -> None:
        self.assertEqual(z_rest_mm(250.0), Z_REST_MIN_MM)
        self.assertEqual(z_rest_mm(900.0), Z_REST_MAX_MM)
        self.assertEqual(z_rest_mm(470.0), 470.0)


def _planes():
    """Tấm nền đục + lớp gần có một ô đục ở giữa."""
    near = np.zeros((64, 64), np.uint8)
    near[24:40, 24:40] = 255
    return (LayerPlane(np.full((64, 64), 255, np.uint8), -0.350,
                       0.344, 0.258, 0.0, 0.0, -0.350, -0.350),
            LayerPlane(near, -0.100, 0.100, 0.100, 0.0, 0.0, -0.100, -0.100))


class FoundTimerTest(unittest.TestCase):
    """Phải thấy LIÊN TỤC 0,5 s; nhìn thoáng qua rồi lệch đi thì không tính."""

    def _game(self):
        return GameState(image="t.png", tracking_on=True, started_at_s=0.0,
                         props=[{"pos_mm": (0.0, 0.0, -200.0)}])

    def test_needs_the_full_hold(self) -> None:
        planes, state = _planes(), self._game()
        seen = (300.0, 0.0, 450.0)          # lệch sang phải: thấy vật
        self.assertEqual(update(state, seen, planes, 0.0, 12.0), [])
        self.assertEqual(update(state, seen, planes, FOUND_HOLD_S - 0.01, 12.0), [])
        self.assertEqual(update(state, seen, planes, FOUND_HOLD_S, 12.0), [0])
        self.assertTrue(state.found[0])
        self.assertIsNotNone(state.finished_at_s)

    def test_the_timer_restarts_when_the_prop_hides_again(self) -> None:
        planes, state = _planes(), self._game()
        seen, hidden = (300.0, 0.0, 450.0), (0.0, 0.0, 450.0)
        update(state, seen, planes, 0.0, 12.0)
        update(state, hidden, planes, 0.3, 12.0)
        self.assertIsNone(state.visible_since_s[0])
        self.assertEqual(update(state, seen, planes, 0.6, 12.0), [])
        self.assertFalse(state.found[0])

    def test_a_game_with_no_props_never_finishes(self) -> None:
        state = GameState(image="flat.png", tracking_on=True,
                          started_at_s=0.0, props=[])
        self.assertEqual(update(state, (0.0, 0.0, 450.0), _planes(), 9.0, 12.0), [])
        self.assertIsNone(state.finished_at_s)


class CsvTest(unittest.TestCase):
    def test_writes_the_header_once_and_records_the_rest_eye(self) -> None:
        state = GameState(image="t22.png", tracking_on=False, started_at_s=10.0,
                          props=[{"pos_mm": (0.0, 0.0, -200.0)}])
        state.found[0] = True
        state.finished_at_s = 17.5
        row = csv_row(state, 20.0, 450.0, (-22.0, 64.0, 503.0), (11.0, 8.0),
                      "finished")
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sub" / "prop_game.csv"
            append_csv(path, row)
            append_csv(path, row)
            lines = list(csv.reader(path.open(encoding="utf-8")))
        self.assertEqual(lines[0], list(CSV_HEADER))
        self.assertEqual(len(lines), 3)
        self.assertEqual(lines[1][1:7], ["t22.png", "1", "1", "7.5", "OFF", "450"])
        self.assertEqual(lines[1][7:], ["-22", "64", "503", "11", "8", "finished"])


if __name__ == "__main__":
    unittest.main()
