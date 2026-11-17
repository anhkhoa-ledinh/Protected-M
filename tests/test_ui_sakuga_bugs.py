"""TASK_ui_sakuga_1117 Phase 2: the three circled UI bugs — the tilt
checklist markers drawn two rows too low (bug A), hover/focus rings with
the wrong shape (bug B), and link hit areas wider than the link text
(bug C). All run without GL, against the real manifest + base.png art."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

from src.ui_album import v3, v3_values
from src.ui_album import v3_flow as F

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = ROOT / "assets" / "ui_v3" / "manifest.json"
MANIFEST_DATA = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _image_for(screen: str) -> Image.Image:
    path = ROOT / "assets" / "ui_v3" / MANIFEST_DATA["screens"][screen]["base"]
    return Image.open(path).convert("RGB")


class FakeManifest:
    """Chỉ đủ bề mặt ``v3_values``/``v3`` cần — tránh dựng ``V3Manifest``
    thật (nó tự nạp font/texture, cần GL cho phần khác của app dù hàm này
    không đụng tới)."""

    def __init__(self) -> None:
        self.screens = MANIFEST_DATA["screens"]

    def sprite(self, screen: str, key: str):
        for record in self.screens.get(screen, {}).get("sprites", ()):
            if record["key"] == key:
                return record
        return None


# ---- Bug A: tilt checklist rows -------------------------------------------

class TiltChecklistRowsTest(unittest.TestCase):
    """Mỗi dấu kiểm phải đứng ĐÚNG hàng nhãn của nó, trong cả bốn trạng
    thái (hiện tại = 0, 1, 2, và đã xong hết)."""

    LABELS = ("check_face", "check_left", "check_right")

    def _run(self, tilt_left_done: bool, tilt_right_done: bool,
             face_seen: bool = True):
        manifest = FakeManifest()
        flow = F.V3Flow()
        flow.screen = F.TILT
        flow.face_seen = face_seen
        flow.tilt_left_done = tilt_left_done
        flow.tilt_right_done = tilt_right_done
        state = SimpleNamespace(texts={}, colours={}, borrowed={},
                                sprites={}, mirror=set(), hidden=set(),
                                text_rect={})
        v3_values._tilt(state, flow, manifest)
        return state

    def _label_center_y(self, index: int) -> float:
        spec = v3_values._spec(FakeManifest(), F.TILT, self.LABELS[index])
        y, h = spec["rect"][1], spec["rect"][3]
        return y + h / 2.0

    def _marker_center_y(self, state, index: int) -> float:
        _screen, _key, rect = state.borrowed[f"check#{index}"]
        return rect[1] + rect[3] / 2.0

    def test_current_is_row_0_face(self) -> None:
        state = self._run(tilt_left_done=False, tilt_right_done=False,
                          face_seen=False)
        for i in range(3):
            with self.subTest(row=i):
                self.assertAlmostEqual(self._marker_center_y(state, i),
                                       self._label_center_y(i), delta=2.0)
        self.assertEqual(state.text_rect["check_hint"][1],
                         v3_values.CHECK_ROWS_Y[0])

    def test_current_is_row_1_left(self) -> None:
        state = self._run(tilt_left_done=False, tilt_right_done=False,
                          face_seen=True)
        for i in range(3):
            with self.subTest(row=i):
                self.assertAlmostEqual(self._marker_center_y(state, i),
                                       self._label_center_y(i), delta=2.0)
        self.assertEqual(state.text_rect["check_hint"][1],
                         v3_values.CHECK_ROWS_Y[1])

    def test_current_is_row_2_right(self) -> None:
        state = self._run(tilt_left_done=True, tilt_right_done=False,
                          face_seen=True)
        for i in range(3):
            with self.subTest(row=i):
                self.assertAlmostEqual(self._marker_center_y(state, i),
                                       self._label_center_y(i), delta=2.0)
        self.assertEqual(state.text_rect["check_hint"][1],
                         v3_values.CHECK_ROWS_Y[2])

    def test_all_done(self) -> None:
        state = self._run(tilt_left_done=True, tilt_right_done=True,
                          face_seen=True)
        for i in range(3):
            with self.subTest(row=i):
                self.assertAlmostEqual(self._marker_center_y(state, i),
                                       self._label_center_y(i), delta=2.0)
        self.assertIn("check_hint", state.hidden)

    def test_check_hint_keeps_its_x(self) -> None:
        spec = v3_values._spec(FakeManifest(), F.TILT, "check_hint")
        original_x = spec["rect"][0]
        state = self._run(tilt_left_done=False, tilt_right_done=False,
                          face_seen=False)
        self.assertEqual(state.text_rect["check_hint"][0], original_x)


# ---- Bug B: HOTSPOT_SHAPES coverage + measured radii -----------------------

class HotspotShapeCoverageTest(unittest.TestCase):
    """Mọi hotspot trong bản kê phải có một hình trong HOTSPOT_SHAPES —
    không rơi về 0 (khung vuông) một cách im lặng."""

    def test_every_manifest_hotspot_resolves_a_shape(self) -> None:
        for screen, s in MANIFEST_DATA["screens"].items():
            for h in s.get("hotspots", ()):
                with self.subTest(screen=screen, name=h["n"]):
                    rect = v3.hotspot_rect(screen, h)
                    shape = v3.hotspot_shape(h["n"], rect)
                    self.assertIsInstance(shape, v3.HotspotShape)

    def test_unknown_hotspot_name_raises_not_silently_falls_back(self) -> None:
        with self.assertRaises(ValueError):
            v3.hotspot_shape("totally_unknown_hotspot_xyz", (0, 0, 10, 10))

    def test_continue_is_none_shape(self) -> None:
        shape = v3.hotspot_shape("continue", (0.0, 0.0, 1280.0, 720.0))
        self.assertEqual(shape.kind, "none")

    def test_a_hotspot_covering_half_the_screen_is_none_regardless_of_name(self) -> None:
        shape = v3.hotspot_shape("btn_whatever", (0.0, 0.0, 1280.0, 400.0))
        self.assertEqual(shape.kind, "none")

    def test_pill_kinds(self) -> None:
        for name in ("tab_Solo", "seg_Solo", "display_full", "theme_day",
                     "switch_remember"):
            with self.subTest(name=name):
                shape = v3.hotspot_shape(name, (0.0, 0.0, 88.0, 36.0))
                self.assertEqual(shape.kind, "pill")

    def test_underline_kinds(self) -> None:
        for name in ("link_guest", "link_forgot", "footer_link", "sign_out",
                     "leave"):
            with self.subTest(name=name):
                shape = v3.hotspot_shape(name, (0.0, 0.0, 100.0, 16.0))
                self.assertEqual(shape.kind, "underline")

    def test_circle_kind(self) -> None:
        shape = v3.hotspot_shape("icon_show_password", (0.0, 0.0, 20.0, 20.0))
        self.assertEqual(shape.kind, "circle")

    def test_thumb_kind(self) -> None:
        shape = v3.hotspot_shape("slider", (0.0, 0.0, 194.0, 30.0))
        self.assertEqual(shape.kind, "thumb")


class MeasuredRadiusMatchesTableTest(unittest.TestCase):
    """Đo lại trên chính ``base.png``/sprite PNG — phải khớp bảng
    ``HOTSPOT_SHAPES`` trong ±2 px (Gate G8)."""

    CASES_BASE_PNG = [
        ("v3_01a_sign_in", "btn_Sign in", (462, 450, 356, 54),
         v3.BTN_CORNER_RADIUS_PX),
        ("v3_03_level_select", "level_row_1", (648, 150, 568, 84),
         v3.ROW_CORNER_RADIUS_PX),
        ("v3_05_loadout", "weapon_Precision gun", (64, 180, 300, 112),
         v3.ROW_CORNER_RADIUS_PX),
        ("v3_01a_sign_in", "field_user", (462, 272, 356, 56),
         v3.FIELD_CORNER_RADIUS_PX),
        ("v3_10_pvp_mode", "mode_Competitive", (64, 150, 560, 450),
         v3.MODE_CORNER_RADIUS_PX),
        ("v3_04_choose_photo", "dropzone", (84, 172, 600, 400),
         v3.DROPZONE_CORNER_RADIUS_PX),
        ("v3_04_choose_photo", "recent_memory", (728, 176, 488, 262),
         v3.RECENT_MEMORY_CORNER_RADIUS_PX),
        ("v3_02b_settings", "dropdown_webcam", (520, 362, 220, 40),
         v3.DROPDOWN_CORNER_RADIUS_PX),
    ]

    def test_base_png_classes(self) -> None:
        for screen, name, rect, expected in self.CASES_BASE_PNG:
            with self.subTest(name=name):
                image = _image_for(screen)
                measured = v3.measure_corner_radius_px(
                    image, rect, MANIFEST_DATA["image_scale"], probe_y=0)
                self.assertIsNotNone(measured, f"{name}: khong do duoc")
                self.assertAlmostEqual(measured, expected, delta=2.0)

    def test_sticker_and_slot_sprite_png(self) -> None:
        cases = [
            ("sprites/v3_05_loadout__sticker_Flare.png",
             v3.STICKER_CORNER_RADIUS_PX),
            ("sprites/v3_05_loadout__slot_empty_1.png",
             v3.SLOT_CORNER_RADIUS_PX),
        ]
        for relpath, expected in cases:
            with self.subTest(relpath=relpath):
                image = Image.open(ROOT / "assets" / "ui_v3" / relpath).convert("RGBA")
                measured = v3.measure_alpha_radius_px(image, 1.5)
                self.assertIsNotNone(measured)
                self.assertAlmostEqual(measured, expected, delta=2.0)


# ---- Bug C: link hit areas --------------------------------------------------

class LinkHitAreaWidthTest(unittest.TestCase):
    """Mọi hộp bấm liên kết không được rộng hơn bề rộng chữ + 16 px —
    đo lại ink bounds từ chính ``base.png``."""

    CASES = [
        ("v3_01a_sign_in", "link_forgot", (618, 404, 200, 17)),
        ("v3_01a_sign_in", "link_guest", (430, 626, 420, 16)),
        ("v3_01c_sign_in_error", "link_forgot", (618, 404, 200, 17)),
        ("v3_01c_sign_in_error", "footer_link", (430, 626, 420, 16)),
        ("v3_01b_sign_up", "footer_link", (430, 578, 420, 16)),
        ("v3_02a_menu", "sign_out", (1174, 49, 48, 15)),
    ]

    def test_measured_ink_width_plus_16_covers_the_final_hit_rect(self) -> None:
        for screen, name, original_rect in self.CASES:
            with self.subTest(screen=screen, name=name):
                image = _image_for(screen)
                ink = v3.measure_ink_bounds(image, original_rect, 1.5)
                self.assertIsNotNone(ink, f"{name}: khong thay chu")
                final_rect = v3.LINK_HIT_RECTS[(screen, name)]
                self.assertLessEqual(final_rect[2], ink[2] + 16.0 + 1e-6)

    def test_final_hit_rects_are_narrower_than_the_old_manifest_rect(self) -> None:
        """Lỗi gốc: hộp bấm rộng bằng cả thẻ. Xác nhận hộp MỚI hẹp hơn hộp
        CŨ cho những ô thật sự quá rộng (bỏ qua sign_out — hộp của nó vốn
        đã sát chữ, công thức +16px có thể làm nó rộng hơn một chút, điều
        đó không sai vì lỗi chỉ nói "quá rộng", không nói "không được đổi")."""
        for screen, name, original_rect in self.CASES:
            if name == "sign_out":
                continue
            with self.subTest(screen=screen, name=name):
                final_rect = v3.LINK_HIT_RECTS[(screen, name)]
                self.assertLess(final_rect[2], original_rect[2])

    def test_hit_test_uses_the_narrower_rect(self) -> None:
        manifest = v3.V3Manifest.__new__(v3.V3Manifest)
        manifest.screens = MANIFEST_DATA["screens"]
        # A point inside the OLD wide rect but outside the new narrow one
        # (far left of the "Play as guest" card, no real link text there).
        result = v3.hit_test(manifest, "v3_01a_sign_in", 440.0, 630.0)
        self.assertNotEqual(result, "link_guest")


if __name__ == "__main__":
    unittest.main()
