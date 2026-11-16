"""Chế độ overscan "pad" và chính sách relief trước kính — PB-045 L2b."""

from __future__ import annotations

import unittest

import numpy as np

from src.math.projection import compute_off_axis_projection
from src.renderer.content_plane import (
    design_eye_offset_mm,
    overscan_need_mm,
    pad_content_scale,
    pad_margin_mm,
    relief_front_offset_m,
)

SW, SH, DD, EMAX = 344.0, 194.0, 450.0, 150.0


class PadGeometryTest(unittest.TestCase):
    """
    ``pad`` phải nới ĐÚNG bằng ``scale``, chỉ khác chỗ phần nới là dải đệm chứ
    không phải phóng to ảnh. Nếu hai bên lệch nhau thì một trong hai sai.
    """

    def test_pad_half_size_matches_the_scale_mode_requirement(self) -> None:
        for t in (0.0, 175.0, 350.0):
            s = pad_content_scale(t, 0.0, DD)          # preset (a): t_plane = 0
            m = pad_margin_mm(t, EMAX, DD)
            self.assertAlmostEqual(
                SW / 2 * s + m, overscan_need_mm(SW, t, EMAX, DD),
                delta=0.5, msg=f"ngang, t={t}")
            self.assertAlmostEqual(
                SH / 2 * s + m, overscan_need_mm(SH, t, EMAX, DD),
                delta=0.5, msg=f"doc, t={t}")

    def test_content_edge_lands_on_the_screen_edge_from_the_design_eye(self) -> None:
        """Mép nội dung (KHÔNG tính dải đệm) chiếu đúng vào mép màn."""
        pa = np.array([-SW / 2000.0, -SH / 2000.0, 0.0])
        pb = np.array([+SW / 2000.0, -SH / 2000.0, 0.0])
        pc = np.array([-SW / 2000.0, +SH / 2000.0, 0.0])
        pe = np.array([0.0, 0.0, DD / 1000.0])
        proj, view = compute_off_axis_projection(pa, pb, pc, pe, 0.01, 8.0)
        for t in (0.0, 175.0, 350.0):
            s = pad_content_scale(t, 0.0, DD)
            p = np.array([SW / 2000.0 * s, 0.0, -t / 1000.0, 1.0])
            clip = proj @ (view @ p)
            ndc_x = float(clip[0] / clip[3])
            # 0.5 mm tren nua be rong 172 mm = 0.0029 trong NDC.
            self.assertAlmostEqual(ndc_x, 1.0, delta=0.0029, msg=f"t={t}")


class DesignEyeTest(unittest.TestCase):
    """
    Mắt nghỉ của người xem không nằm ở tâm màn.

    Đo trên ``run.log`` 2026-10-13 (78 khung có mặt): Eye y trung vị **+74 mm**,
    p90 +92.9, tối đa +145 — nắp laptop ngả ra sau nên mắt cao hơn tâm màn.
    Mọi lớp vì thế đã lệch sẵn lúc NGHỈ, và mép dưới hở nền ở Z 290-340.
    """

    def test_default_changes_nothing(self) -> None:
        for t in (-100.0, 0.0, 175.0, 350.0):
            self.assertEqual(design_eye_offset_mm((0.0, 0.0), t, DD), (0.0, 0.0))

    def test_offset_for_a_raised_eye(self) -> None:
        ox, oy = design_eye_offset_mm((0.0, 140.0), 350.0, DD)
        self.assertAlmostEqual(ox, 0.0, places=9)
        self.assertAlmostEqual(oy, -108.9, delta=0.05)

    def test_offset_quad_centre_projects_to_the_screen_centre(self) -> None:
        """Nhìn từ mắt thiết kế ``(x0, y0, Dd)``, tâm quad phải về giữa màn."""
        pa = np.array([-SW / 2000.0, -SH / 2000.0, 0.0])
        pb = np.array([+SW / 2000.0, -SH / 2000.0, 0.0])
        pc = np.array([-SW / 2000.0, +SH / 2000.0, 0.0])
        x0, y0 = 40.0, 140.0
        pe = np.array([x0 / 1000.0, y0 / 1000.0, DD / 1000.0])
        proj, view = compute_off_axis_projection(pa, pb, pc, pe, 0.01, 8.0)
        for t in (175.0, 350.0):
            ox, oy = design_eye_offset_mm((x0, y0), t, DD)
            p = np.array([ox / 1000.0, oy / 1000.0, -t / 1000.0, 1.0])
            clip = proj @ (view @ p)
            self.assertAlmostEqual(float(clip[0] / clip[3]), 0.0, delta=0.003)
            self.assertAlmostEqual(float(clip[1] / clip[3]), 0.0, delta=0.003)


class ReliefFrontPolicyTest(unittest.TestCase):
    """
    ``shift`` lùi lớp lại đúng bằng phần nhô ra lớn nhất, giữ nguyên hình dạng.

    Đo t21 (B2 closure): relief của lớp ghim là −15.5..+24.6 mm, 6.8% số đỉnh
    nằm trước kính và cực đại rơi vào mép ảnh. Lớp "ghim" vì thế không đứng yên
    — một đỉnh ở +24.6 mm trượt ~27 px mỗi 100 mm đầu ở D = 380.
    """

    def test_shift_moves_the_front_most_vertex_onto_the_glass(self) -> None:
        zw_mm = np.array([-30.0, -5.0, 0.0, 12.5, 20.0], dtype=np.float64)
        span_before = float(zw_mm.max() - zw_mm.min())

        offset = relief_front_offset_m(zw_mm / 1000.0)
        self.assertAlmostEqual(offset, 0.020, places=9)

        after = zw_mm - offset * 1000.0
        self.assertAlmostEqual(float(after.max()), 0.0, delta=0.1)
        self.assertAlmostEqual(float(after.max() - after.min()), span_before, delta=0.1)

    def test_no_shift_when_nothing_is_in_front(self) -> None:
        zw_mm = np.array([-30.0, -5.0, -0.5], dtype=np.float64)
        self.assertEqual(relief_front_offset_m(zw_mm / 1000.0), 0.0)


if __name__ == "__main__":
    unittest.main()
