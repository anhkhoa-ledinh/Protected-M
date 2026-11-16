"""Hình học quad HUD (F1) — kiểm thuần numpy, không cần context OpenGL."""

from __future__ import annotations

import unittest

import numpy as np

from src.renderer.debug_overlay import (
    HUD_MARGIN_PX,
    HUD_QUAD_INDICES,
    hud_quad_vertices,
)


def _signed_area(v0, v1, v2) -> float:
    """
    Diện tích có dấu của tam giác trong NDC.

    Dương = quấn ngược chiều kim đồng hồ (CCW) = **mặt trước**, vì
    ``glFrontFace`` để mặc định ``GL_CCW`` và không chỗ nào đổi nó.
    """
    return 0.5 * float(
        (v1[0] - v0[0]) * (v2[1] - v0[1]) - (v2[0] - v0[0]) * (v1[1] - v0[1])
    )


def _hud_rect(tex_w: float, tex_h: float, win_w: float, win_h: float):
    """Khung NDC của HUD, đúng phép tính trong ``_draw_fallback``."""
    x0 = -1.0 + 2.0 * HUD_MARGIN_PX / win_w
    x1 = x0 + 2.0 * tex_w / win_w
    y1 = 1.0 - 2.0 * HUD_MARGIN_PX / win_h
    y0 = y1 - 2.0 * tex_h / win_h
    return x0, y0, x1, y1


class HudQuadWindingTest(unittest.TestCase):
    """
    Quad HUD phải quấn CCW, nếu không nó biến mất sạch.

    ``window.py:106-108`` bật ``GL_CULL_FACE`` với ``glCullFace(GL_BACK)`` ngay
    lúc tạo context, và ``DebugOverlay._draw_fallback`` **không** tắt culling —
    khác ``debug_geometry`` (tắt rồi bật lại quanh phần vẽ của nó). Quad quấn
    theo chiều kim đồng hồ vì thế bị loại trước khi raster hoá: không một pixel
    nào, kể cả nền, và cũng **không có lỗi GL** nào để lần ra. Đo trên ảnh chụp
    2026-10-11: vùng quad có luminance tối thiểu 76/255, không hề có hình chữ
    nhật nền (20, 24, 30).
    """

    def test_both_triangles_are_counter_clockwise(self) -> None:
        corners = hud_quad_vertices(*_hud_rect(500.0, 344.0, 1280.0, 720.0))
        corners = corners.reshape(-1, 4)
        triangles = HUD_QUAD_INDICES.reshape(-1, 3)
        self.assertEqual(len(triangles), 2, "quad phải gồm đúng hai tam giác")

        for order, (a, b, c) in enumerate(triangles):
            area = _signed_area(corners[a], corners[b], corners[c])
            self.assertGreater(
                area,
                0.0,
                f"tam giác {order} (đỉnh {a}->{b}->{c}) quấn CW, diện tích có "
                f"dấu {area:+.3f} — sẽ bị glCullFace(GL_BACK) loại",
            )

    def test_top_edge_maps_to_v_zero(self) -> None:
        """
        Cạnh trên của quad phải ứng với ``v = 0``, cạnh dưới với ``v = 1``.

        ``glTexImage2D`` coi hàng 0 của mảng — tức **đỉnh** canvas chữ — là
        ``v = 0``. Chốt điều này lại để không ai "sửa" chiều quấn bằng cách lật
        V: lật V cũng cho diện tích dương nhưng làm chữ lộn ngược.
        """
        corners = hud_quad_vertices(*_hud_rect(500.0, 344.0, 1280.0, 720.0))
        corners = corners.reshape(-1, 4)
        ys = corners[:, 1]

        for corner in corners[np.isclose(ys, ys.max())]:
            self.assertEqual(float(corner[3]), 0.0, "đỉnh trên phải có v = 0")
        for corner in corners[np.isclose(ys, ys.min())]:
            self.assertEqual(float(corner[3]), 1.0, "đỉnh dưới phải có v = 1")


class HudQuadSizeTest(unittest.TestCase):
    """
    Quad phải bằng ĐÚNG kích thước canvas chữ và nằm lọt trong NDC.

    Ô NDC cố định cũ (25% x 20% viewport) nén canvas 500x344 xuống 320x144 px,
    tức 0.64x ngang và 0.42x dọc; font Hershey scale 0.42 còn cao ~4 px, hiện ra
    nhưng không đọc được (người dùng xác nhận 2026-10-11).
    """

    def test_quad_is_one_to_one_with_the_text_canvas(self) -> None:
        win_w, win_h = 1280.0, 720.0
        tex_w, tex_h = 500.0, 344.0
        x0, y0, x1, y1 = _hud_rect(tex_w, tex_h, win_w, win_h)

        self.assertAlmostEqual((x1 - x0) * win_w / 2.0, tex_w, places=6)
        self.assertAlmostEqual((y1 - y0) * win_h / 2.0, tex_h, places=6)

    def test_quad_stays_inside_ndc_for_the_tallest_realistic_canvas(self) -> None:
        """Canvas cao nhất thực tế: 11 dòng cố định + nhãn lớp + 4 lớp."""
        tallest = 24 + 16 * 20
        x0, y0, x1, y1 = _hud_rect(500.0, float(tallest), 1280.0, 720.0)

        self.assertGreaterEqual(x0, -1.0)
        self.assertLessEqual(x1, 1.0)
        self.assertGreaterEqual(y0, -1.0)
        self.assertLessEqual(y1, 1.0)


if __name__ == "__main__":
    unittest.main()
