"""Dấu của t và giới hạn bề rộng cho lớp trước kính — PB-045 L2a."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.renderer.content_plane import (
    SCREEN_Z_MM,
    ContentPlaneRenderer,
    front_layer_half_width_limit,
)


def _stub(e_max=150.0, design_d=600.0):
    """``ContentPlaneRenderer`` không gọi ``__init__`` — chỉ mượn `_overscan_scale`."""
    s = ContentPlaneRenderer.__new__(ContentPlaneRenderer)
    s._content = SimpleNamespace(
        overscan_e_max_mm=e_max, overscan_design_distance_mm=design_d
    )
    s._screen_w_mm, s._screen_h_mm = 344.0, 194.0
    return s


class SignedTTest(unittest.TestCase):
    """
    Lớp nằm TRƯỚC kính không được phóng to.

    Dạng cũ lấy ``abs(plane_z + layer.z_m)`` nên mất dấu: preset (b) có lớp ở
    ``z_world = +0.10 m`` tức ``t = −100 mm``, mà renderer đọc thành ``+100`` và
    cho ``k = 2.058`` (đo 2026-10-13), trong khi ``measure_parallax.py`` dùng t
    có dấu và trả ``1.000``. Lớp trước kính cần HẸP lại chứ không rộng ra.
    """

    def test_layer_in_front_of_the_glass_is_not_enlarged(self) -> None:
        plane_z, layer_z = 0.10, 0.0            # preset (b), lop gan nhat
        t_mm = SCREEN_Z_MM - (plane_z + layer_z) * 1000.0
        self.assertEqual(t_mm, -100.0, "t phai AM khi lop nam truoc kinh")

        stub = _stub(design_d=450.0)        # Dd cua phien do 2026-10-13
        scale = stub._overscan_scale(t_mm, 0.2367, 0.1775, 344.0, 194.0)
        self.assertEqual(scale, 1.0, f"lop t<0 bi phong x{scale:.3f}")

        # Ghim lại hành vi CŨ để test không thành hiển nhiên: cùng lớp đó, đưa
        # t qua ``abs()`` như dạng cũ thì ra đúng 2.058 đã đo trên app.
        old = stub._overscan_scale(abs(t_mm), 0.2367, 0.1775, 344.0, 194.0)
        self.assertAlmostEqual(old, 2.058, delta=0.01, msg=f"do duoc {old:.3f}")

    def test_layers_behind_the_glass_are_unchanged(self) -> None:
        """Preset (a) có mọi lớp t ≥ 0 nên phải ra y hệt dạng cũ (abs)."""
        stub = _stub()
        for layer_z in (0.0, -0.1167, -0.2333, -0.35):
            t_signed = SCREEN_Z_MM - (0.0 + layer_z) * 1000.0
            t_abs = abs(0.0 + layer_z) * 1000.0
            self.assertAlmostEqual(t_signed, t_abs, places=9)
            self.assertAlmostEqual(
                stub._overscan_scale(t_signed, 0.344, 0.258, 344.0, 194.0),
                stub._overscan_scale(t_abs, 0.344, 0.258, 344.0, 194.0),
                places=12,
            )


class FrontLayerLimitTest(unittest.TestCase):
    """
    Giới hạn bề rộng cho lớp trước kính (PB-010, FR-LAYER-04).

    ``limit = (W/2)(Dd+t)/Dd − e_max·|t|/Dd``. Quá giới hạn thì mép lớp chạm
    viền màn và bị khung cắt — vi phạm cửa sổ. Chỉ cảnh báo, không tự sửa.
    """

    def test_limit_at_minus_100mm(self) -> None:
        self.assertAlmostEqual(
            front_layer_half_width_limit(-100.0, 344.0, 150.0, 600.0),
            118.3, delta=0.1,
        )

    def test_no_limit_behind_the_glass(self) -> None:
        for t in (0.0, 175.0, 350.0):
            self.assertEqual(
                front_layer_half_width_limit(t, 344.0, 150.0, 600.0), float("inf")
            )


if __name__ == "__main__":
    unittest.main()
