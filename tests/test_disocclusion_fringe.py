r"""
TASK_game_1110 L3 — nở lỗ ra trước khi lấp, bù phần feather tràn ra ngoài
mặt nạ cứng của lớp gần.

``fringe_dilate_px=0`` (mặc định) phải giữ nguyên hành vi hôm nay — các golden
test MD5 ở ``test_fill_method.py``/``test_plate_fill.py`` đứng trên lời hứa
đó. Giá trị > 0 thì vành NGAY SÁT silhouette của lớp gần, vốn trước đây được
coi là "đã biết" (``own_mask`` giữ nguyên), nay bị kéo vào vùng lỗ.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.content.disocclusion import DisocclusionParams, fill_layer_disocclusion


def _scene(size: int = 200, fg_left: int = 100):
    """Nền xanh, khối tiền cảnh đỏ đặc nửa phải — không có khe hở nào."""
    rgb = np.zeros((size, size, 3), dtype=np.uint8)
    rgb[..., 2] = 200
    rgb[:, fg_left:, :] = (220, 30, 30)
    own = np.zeros((size, size), dtype=bool)
    own[:, :fg_left] = True
    nearer = np.zeros((size, size), dtype=bool)
    nearer[:, fg_left:] = True
    return rgb, own, nearer


class FringeDilateDefaultOffTest(unittest.TestCase):
    def test_zero_matches_the_undilated_result_exactly(self) -> None:
        rgb, own, nearer = _scene()
        base = DisocclusionParams(fill_px=25, feather_px=4)
        dilated = DisocclusionParams(fill_px=25, feather_px=4,
                                     fringe_dilate_px=0)
        filled_a, alpha_a, _ = fill_layer_disocclusion(rgb, own, nearer, base)
        filled_b, alpha_b, _ = fill_layer_disocclusion(rgb, own, nearer, dilated)
        np.testing.assert_array_equal(filled_a, filled_b)
        np.testing.assert_array_equal(alpha_a, alpha_b)


class FringeDilateGrowsTheHoleTest(unittest.TestCase):
    """Vành ngay trong ``own_mask`` sát biên phải chuyển thành lỗ, được lấp."""

    def test_a_strip_of_own_mask_near_the_boundary_becomes_filled(self) -> None:
        rgb, own, nearer = _scene(size=200, fg_left=100)
        params = DisocclusionParams(fill_px=25, feather_px=0,
                                    fringe_dilate_px=6,
                                    fringe_reference_width_px=200)
        _filled, alpha, _ = fill_layer_disocclusion(rgb, own, nearer, params)
        row = 100
        # Khong no: cot 99 (ngay trai bien) thuoc own, khong bao gio duoc lap.
        # Co no 6px tren anh rong 200 (ty le 1:1): cot 94..99 nay la lo.
        self.assertGreater(int(alpha[row, 95]), 0)
        self.assertGreater(int(alpha[row, 99]), 0)
        # Xa hon nua ve ben trai thi van la "da biet", khong bi dong vao lo.
        self.assertEqual(int(alpha[row, 80]), 0)

    def test_the_filled_fringe_still_uses_background_colour(self) -> None:
        """C3 (khong tron mau tien canh) phai dung ca o vanh moi mo rong."""
        rgb, own, nearer = _scene(size=200, fg_left=100)
        params = DisocclusionParams(fill_px=25, feather_px=0,
                                    fringe_dilate_px=6,
                                    fringe_reference_width_px=200)
        filled, alpha, _ = fill_layer_disocclusion(rgb, own, nearer, params)
        row = 100
        self.assertGreater(int(alpha[row, 97]), 0)
        # Mau da lap phai LECH han ve phia nen xanh, khong con la do tien canh.
        r, g, b = (int(v) for v in filled[row, 97])
        self.assertLess(r, 150)
        self.assertGreater(b, 100)

    def test_width_scaling_halves_the_dilation_at_half_resolution(self) -> None:
        full_rgb, full_own, full_nearer = _scene(size=200, fg_left=100)
        half_rgb, half_own, half_nearer = _scene(size=100, fg_left=50)
        params = DisocclusionParams(fill_px=12, feather_px=0,
                                    fringe_dilate_px=6,
                                    fringe_reference_width_px=200)
        _f, alpha_full, _ = fill_layer_disocclusion(full_rgb, full_own,
                                                     full_nearer, params)
        _h, alpha_half, _ = fill_layer_disocclusion(half_rgb, half_own,
                                                     half_nearer, params)
        full_reach = int(np.nonzero(alpha_full[100, :100] > 0)[0].min())
        half_reach = int(np.nonzero(alpha_half[50, :50] > 0)[0].min())
        full_depth = 100 - full_reach
        half_depth = 50 - half_reach
        self.assertAlmostEqual(half_depth, full_depth / 2.0, delta=1.5)


if __name__ == "__main__":
    unittest.main()
