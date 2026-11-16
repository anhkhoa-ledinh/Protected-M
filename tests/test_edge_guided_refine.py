r"""
TASK_game_1110 L2 — guided filter phải SỬA được mép bậc thang, không chỉ chạy.

Dựng một ảnh có cạnh CHÉO thật (nửa tối / nửa sáng), rồi một mặt nạ "bầu
640 px" giả: đúng cạnh chéo đó nhưng lượng tử hoá theo khối 8 px — y như
``INTER_NEAREST`` phóng một bản đồ nhãn 640 px lên ảnh 2048 px. Guided filter
phải kéo biên alpha về gần cạnh chéo THẬT hơn hẳn so với mặt nạ bậc thang gốc.
"""

from __future__ import annotations

import unittest

import numpy as np

from src.content.layer_clustering import refine_edge_alpha_guided

SIZE = 256
BLOCK = 8


def _diagonal_scene():
    """Ảnh HxW: tối bên trái cạnh ``y = x``, sáng bên phải — cạnh THẬT mịn."""
    yy, xx = np.mgrid[0:SIZE, 0:SIZE]
    rgb = np.where((xx > yy)[..., None], np.uint8(220), np.uint8(30))
    rgb = np.repeat(rgb, 3, axis=2)
    true_mask = xx > yy
    return np.ascontiguousarray(rgb, dtype=np.uint8), true_mask


def _blocky_mask(true_mask):
    """``true_mask`` lượng tử hoá theo khối ``BLOCK`` px — mô phỏng vote 640."""
    small = true_mask[::BLOCK, ::BLOCK]
    blocky = np.kron(small, np.ones((BLOCK, BLOCK), dtype=bool))
    return blocky[:SIZE, :SIZE]


def _boundary_error(alpha_u8, true_mask) -> float:
    """Khoảng cách trung bình (px) giữa biên alpha=127 và biên thật ``y=x``."""
    predicted = alpha_u8.astype(np.float32) > 127.0
    # Với mỗi hàng, cột đầu tiên mà predicted lật True — so với y=x (cột = hàng).
    errs = []
    for row in range(SIZE):
        cols = np.nonzero(predicted[row])[0]
        boundary = cols[0] if cols.size else SIZE
        errs.append(abs(boundary - row))
    return float(np.mean(errs))


class GuidedRefineFixesTheBlockGridTest(unittest.TestCase):
    def setUp(self) -> None:
        self.rgb, self.true_mask = _diagonal_scene()
        self.blocky = _blocky_mask(self.true_mask)
        self.blocky_alpha = np.where(self.blocky, np.uint8(255), np.uint8(0))

    def test_refined_boundary_is_closer_to_the_true_edge(self) -> None:
        before = _boundary_error(self.blocky_alpha, self.true_mask)
        refined = refine_edge_alpha_guided(self.rgb, self.blocky_alpha,
                                           radius_px=8, feather_px=3)
        after = _boundary_error(refined, self.true_mask)
        self.assertLess(after, before * 0.6,
                        f"truoc {before:.2f}px, sau {after:.2f}px — "
                        f"guided filter khong sua duoc may bac thang")

    def test_output_shape_and_dtype_are_unchanged(self) -> None:
        refined = refine_edge_alpha_guided(self.rgb, self.blocky_alpha)
        self.assertEqual(refined.shape, self.blocky_alpha.shape)
        self.assertEqual(refined.dtype, np.uint8)

    def test_feather_band_stays_within_a_few_pixels(self) -> None:
        """Dải chuyển alpha (10%..90%) phải hẹp — kịch bản đòi 2-3 px."""
        refined = refine_edge_alpha_guided(self.rgb, self.blocky_alpha,
                                           radius_px=8, feather_px=3)
        row = SIZE // 2
        line = refined[row].astype(np.float32)
        lo_idx = np.argmax(line > 25.5)
        hi_idx = np.argmax(line > 229.5)
        self.assertLess(abs(hi_idx - lo_idx), 10,
                        f"dai feather o hang giua rong {abs(hi_idx - lo_idx)} px")


if __name__ == "__main__":
    unittest.main()
