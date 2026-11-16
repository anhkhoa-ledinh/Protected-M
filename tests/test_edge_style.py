"""PB-072: key ``edge_style`` — mép mềm (hôm nay) và mép cắt dập."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from src.config.schema import ContentSettings
from src.content.disocclusion import DisocclusionParams
from src.content.layer_clustering import (DIECUT_BLUR_PX, LayerEdgeParams,
                                          LayerStagingParams,
                                          cluster_into_layers,
                                          decontaminate_edge_colour)

ROOT = Path(__file__).resolve().parents[1]

#: Vân tay của ĐẦU RA HÔM NAY, chụp ở `d436f57` TRƯỚC khi thêm ``edge_style``.
GOLDEN_SOFT = ("54304d378dc40de9e0ec3480eacaa8fc",
               "01d6ec895ad8aba6bb45b0ea9db4e19c",
               "8d62f1dfea18af8658e4b485a86550c9")


def _scene():
    """Ảnh + depth giả lập cố định, đúng bộ đã dùng để chụp vân tay."""
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (90, 120, 3)).astype(np.uint8)
    ramp = np.linspace(0, 1, 90)[:, None] * np.ones((1, 120))
    depth = (ramp * 0.8 + rng.random((90, 120)) * 0.2).astype(np.float32)
    return rgb, depth


def _layers(style):
    rgb, depth = _scene()
    return cluster_into_layers(
        rgb, depth, n_clusters=3, seed=0, staging=LayerStagingParams(),
        edge=LayerEdgeParams(edge_style=style),
        disocclusion=DisocclusionParams()).layers


class TestSoftIsByteIdentical(unittest.TestCase):
    """Mặc định ``"soft"`` không được đổi một byte nào so với hôm nay."""

    def test_default_style_is_soft(self) -> None:
        self.assertEqual(LayerEdgeParams().edge_style, "soft")
        self.assertEqual(ContentSettings().edge_style, "soft")
        self.assertEqual(ContentSettings.from_dict({}).edge_style, "soft")

    def test_soft_matches_golden(self) -> None:
        for layer, want in zip(_layers("soft"), GOLDEN_SOFT):
            got = hashlib.sha256(layer.rgba.tobytes()).hexdigest()[:32]
            self.assertEqual(got, want, f"lop {layer.index}")

    def test_committed_demo_config_matches_the_user_pick(self) -> None:
        """
        Demo config để ``"diecut"`` vì **người dùng chọn bằng mắt** ở buổi A/B
        Phase 6 (31/10, preset P1 trên `ab\\index_*.png`). Mặc định của CODE
        vẫn là ``"soft"`` — hai thứ khác nhau, test trên đã giữ phần đó.

        Trước buổi A/B test này chốt demo config phải là ``"soft"``, để không
        ai lặng lẽ đổi thứ người dùng nhìn thấy trước khi họ duyệt.
        """
        data = json.loads((ROOT / "assets/config/content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data.get("edge_style", "soft"), "diecut")


class TestDiecut(unittest.TestCase):
    """``"diecut"``: mép hẹp hơn, không viền sáng, không pha màu lớp sau."""

    def test_alpha_ramp_is_narrower(self) -> None:
        soft = _layers("soft")[-1].rgba[..., 3]
        cut = _layers("diecut")[-1].rgba[..., 3]
        partial_soft = int(((soft > 0) & (soft < 255)).sum())
        partial_cut = int(((cut > 0) & (cut < 255)).sum())
        self.assertLess(partial_cut, partial_soft)

    def test_no_rim_light_on_the_nearest_layer(self) -> None:
        """Viền sáng làm sáng dải trong mép; die-cut thì không có dải đó."""
        soft, cut = _layers("soft")[-1], _layers("diecut")[-1]
        mask = soft.mask.astype(bool)
        self.assertGreater(float(soft.rgba[..., :3][mask].mean()),
                           float(cut.rgba[..., :3][mask].mean()))

    def test_pixels_inside_the_mask_keep_their_colour(self) -> None:
        """Tẩy màu chỉ được đụng vành NGOÀI mask."""
        rgb, _ = _scene()
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[20:60, 30:90] = True
        out = decontaminate_edge_colour(rgb, mask, DIECUT_BLUR_PX + 1)
        np.testing.assert_array_equal(out[mask], rgb[mask])

    def test_band_borrows_the_nearest_inside_colour(self) -> None:
        rgb = np.zeros((20, 20, 3), dtype=np.uint8)
        rgb[:, :] = (10, 20, 30)            # nền
        mask = np.zeros((20, 20), dtype=bool)
        mask[8:12, 8:12] = True
        rgb[mask] = (200, 100, 50)          # vật
        out = decontaminate_edge_colour(rgb, mask, 2)
        self.assertTrue((out[7, 8:12] == (200, 100, 50)).all())
        self.assertTrue((out[0, 0] == (10, 20, 30)).all())   # ngoài vành

    def test_far_band_is_untouched(self) -> None:
        rgb, _ = _scene()
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[40:50, 40:50] = True
        out = decontaminate_edge_colour(rgb, mask, 2)
        np.testing.assert_array_equal(out[:30], rgb[:30])

    def test_empty_mask_is_a_no_op(self) -> None:
        rgb, _ = _scene()
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        np.testing.assert_array_equal(
            decontaminate_edge_colour(rgb, mask, 4), rgb)


if __name__ == "__main__":
    unittest.main()
