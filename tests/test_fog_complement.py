"""PB-071: key ``far_fog_mode`` và cách xám hoá bằng màu bù."""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import numpy as np

from src.config.schema import ContentSettings
from src.content.fog_complement import (T_MAX, complement_mix,
                                        mean_saturation, mix_prepared,
                                        prepare, solve_t_for_layer)
from src.content.layer_clustering import (LayerStagingParams,
                                          stage_layer_color)

ROOT = Path(__file__).resolve().parents[1]
ATMO = (180.0, 190.0, 200.0)

#: Vân tay của ĐẦU RA HÔM NAY, chụp trên nhánh ``sprint-1030`` @ ``bef606a``
#: TRƯỚC khi thêm ``far_fog_mode`` (ảnh ``default_rng(0)`` 48×64×3, atmo
#: (180,190,200), pivot 0.21 / 0.3). Chế độ mặc định phải ra đúng từng byte.
#: 23/11/2026: on Linux, OpenCV's optimized HSV->RGB differs by 1 level on ~1/3072
#: pixels, so the tests below compare arrays (tests/data/fog_golden_*.npz) with a
#: 1-level tolerance. The arrays were generated with cv2.setUseOptimized(False)
#: and each one hashes to the value below (verified 23/11/2026).
GOLDEN_DEFAULT = {
    0.0: "e205184feac3a078228032cf96a85519",
    0.5: "b1166fb6c544b6fa7c8e56a87aa95a09",
    1.0: "562f516e5a01f4549317d79bbc314715",
}
GOLDEN_BMINUS = {
    0.0: "26771acefb99a2603b4d83badd1299be",
    1.0: "bdab462ac6645ee2466f601f0a7ca4a5",
}


def _rgb():
    """Ảnh giả lập cố định, đúng ảnh đã dùng để chụp vân tay."""
    return np.random.default_rng(0).integers(0, 256, (48, 64, 3)).astype(np.uint8)


def _digest(array):
    return hashlib.sha256(array.tobytes()).hexdigest()[:32]


def _assert_close_to_golden(case, out, golden, msg):
    """max |diff| <= 1 level and at most 1 % of pixels differ."""
    case.assertEqual(out.shape, golden.shape, msg)
    diff = np.abs(out.astype(np.int16) - golden.astype(np.int16))
    case.assertLessEqual(int(diff.max()), 1, msg)
    frac = float(np.count_nonzero(diff.any(axis=-1))) / diff.shape[0] / diff.shape[1]
    case.assertLessEqual(frac, 0.01, f"{msg}: {frac:.4%} pixels differ")


def _bminus():
    """Tham số B− (s = 0.35): blend 0.1225, sat 0.86, contrast 0.9125."""
    return LayerStagingParams(enabled=True, atmosphere_blend_max=0.1225,
                              far_desaturation=0.86, far_contrast_reduce=0.9125)


class TestAtmoIsByteIdentical(unittest.TestCase):
    """Mặc định ``"atmo"`` không được đổi một byte nào so với hôm nay."""

    def test_default_mode_matches_golden(self) -> None:
        rgb, params = _rgb(), LayerStagingParams()
        self.assertEqual(params.far_fog_mode, "atmo")
        golden = np.load(ROOT / "tests" / "data" / "fog_golden_default.npz")
        for depth_norm in GOLDEN_DEFAULT:
            out, report = stage_layer_color(rgb, depth_norm, ATMO, params,
                                            index=0, luminance_pivot=0.21)
            _assert_close_to_golden(self, out, golden[f"d{depth_norm}"],
                                    f"depth_norm={depth_norm}")
            self.assertEqual(report.complement_t, 0.0)

    def test_bminus_strength_matches_golden(self) -> None:
        rgb = _rgb()
        golden = np.load(ROOT / "tests" / "data" / "fog_golden_bminus.npz")
        for depth_norm in GOLDEN_BMINUS:
            out, _ = stage_layer_color(rgb, depth_norm, ATMO, _bminus(),
                                       index=1, luminance_pivot=0.3)
            _assert_close_to_golden(self, out, golden[f"d{depth_norm}"],
                                    f"depth_norm={depth_norm}")

    def test_mask_argument_does_not_change_atmo(self) -> None:
        rgb = _rgb()
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[10:30, 5:40] = True
        plain, _ = stage_layer_color(rgb, 1.0, ATMO, LayerStagingParams(),
                                     luminance_pivot=0.21)
        masked, _ = stage_layer_color(rgb, 1.0, ATMO, LayerStagingParams(),
                                      luminance_pivot=0.21, mask=mask)
        np.testing.assert_array_equal(plain, masked)


class TestComplementMode(unittest.TestCase):
    """``"complement"`` đổi lớp xa, khớp bão hoà, và để yên lớp gần."""

    def _params(self):
        return LayerStagingParams(enabled=True, atmosphere_blend_max=0.1225,
                                  far_desaturation=0.86,
                                  far_contrast_reduce=0.9125,
                                  far_fog_mode="complement")

    def test_far_layer_differs_and_reports_t(self) -> None:
        rgb = _rgb()
        atmo_out, _ = stage_layer_color(rgb, 1.0, ATMO, _bminus(),
                                        luminance_pivot=0.3)
        comp_out, report = stage_layer_color(rgb, 1.0, ATMO, self._params(),
                                             luminance_pivot=0.3)
        self.assertGreater(report.complement_t, 0.0)
        self.assertLessEqual(report.complement_t, T_MAX)
        self.assertFalse(np.array_equal(atmo_out, comp_out))

    def test_saturation_matches_the_atmo_arm(self) -> None:
        """Khớp bão hoà: A/B hỏi "xám kiểu nào", không phải "xám bao nhiêu"."""
        rgb = _rgb()
        atmo_out, _ = stage_layer_color(rgb, 1.0, ATMO, _bminus(),
                                        luminance_pivot=0.3)
        comp_out, _ = stage_layer_color(rgb, 1.0, ATMO, self._params(),
                                        luminance_pivot=0.3)
        self.assertAlmostEqual(mean_saturation(comp_out),
                               mean_saturation(atmo_out), delta=1.0)

    def test_nearest_layer_identical_in_both_modes(self) -> None:
        """``depth_norm = 0``: không trộn, không hạ bão hoà → hai chế độ bằng nhau."""
        rgb = _rgb()
        plain, _ = stage_layer_color(rgb, 0.0, ATMO, _bminus(),
                                     luminance_pivot=0.21)
        comp, report = stage_layer_color(rgb, 0.0, ATMO, self._params(),
                                         luminance_pivot=0.21)
        np.testing.assert_array_equal(plain, comp)
        self.assertEqual(report.complement_t, 0.0)

    def test_disabled_staging_bypasses_complement(self) -> None:
        rgb = _rgb()
        out, report = stage_layer_color(
            rgb, 1.0, ATMO,
            LayerStagingParams(enabled=False, far_fog_mode="complement"))
        np.testing.assert_array_equal(out, rgb)
        self.assertEqual(report.complement_t, 0.0)


class TestFormula(unittest.TestCase):
    """Công thức màu bù, chép từ ``scripts/_fog_complement.py``."""

    def test_t_zero_returns_the_image(self) -> None:
        """``t = 0`` chỉ còn sai số làm tròn của vòng sRGB→linear→sRGB (≤ 1)."""
        rgb = _rgb()
        delta = np.abs(complement_mix(rgb, 0.0).astype(int) - rgb.astype(int))
        self.assertLessEqual(int(delta.max()), 1)

    def test_mix_darkens_a_vivid_patch(self) -> None:
        """Trung bình hình học với màu bù vừa xám vừa TỐI đi."""
        patch = np.full((8, 8, 3), (220, 40, 30), dtype=np.uint8)
        mixed = complement_mix(patch, 0.5)
        self.assertLess(float(mixed.mean()), float(patch.mean()))
        self.assertLess(mean_saturation(mixed), mean_saturation(patch))

    def test_prepare_then_mix_equals_the_one_shot_call(self) -> None:
        """Tách ``prepare`` ra khỏi vòng chia đôi không được đổi kết quả."""
        rgb = _rgb()
        prepared = prepare(rgb)
        for t_value in (0.0, 0.05, 0.25, 0.5):
            np.testing.assert_array_equal(mix_prepared(prepared, t_value),
                                          complement_mix(rgb, t_value))

    def test_t_ignores_pixels_outside_the_mask(self) -> None:
        """
        ``t`` chỉ được nhìn điểm CỦA LỚP.

        Thất bại trên bản dò ``t`` ở ảnh thu nhỏ: ``INTER_AREA`` trộn cả điểm
        ngoài mask vào, nên đổi nền là đổi ``t`` (đo được 0,033 trên lớp giữa
        của t25, tức 13/255 ở điểm lệch nhất).
        """
        rgb = _rgb()
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        mask[::3, ::3] = True                   # mask vụn, đúng ca khó
        atmo_out, _ = stage_layer_color(rgb, 1.0, ATMO, _bminus(),
                                        luminance_pivot=0.3)
        elsewhere = rgb.copy()
        elsewhere[~mask] = 255 - elsewhere[~mask]
        atmo_elsewhere = atmo_out.copy()
        atmo_elsewhere[~mask] = 255 - atmo_elsewhere[~mask]
        self.assertAlmostEqual(solve_t_for_layer(rgb, atmo_out, mask),
                               solve_t_for_layer(elsewhere, atmo_elsewhere,
                                                 mask), places=6)

    def test_empty_mask_gives_zero_t(self) -> None:
        rgb = _rgb()
        mask = np.zeros(rgb.shape[:2], dtype=bool)
        self.assertEqual(solve_t_for_layer(rgb, rgb, mask), 0.0)


class TestConfigKey(unittest.TestCase):
    """Key cấu hình: mặc định ``atmo``; demo dùng ``complement`` B−."""

    def test_schema_default_is_atmo(self) -> None:
        self.assertEqual(ContentSettings().far_fog_mode, "atmo")
        self.assertEqual(ContentSettings.from_dict({}).far_fog_mode, "atmo")

    def test_demo_config_uses_complement_with_bminus(self) -> None:
        data = json.loads((ROOT / "assets/config/content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data["far_fog_mode"], "complement")
        self.assertAlmostEqual(data["atmosphere_blend_max"], 0.1225)
        self.assertAlmostEqual(data["far_desaturation"], 0.86)
        self.assertAlmostEqual(data["far_contrast_reduce"], 0.9125)
        self.assertEqual(data["layer_method"], "kmeans_vote")


if __name__ == "__main__":
    unittest.main()
