"""PB-073: bóng lớp (``layer_shadow``) — hình học và điều kiện không đổ vào void."""

from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from src.config.schema import ContentSettings
from src.content.layer_shadow import (REFERENCE_GAP_M, resample_to,
                                      shadow_alpha, shadow_geometry,
                                      shadow_rgba)

ROOT = Path(__file__).resolve().parents[1]


def _caster():
    """Alpha 60×80 với một khối đục ở giữa."""
    alpha = np.zeros((60, 80), dtype=np.uint8)
    alpha[20:40, 30:50] = 255
    return alpha


class TestGeometry(unittest.TestCase):
    """Độ dời và độ nhoè tỉ lệ với Δz."""

    def test_scales_with_gap(self) -> None:
        one = shadow_geometry(1000, REFERENCE_GAP_M, 0.012, 0.010)
        self.assertAlmostEqual(one[0], 12.0)
        self.assertAlmostEqual(one[1], 10.0)
        two = shadow_geometry(1000, 2 * REFERENCE_GAP_M, 0.012, 0.010)
        self.assertAlmostEqual(two[0], 24.0)
        self.assertAlmostEqual(two[1], 20.0)

    def test_sign_of_gap_does_not_matter(self) -> None:
        self.assertEqual(shadow_geometry(500, -0.35, 0.012, 0.010),
                         shadow_geometry(500, 0.35, 0.012, 0.010))


class TestShadowAlpha(unittest.TestCase):
    """Điều kiện cốt lõi: không có bóng ở chỗ mặt nhận trong suốt."""

    def test_no_shadow_where_receiver_is_empty(self) -> None:
        caster = _caster()
        receiver = np.zeros_like(caster)
        receiver[:, :40] = 255          # nửa trái là mặt nhận, nửa phải trống
        out = shadow_alpha(caster, receiver, offset_px=4.0, sigma_px=2.0,
                           angle_deg=90.0, opacity=0.45)
        self.assertEqual(int(out[:, 40:].max()), 0)
        self.assertGreater(int(out[:, :40].max()), 0)

    def test_opacity_caps_the_darkness(self) -> None:
        caster = _caster()
        receiver = np.full_like(caster, 255)
        out = shadow_alpha(caster, receiver, offset_px=0.0, sigma_px=0.0,
                           angle_deg=90.0, opacity=0.45)
        self.assertLessEqual(int(out.max()), int(round(255 * 0.45)) + 1)

    def test_angle_90_moves_the_shadow_down(self) -> None:
        caster = _caster()
        receiver = np.full_like(caster, 255)
        out = shadow_alpha(caster, receiver, offset_px=10.0, sigma_px=0.0,
                           angle_deg=90.0, opacity=1.0)
        rows = np.nonzero(out.max(axis=1))[0]
        self.assertEqual(int(rows.min()), 30)       # 20 + 10
        self.assertEqual(int(rows.max()), 49)       # 39 + 10

    def test_zero_opacity_is_no_shadow(self) -> None:
        caster = _caster()
        out = shadow_alpha(caster, np.full_like(caster, 255), offset_px=5.0,
                           sigma_px=1.0, angle_deg=90.0, opacity=0.0)
        self.assertEqual(int(out.max()), 0)

    def test_shape_mismatch_is_refused(self) -> None:
        with self.assertRaises(ValueError):
            shadow_alpha(_caster(), np.zeros((10, 10), np.uint8),
                         offset_px=1.0, sigma_px=1.0, angle_deg=90.0,
                         opacity=0.5)

    def test_rgba_is_black_with_shadow_in_alpha(self) -> None:
        caster = _caster()
        rgba = shadow_rgba(caster, np.full_like(caster, 255), offset_px=2.0,
                           sigma_px=1.0, angle_deg=90.0, opacity=0.45)
        self.assertEqual(rgba.shape, (60, 80, 4))
        self.assertEqual(int(rgba[..., :3].max()), 0)
        self.assertGreater(int(rgba[..., 3].max()), 0)


class TestResample(unittest.TestCase):
    """Hai lớp khác bề rộng quad: phải quy về hệ của mặt nhận."""

    def test_same_quad_is_identity(self) -> None:
        caster = _caster()
        out = resample_to(caster.astype(np.float32), 1.0, 0.0, caster.shape,
                          1.0, 0.0)
        np.testing.assert_allclose(out, caster, atol=1.0)

    def test_wider_receiver_shrinks_the_caster(self) -> None:
        """Mặt nhận rộng gấp đôi thì vật đổ bóng chiếm nửa số cột."""
        caster = np.zeros((40, 40), dtype=np.float32)
        caster[:, 10:30] = 255.0
        out = resample_to(caster, 1.0, 0.0, caster.shape, 2.0, 0.0)
        cols = np.nonzero(out.max(axis=0) > 1.0)[0]
        self.assertAlmostEqual(len(cols), 10, delta=2)


class TestConfigKey(unittest.TestCase):
    """Mặc định tắt; config demo đã commit cũng phải tắt."""

    def test_schema_default_is_off(self) -> None:
        self.assertFalse(ContentSettings().layer_shadow)
        self.assertFalse(ContentSettings.from_dict({}).layer_shadow)

    def test_default_values(self) -> None:
        content = ContentSettings()
        self.assertAlmostEqual(content.shadow_offset_frac, 0.012)
        self.assertAlmostEqual(content.shadow_blur_frac, 0.010)
        self.assertAlmostEqual(content.shadow_opacity, 0.45)
        self.assertAlmostEqual(content.shadow_angle_deg, 90.0)

    def test_committed_demo_config_has_no_shadow(self) -> None:
        data = json.loads((ROOT / "assets/config/content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertFalse(data.get("layer_shadow", False))


if __name__ == "__main__":
    unittest.main()
