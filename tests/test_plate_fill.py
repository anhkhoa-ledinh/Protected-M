"""
PB-075: khoá ``plate_fill`` — cách lấp riêng cho **tấm nền**.

Tấm nền (lớp xa nhất) lấp **cả lỗ**, nên vùng lấp trên nó chiếm 4,5–31,2 %
khung hình và mịn hơn hẳn xung quanh (PB-074 Phase 0b). Lớp giữa chỉ lấp một
dải 64 px nên không đáng đổi — khoá này **chỉ** đụng tới tấm nền.
"""

from __future__ import annotations

import hashlib
import json
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.config.schema import ContentSettings
from src.content.disocclusion import (DisocclusionParams,
                                      fill_layer_disocclusion, inpaint_hole)
from src.content.layer_clustering import (LayerEdgeParams, LayerStagingParams,
                                          cluster_into_layers)

ROOT = Path(__file__).resolve().parents[1]

#: Vân tay ĐẦU RA HÔM NAY, chụp ở `2cf8aad` TRƯỚC khi thêm ``plate_fill``.
GOLDEN_TELEA = ("54304d378dc40de9e0ec3480eacaa8fc",
                "01d6ec895ad8aba6bb45b0ea9db4e19c",
                "8d62f1dfea18af8658e4b485a86550c9")


def _scene():
    rng = np.random.default_rng(1)
    rgb = rng.integers(0, 256, (90, 120, 3)).astype(np.uint8)
    ramp = np.linspace(0, 1, 90)[:, None] * np.ones((1, 120))
    depth = (ramp * 0.8 + rng.random((90, 120)) * 0.2).astype(np.float32)
    return rgb, depth


def _layers(**disocclusion):
    rgb, depth = _scene()
    return cluster_into_layers(
        rgb, depth, n_clusters=3, seed=0, staging=LayerStagingParams(),
        edge=LayerEdgeParams(),
        disocclusion=DisocclusionParams(**disocclusion)).layers


class TestDefaultIsUnchanged(unittest.TestCase):
    """Mặc định ``"telea"`` phải dựng lại đúng từng byte ảnh hôm nay."""

    def test_default_value(self) -> None:
        self.assertEqual(ContentSettings().plate_fill, "telea")
        self.assertEqual(ContentSettings.from_dict({}).plate_fill, "telea")
        self.assertEqual(DisocclusionParams().plate_fill, "telea")

    def test_matches_golden(self) -> None:
        for layer, want in zip(_layers(), GOLDEN_TELEA):
            got = hashlib.sha256(layer.rgba.tobytes()).hexdigest()[:32]
            self.assertEqual(got, want, f"lop {layer.index}")

    def test_committed_demo_config_is_telea(self) -> None:
        data = json.loads((ROOT / "assets/config/content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data.get("plate_fill", "telea"), "telea")


class TestShiftmap(unittest.TestCase):

    def _hole_scene(self):
        yy, xx = np.mgrid[0:120, 0:160]
        band = ((np.sin((xx + yy) / 5.0) * 0.5 + 0.5) * 255).astype(np.uint8)
        rgb = np.dstack([band, np.roll(band, 5, 1), np.roll(band, 9, 0)])
        hole = np.zeros((120, 160), np.uint8)
        hole[40:80, 50:110] = 255
        return rgb, hole

    def test_mask_polarity_is_inverted_vs_cv2_inpaint(self) -> None:
        """
        ``cv2.xphoto`` nhận **khác 0 = điểm ĐÃ BIẾT**, ngược với ``cv2.inpaint``.

        Đưa nhầm chiều thì nó đi dựng lại cả ảnh TRỪ cái lỗ: đo thử thấy
        3 303 ms (so với 472 ms) và **điểm ngoài lỗ cũng bị đổi hẳn**. Test
        ghim lại bằng hệ quả đó: ngoài lỗ phải gần như y nguyên.

        "Gần như" chứ không "đúng từng byte" vì đường này đi vòng qua Lab rồi
        quay lại RGB, mà Lab 8 bit làm tròn có mất mát (đo: lệch trung bình
        0,74/255, cao nhất 18 ở điểm màu gắt). Chỗ gọi chỉ chép **vùng lỗ**
        từ kết quả ra (``out[zone] = filled[zone]``) nên phần mất mát này
        không bao giờ chạm tới điểm đã biết của ảnh cuối.
        """
        rgb, hole = self._hole_scene()
        out = inpaint_hole(rgb, hole, 5, "shiftmap")
        outside = ~(hole > 0)
        delta = np.abs(out[outside].astype(int) - rgb[outside].astype(int))
        self.assertLess(float(delta.mean()), 2.0)
        self.assertLess(int(delta.max()), 25)
        self.assertTrue((out[hole > 0] != rgb[hole > 0]).any())

    def test_texture_level_lands_closer_to_the_real_image(self) -> None:
        """
        Thứ cần là vân **đúng bằng** xung quanh, không phải vân nhiều nhất.

        Trên cảnh sọc này Telea còn cho phương sai CAO hơn ảnh thật (63,1 so
        với 40,5) vì nó tự bịa ra một dải chuyển màu; shiftmap cho 47,8, gần
        mức thật hơn. Chỉ số của cả PB-074 lẫn PB-075 (``fill_tex_ratio``)
        cũng đo đúng kiểu này: gần 1 là tốt, không phải càng lớn càng tốt.
        """
        rgb, hole = self._hole_scene()
        patch = (slice(40, 80), slice(50, 110))

        def variance(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64)
            return cv2.Laplacian(gray, cv2.CV_64F)[patch].var()

        real = variance(rgb)
        telea = abs(variance(inpaint_hole(rgb, hole, 5, "telea")) - real)
        shift = abs(variance(inpaint_hole(rgb, hole, 5, "shiftmap")) - real)
        self.assertLess(shift, telea)


class TestPlateOnlyRouting(unittest.TestCase):
    """Khoá chỉ đổi tấm nền; lớp giữa vẫn đi theo ``fill_method``."""

    def _call(self, plate: bool, plate_fill: str):
        seen = {}
        rng = np.random.default_rng(3)
        rgb = rng.integers(0, 256, (60, 80, 3)).astype(np.uint8)
        own = np.zeros((60, 80), bool)
        own[:, :40] = True
        nearer = np.zeros((60, 80), bool)
        nearer[20:40, 30:60] = True
        params = DisocclusionParams(plate_fill=plate_fill)

        import src.content.disocclusion as module

        real = module.inpaint_hole

        def spy(rgb_in, hole, radius, method="telea"):
            seen["method"] = method
            return real(rgb_in, hole, radius, "telea")

        module.inpaint_hole = spy
        try:
            fill_layer_disocclusion(rgb, own, nearer, params, plate=plate)
        finally:
            module.inpaint_hole = real
        return seen.get("method")

    def test_plate_uses_plate_fill(self) -> None:
        self.assertEqual(self._call(plate=True, plate_fill="shiftmap"),
                         "shiftmap")

    def test_mid_layer_ignores_plate_fill(self) -> None:
        self.assertEqual(self._call(plate=False, plate_fill="shiftmap"),
                         "telea")


if __name__ == "__main__":
    unittest.main()
