"""PB-050: ảnh rất lớn bị thu nhỏ ngay lúc nạp; ảnh cỡ demo giữ nguyên."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.content.image_source import ImageSource


class ImageSizeCapTest(unittest.TestCase):
    """
    Ảnh 5823×2700 (OneDrive, 16/10) làm app chết 3/3 lần khi dựng lớp.

    Đỏ trên code trước PB-050: ``ImageSource`` giữ nguyên 5823×2700.
    """

    def test_a_5823x2700_photo_is_capped_and_a_demo_image_is_not(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hcpd_cap_") as tmp:
            big, small = Path(tmp) / "big.png", Path(tmp) / "small.png"
            cv2.imwrite(str(big), np.zeros((2700, 5823, 3), np.uint8))
            cv2.imwrite(str(small), np.zeros((900, 1200, 3), np.uint8))
            capped, demo = ImageSource(big), ImageSource(small)
        self.assertEqual((capped.width, capped.height), (2048, 950))
        self.assertEqual(capped.tick(0.0).shape, (950, 2048, 3))
        self.assertAlmostEqual(capped.aspect_ratio, 5823 / 2700, delta=0.01)
        self.assertEqual((demo.width, demo.height), (1200, 900))


if __name__ == "__main__":
    unittest.main()
