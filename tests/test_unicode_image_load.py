"""Tests for Unicode-safe OpenCV file loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.content.cv_io import imread_bgr
from src.content.image_source import ImageSource


class TestUnicodeImageLoad(unittest.TestCase):
    def test_imread_bgr_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hcpd_img_") as tmp:
            folder = Path(tmp) / "Hình ảnh"
            folder.mkdir()
            image_path = folder / "1 - Copy.PNG"
            bgr = np.zeros((48, 64, 3), dtype=np.uint8)
            bgr[:, :, 2] = 200
            ok, encoded = cv2.imencode(".png", bgr)
            self.assertTrue(ok)
            image_path.write_bytes(encoded.tobytes())

            loaded = imread_bgr(image_path)
            self.assertIsNotNone(loaded)
            assert loaded is not None
            self.assertEqual(loaded.shape[:2], (48, 64))

            source = ImageSource(image_path)
            self.assertEqual(source.width, 64)
            self.assertEqual(source.height, 48)


if __name__ == "__main__":
    unittest.main()
