"""Tests for Unicode-safe OpenCV file loading."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from src.content.image_source import ImageSource
from src.content.media_gate import import_photo


class TestUnicodeImageLoad(unittest.TestCase):
    def test_gate_and_source_accept_a_unicode_path(self) -> None:
        with tempfile.TemporaryDirectory(prefix="hcpd_img_") as tmp:
            folder = Path(tmp) / "Hình ảnh"
            folder.mkdir()
            image_path = folder / "1 - Copy.PNG"
            bgr = np.zeros((48, 64, 3), dtype=np.uint8)
            bgr[:, :, 2] = 200
            ok, encoded = cv2.imencode(".png", bgr)
            self.assertTrue(ok)
            image_path.write_bytes(encoded.tobytes())

            clean = import_photo(image_path, Path(tmp) / "clean")
            self.assertEqual((clean.height, clean.width), (48, 64))

            source = ImageSource(clean)
            self.assertEqual(source.width, 64)
            self.assertEqual(source.height, 48)


if __name__ == "__main__":
    unittest.main()
