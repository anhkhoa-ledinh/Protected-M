"""Unit tests for webcam preview HUD layout."""

from __future__ import annotations

import unittest

from src.renderer.webcam_preview import _preview_rect_ndc


class TestWebcamPreviewLayout(unittest.TestCase):
    def test_bottom_right_within_ndc(self) -> None:
        x0, y0, x1, y1 = _preview_rect_ndc(1280, 720, 16 / 9, "bottom-right", 0.18, 16)
        self.assertLess(x0, x1)
        self.assertLess(y0, y1)
        self.assertGreaterEqual(x1, 0.5)
        self.assertLessEqual(y0, 0.0)

    def test_aspect_preserved_height(self) -> None:
        _, y0, _, y1 = _preview_rect_ndc(1280, 720, 16 / 9, "bottom-right", 0.18, 16)
        pw = 1280 * 0.18
        ph = pw / (16 / 9)
        expected_h_ndc = 2.0 * ph / 720
        self.assertAlmostEqual(y1 - y0, expected_h_ndc, places=4)


if __name__ == "__main__":
    unittest.main()
