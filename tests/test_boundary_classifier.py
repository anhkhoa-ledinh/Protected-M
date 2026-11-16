"""Unit tests for depth-gradient boundary / weld-weight classifier."""

from __future__ import annotations

import unittest

import numpy as np

from src.content.boundary_classifier import (
    DEPTH_EDGE_HIGH,
    DEPTH_EDGE_LOW,
    BoundaryClassifierResult,
    classify_layer_seams,
    depth_gradient_magnitude,
    render_weldmap,
    weld_weight_from_depth_grad,
)


class TestBoundaryClassifier(unittest.TestCase):
    def test_depth_step_yields_hard_weld(self) -> None:
        depth = np.zeros((64, 64), dtype=np.float32)
        depth[:, 32:] = 1.0
        masks = [depth < 0.5, depth >= 0.5]
        # High-contrast RGB edge along the same seam.
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[:, 32:] = 255
        result = classify_layer_seams(depth, masks, rgb)
        seam = np.isfinite(result.weld_weight)
        self.assertTrue(bool(seam.any()))
        mean_w = float(result.weld_weight[seam].mean())
        self.assertLess(mean_w, 0.25)
        self.assertEqual(len(result.seams), 1)
        self.assertGreater(result.seams[0].hard_frac, 0.5)

    def test_smooth_ramp_yields_soft_weld(self) -> None:
        # Gentle horizontal ramp (Δdepth=0.1 over 64 px) — no sharp discontinuity.
        xs = np.linspace(0.40, 0.50, 64, dtype=np.float32)
        depth = np.tile(xs, (64, 1))
        masks = [depth < 0.45, depth >= 0.45]
        rgb = np.full((64, 64, 3), 128, dtype=np.uint8)
        result = classify_layer_seams(depth, masks, rgb)
        seam = np.isfinite(result.weld_weight)
        self.assertTrue(bool(seam.any()))
        mean_w = float(result.weld_weight[seam].mean())
        self.assertGreater(mean_w, 0.75)

    def test_rgb_edge_without_depth_edge_stays_soft(self) -> None:
        # Flat depth (should weld) but strong RGB color boundary.
        depth = np.full((48, 48), 0.4, dtype=np.float32)
        # Artificial two-layer masks meeting in the middle (same depth).
        masks = [
            np.zeros((48, 48), dtype=bool),
            np.zeros((48, 48), dtype=bool),
        ]
        masks[0][:, :24] = True
        masks[1][:, 24:] = True
        rgb = np.zeros((48, 48, 3), dtype=np.uint8)
        rgb[:, 24:, 0] = 255  # pink/red vs black — strong RGB edge
        result = classify_layer_seams(depth, masks, rgb)
        seam = np.isfinite(result.weld_weight)
        self.assertTrue(bool(seam.any()))
        # Depth gradient ~0 → RGB alone must NOT force hard.
        mean_w = float(result.weld_weight[seam].mean())
        self.assertGreater(mean_w, 0.85)

    def test_ramp_thresholds(self) -> None:
        g = np.array(
            [0.0, DEPTH_EDGE_LOW, (DEPTH_EDGE_LOW + DEPTH_EDGE_HIGH) * 0.5, DEPTH_EDGE_HIGH, 1.0],
            dtype=np.float32,
        )
        w = weld_weight_from_depth_grad(g)
        self.assertAlmostEqual(float(w[0]), 1.0, places=5)
        self.assertAlmostEqual(float(w[1]), 1.0, places=5)
        self.assertAlmostEqual(float(w[3]), 0.0, places=5)
        self.assertAlmostEqual(float(w[4]), 0.0, places=5)
        self.assertGreater(float(w[2]), 0.0)
        self.assertLess(float(w[2]), 1.0)

    def test_gradient_normalization_shape(self) -> None:
        depth = np.linspace(0, 1, 100, dtype=np.float32).reshape(10, 10)
        g = depth_gradient_magnitude(depth)
        self.assertEqual(g.shape, depth.shape)
        self.assertGreaterEqual(float(g.min()), 0.0)


if __name__ == "__main__":
    unittest.main()


class WeldmapRenderingTest(unittest.TestCase):
    """render_weldmap must not bias the widened band toward 'soft'."""

    def test_widened_band_uses_nearest_seam_not_local_maximum(self) -> None:
        """
        A hard seam next to a soft one must stay red.

        The band used to be produced with cv2.dilate on the weight map, which
        is a local MAXIMUM: it propagated the softest weight outward and
        painted almost every weldmap green regardless of the classifier.
        """
        h = w = 41
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        weld = np.full((h, w), np.nan, dtype=np.float32)
        weld[20, 10] = 0.0   # hard
        weld[20, 30] = 1.0   # soft, far enough not to be the nearest pixel
        result = BoundaryClassifierResult(
            weld_weight=weld,
            depth_grad_norm=np.zeros((h, w), dtype=np.float32),
            seams=[],
        )
        out = render_weldmap(rgb, result, seam_alpha=1.0)

        # Immediately beside the hard seam pixel: red channel dominant.
        self.assertGreater(int(out[20, 11, 0]), int(out[20, 11, 1]))
        # Immediately beside the soft seam pixel: green channel dominant.
        self.assertGreater(int(out[20, 29, 1]), int(out[20, 29, 0]))
