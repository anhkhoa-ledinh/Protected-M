"""Tests for semantic-guided depth adjustment before layer clustering."""

from __future__ import annotations

import unittest

import numpy as np

from src.content.semantic_depth import (
    _flatten_segments,
    _separate_similar_adjacent_segments,
)

from src.content.layer_clustering import cluster_into_layers
from src.content.semantic_depth import SemanticDepthParams, adjust_depth_semantic


def _village_scene() -> tuple[np.ndarray, np.ndarray]:
    """Sky + ridge + village share similar depth; road is near."""
    h, w = 120, 180
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    depth = np.zeros((h, w), dtype=np.float32)

    rgb[0 : h // 3, :] = (140, 180, 220)
    rgb[h // 3 : 2 * h // 3, :] = (90, 110, 80)
    rgb[2 * h // 3 :, :] = (50, 45, 40)

    depth[0 : h // 3, :] = 0.14
    depth[h // 3 : h // 2, :] = 0.15
    depth[h // 2 : 2 * h // 3, w // 5 : 4 * w // 5] = 0.16
    depth[2 * h // 3 :, :] = 0.72
    return rgb, depth


def _portrait_scene() -> tuple[np.ndarray, np.ndarray]:
    """Face + body share depth with small internal variance."""
    h, w = 100, 80
    rgb = np.full((h, w, 3), 180, dtype=np.uint8)
    depth = np.full((h, w), 0.55, dtype=np.float32)
    # Background farther.
    depth[0:20, :] = 0.20
    rgb[0:20, :] = (120, 150, 200)
    # Person blob with slight hair/cheek noise.
    person = np.zeros((h, w), dtype=bool)
    person[25:85, 28:52] = True
    depth[person] = 0.54 + np.random.default_rng(0).normal(0, 0.008, size=person.sum())
    rgb[person] = (210, 170, 150)
    return rgb, depth


class TestSemanticDepth(unittest.TestCase):
    def test_separates_similar_depth_bands(self) -> None:
        rgb, depth = _village_scene()
        adjusted, report = adjust_depth_semantic(rgb, depth, SemanticDepthParams())
        self.assertGreater(report.segment_pairs_separated, 0)
        sky = adjusted[0 : depth.shape[0] // 3, :]
        village = adjusted[
            depth.shape[0] // 2 : 2 * depth.shape[0] // 3,
            depth.shape[1] // 5 : 4 * depth.shape[1] // 5,
        ]
        self.assertLess(float(sky.mean()), float(village.mean()) - 0.03)

    def test_portrait_stays_single_cluster(self) -> None:
        rgb, depth = _portrait_scene()
        params = SemanticDepthParams(max_segment_variance=0.02)
        adjusted, _ = adjust_depth_semantic(rgb, depth, params)
        person = (depth > 0.4) & (depth < 0.65)
        spread = float(adjusted[person].max() - adjusted[person].min())
        self.assertLess(spread, 0.04)

    def test_clustering_splits_village_with_semantic_on(self) -> None:
        rgb, depth = _village_scene()
        on = cluster_into_layers(
            rgb,
            depth,
            n_clusters=3,
            seed=0,
            semantic=SemanticDepthParams(similarity_threshold=0.05),
        ).layers
        sky_row, village_row = 5, depth.shape[0] // 2 + 5
        col = depth.shape[1] // 2
        sky_layer = next(lyr.index for lyr in on if lyr.mask[sky_row, col])
        village_layer = next(lyr.index for lyr in on if lyr.mask[village_row, col])
        self.assertNotEqual(sky_layer, village_layer)


if __name__ == "__main__":
    unittest.main()


class FlattenVarianceThresholdTest(unittest.TestCase):
    """max_segment_variance must actually gate flattening."""

    def _two_segments(self):
        depth = np.zeros((10, 20), dtype=np.float32)
        segments = np.zeros((10, 20), dtype=np.int32)
        segments[:, :10] = 1
        segments[:, 10:] = 2
        depth[:, :10] = 0.5                                  # variance 0
        depth[:, 10:] = np.linspace(0.1, 0.9, 10, dtype=np.float32)  # high variance
        return depth, segments

    def test_high_variance_segment_is_left_alone(self) -> None:
        depth, segments = self._two_segments()
        adjusted, flattened = _flatten_segments(depth, segments, max_variance=0.02)
        # Coherent segment collapsed to its median...
        self.assertTrue(np.allclose(adjusted[:, :10], 0.5))
        # ...structured one keeps its gradient.
        np.testing.assert_allclose(adjusted[:, 10:], depth[:, 10:])
        self.assertEqual(flattened, 1)

    def test_threshold_changes_the_result(self) -> None:
        """
        Regression: both branches used to compute the same median, so the
        threshold did nothing and `flattened` always equalled the segment count.
        """
        depth, segments = self._two_segments()
        strict, n_strict = _flatten_segments(depth, segments, max_variance=0.02)
        loose, n_loose = _flatten_segments(depth, segments, max_variance=999.0)
        self.assertFalse(
            np.array_equal(strict, loose), "max_segment_variance has no effect"
        )
        self.assertEqual(n_strict, 1)
        self.assertEqual(n_loose, 2)


class SeparationHorizontalSpanTest(unittest.TestCase):
    """The separation rule must not slice vertically extended objects."""

    def _stacked(self, span_fraction: float):
        """Two segments stacked vertically, each spanning `span_fraction` of width."""
        h, w = 40, 100
        depth = np.full((h, w), 0.5, dtype=np.float32)
        segments = np.zeros((h, w), dtype=np.int32)
        cols = max(1, int(w * span_fraction))
        left = (w - cols) // 2
        segments[:20, left:left + cols] = 1     # upper part
        segments[20:, left:left + cols] = 2     # lower part, same depth
        return depth, segments

    def test_wide_landscape_bands_are_still_separated(self) -> None:
        depth, segments = self._stacked(0.95)
        adjusted, separated = _separate_similar_adjacent_segments(
            depth, segments,
            similarity_threshold=0.06, separation_offset=0.08,
            min_vertical_gap=0.10, min_horizontal_span=0.6,
        )
        self.assertGreater(separated, 0)
        upper = float(adjusted[segments == 1].mean())
        lower = float(adjusted[segments == 2].mean())
        self.assertLess(upper, lower, "top of frame should end up farther")

    def test_narrow_figure_is_not_split(self) -> None:
        depth, segments = self._stacked(0.25)   # a standing person, not a band
        adjusted, separated = _separate_similar_adjacent_segments(
            depth, segments,
            similarity_threshold=0.06, separation_offset=0.08,
            min_vertical_gap=0.10, min_horizontal_span=0.6,
        )
        self.assertEqual(separated, 0, "torso and legs were pushed apart")
        np.testing.assert_allclose(adjusted, depth)

    def test_span_zero_restores_old_behaviour(self) -> None:
        depth, segments = self._stacked(0.25)
        _, separated = _separate_similar_adjacent_segments(
            depth, segments,
            similarity_threshold=0.06, separation_offset=0.08,
            min_vertical_gap=0.10, min_horizontal_span=0.0,
        )
        self.assertGreater(separated, 0)
