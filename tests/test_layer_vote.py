"""PB-070b: key ``layer_method`` và bước bầu theo vùng (SPIKE-15, P18b)."""

from __future__ import annotations

import unittest

import numpy as np

from src.config.schema import ContentSettings
from src.content.layer_vote import MIN_PX, TAU_F, centroid_gaps, vote_labels


def _scene():
    """Ảnh 20×20: nửa trên nhãn 0, nửa dưới nhãn 1; tâm cụm 0.2 và 0.8."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[10:] = 1
    depth = np.where(labels == 0, 0.2, 0.8).astype(np.float32)
    centroids = np.array([0.2, 0.8], dtype=np.float64)
    return labels, centroids, depth


class TestDefaultIsUnchanged(unittest.TestCase):
    """Mặc định của schema phải là k-means thuần."""

    def test_schema_default_is_kmeans(self) -> None:
        self.assertEqual(ContentSettings().layer_method, "kmeans")
        self.assertEqual(ContentSettings.from_dict({}).layer_method, "kmeans")

    def test_demo_config_uses_the_vote_after_the_user_pick(self) -> None:
        """
        30/10 [XÁC NHẬN]: người dùng chọn ``kmeans_vote`` cho chế độ demo.

        Trước ngày đó test này đòi ``content.demo.json`` phải là ``kmeans``
        (chưa có lựa chọn bằng mắt). Chỉ **demo** đổi; mặc định của schema và
        ``content.json`` vẫn là k-means thuần.
        """
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "assets" / "config" / "content.demo.json")
                          .read_text(encoding="utf-8"))
        self.assertEqual(data["layer_method"], "kmeans_vote")

    def test_config_can_select_the_vote(self) -> None:
        loaded = ContentSettings.from_dict({"layer_method": "kmeans_vote"})
        self.assertEqual(loaded.layer_method, "kmeans_vote")


class TestVoteKeepsKmeansOutsideAcceptedRegions(unittest.TestCase):
    """Điều kiện cốt lõi: ngoài vùng được nhận, nhãn phải y k-means."""

    def test_pixels_outside_accepted_masks_are_untouched(self) -> None:
        labels, centroids, depth = _scene()
        mask = np.zeros(labels.shape, dtype=bool)
        mask[8:12, :] = True
        out, _ = vote_labels(labels, centroids, depth, [mask])
        self.assertTrue(np.array_equal(out[~mask], labels[~mask]))

    def test_no_masks_is_identity(self) -> None:
        labels, centroids, depth = _scene()
        out, frac = vote_labels(labels, centroids, depth, [])
        self.assertTrue(np.array_equal(out, labels))
        self.assertEqual(frac, 0.0)

    def test_accepted_region_becomes_one_label(self) -> None:
        labels, centroids, depth = _scene()
        mask = np.zeros(labels.shape, dtype=bool)
        mask[:16, :5] = True
        depth = depth.copy()
        depth[mask] = 0.2
        out, frac = vote_labels(labels, centroids, depth, [mask])
        self.assertEqual(np.unique(out[mask]).size, 1)
        self.assertGreater(frac, 0.0)

    def test_region_failing_share_or_spread_is_untouched(self) -> None:
        labels, centroids, depth = _scene()
        half = np.zeros(labels.shape, dtype=bool)
        half[9:11, :] = True                      # f* = 0.5 < 0.6
        out, _ = vote_labels(labels, centroids, depth, [half])
        self.assertTrue(np.array_equal(out, labels))

        wide = np.zeros(labels.shape, dtype=bool)
        wide[:16, :5] = True                      # f* đạt, nhưng s > g
        spread = depth.copy()
        spread[wide] = np.linspace(0.0, 0.9, int(wide.sum())).astype(np.float32)
        out2, _ = vote_labels(labels, centroids, spread, [wide])
        self.assertTrue(np.array_equal(out2, labels))

    def test_tiny_regions_are_ignored(self) -> None:
        labels, centroids, depth = _scene()
        mask = np.zeros(labels.shape, dtype=bool)
        mask.flat[:MIN_PX - 1] = True
        out, frac = vote_labels(labels, centroids, depth, [mask])
        self.assertTrue(np.array_equal(out, labels))
        self.assertEqual(frac, 0.0)

    def test_frozen_constants(self) -> None:
        self.assertEqual(TAU_F, 0.6)
        self.assertEqual(MIN_PX, 64)

    def test_single_centroid_gap_is_infinite(self) -> None:
        gaps = centroid_gaps(np.array([0.5]))
        self.assertEqual(gaps.size, 1)
        self.assertTrue(np.isinf(gaps[0]))


if __name__ == "__main__":
    unittest.main()
