"""Tests for GVF-threshold adaptive depth-layer count selection."""

from __future__ import annotations

import unittest

import numpy as np

from src.content.adaptive_k import GVF_THRESHOLD, select_optimal_k


class TestAdaptiveK(unittest.TestCase):
    def test_flat_image_returns_k_min(self) -> None:
        depth = np.full((64, 64), 0.42, dtype=np.float32)
        k, report = select_optimal_k(depth, k_min=2, k_max=4)
        self.assertEqual(k, 2)
        self.assertTrue(report.flat_fallback)
        self.assertEqual(report.k_policy, "gvf")

    def test_gvf_threshold_picks_smallest_passing_k(self) -> None:
        # Three well-separated depth bands → GVF rises quickly with K.
        depth = np.zeros((80, 120), dtype=np.float32)
        depth[:, :40] = 0.12
        depth[:, 40:80] = 0.45
        depth[:, 80:] = 0.88
        k, report = select_optimal_k(
            depth, k_min=2, k_max=4, seed=0, gvf_threshold=0.90
        )
        self.assertIn(k, (2, 3, 4))
        self.assertGreaterEqual(report.gvf_variance[k], 0.90 - 1e-6)
        if k > 2:
            self.assertLess(report.gvf_variance[k - 1], 0.90)

    def test_gvf_variance_reported_for_each_candidate(self) -> None:
        depth = np.linspace(0.1, 0.9, 100, dtype=np.float32).reshape(10, 10)
        _, report = select_optimal_k(depth, k_min=2, k_max=4, seed=1)
        self.assertEqual(set(report.gvf_variance.keys()), {2, 3, 4})
        self.assertEqual(report.scores, {})  # silhouette off by default

    def test_gvf_variance_is_non_decreasing_in_k(self) -> None:
        depth = np.linspace(0.05, 0.95, 256, dtype=np.float32).reshape(16, 16)
        _, report = select_optimal_k(depth, k_min=2, k_max=4, seed=0)
        gv = [report.gvf_variance[k] for k in sorted(report.gvf_variance)]
        self.assertEqual(gv, sorted(gv))

    def test_none_reach_threshold_returns_k_max(self) -> None:
        # Nearly continuous depth — high threshold forces k_max.
        depth = np.linspace(0.0, 1.0, 64 * 64, dtype=np.float32).reshape(64, 64)
        k, report = select_optimal_k(
            depth, k_min=2, k_max=4, seed=0, gvf_threshold=0.999
        )
        self.assertEqual(k, 4)
        self.assertEqual(report.gvf_threshold, 0.999)

    def test_default_threshold_constant(self) -> None:
        self.assertAlmostEqual(GVF_THRESHOLD, 0.90)

    def test_richness_alias_maps_to_gvf_rule(self) -> None:
        depth = np.zeros((40, 60), dtype=np.float32)
        depth[:, :20] = 0.1
        depth[:, 20:40] = 0.5
        depth[:, 40:] = 0.9
        k, report = select_optimal_k(
            depth, k_min=2, k_max=4, seed=0, k_policy="richness"
        )
        self.assertEqual(report.k_policy, "gvf")
        self.assertIn(k, (2, 3, 4))


if __name__ == "__main__":
    unittest.main()
