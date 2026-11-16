"""Tests for intra-layer relief mesh generation (P1)."""

from __future__ import annotations

import unittest

import numpy as np

from src.content.layer_clustering import inter_layer_gap_m, relief_depth_scale
from src.scene.primitives import (
    RELIEF_CLAMP_GAP_FRACTION,
    create_layer_relief_vertical_plane_uv,
    create_vertical_plane_uv,
)


class TestLayerRelief(unittest.TestCase):
    def test_inter_layer_gap_even_spacing(self) -> None:
        gap = inter_layer_gap_m(0.0, -0.35, 3)
        self.assertAlmostEqual(gap, 0.175)

    def test_relief_depth_scale_near_full_far_reduced(self) -> None:
        self.assertAlmostEqual(relief_depth_scale(0.0, 0.3), 1.0)
        self.assertAlmostEqual(relief_depth_scale(1.0, 0.3), 0.3)
        self.assertAlmostEqual(relief_depth_scale(0.5, 0.3), 0.65)

    def test_relief_mesh_curves_not_flat(self) -> None:
        h, w = 64, 64
        depth = np.linspace(0.2, 0.8, w * h, dtype=np.float32).reshape(h, w)
        mask = np.ones((h, w), dtype=bool)
        gap = inter_layer_gap_m(0.0, -0.35, 3)
        mesh = create_layer_relief_vertical_plane_uv(
            1.28,
            0.72,
            depth,
            mask,
            z_base=-0.175,
            depth_lo=0.2,
            depth_hi=0.8,
            mean_depth=0.5,
            depth_norm=0.0,
            inter_layer_gap_m=gap,
            relief_amplitude=0.25,
            segments_x=32,
            segments_y=32,
        )
        z = mesh.vertices.reshape(-1, 5)[:, 2]
        self.assertGreater(float(z.std()), 1e-4)
        self.assertAlmostEqual(float(z.mean()), -0.175, places=2)

    def test_far_layer_has_less_relief_than_near(self) -> None:
        h, w = 48, 48
        grid = np.linspace(0.0, 1.0, w, dtype=np.float32)
        depth = np.tile(grid, (h, 1))
        mask = np.ones((h, w), dtype=bool)
        gap = inter_layer_gap_m(0.0, -0.35, 3)
        common = dict(
            width=1.0,
            height=1.0,
            depth_map=depth,
            layer_mask=mask,
            z_base=-0.175,
            depth_lo=0.0,
            depth_hi=1.0,
            mean_depth=0.5,
            inter_layer_gap_m=gap,
            relief_amplitude=0.25,
            segments_x=24,
            segments_y=24,
        )
        near = create_layer_relief_vertical_plane_uv(**common, depth_norm=0.0)
        far = create_layer_relief_vertical_plane_uv(**common, depth_norm=1.0)
        near_range = float(np.ptp(near.vertices.reshape(-1, 5)[:, 2]))
        far_range = float(np.ptp(far.vertices.reshape(-1, 5)[:, 2]))
        self.assertGreater(near_range, far_range)

    def test_relief_stays_within_clamp(self) -> None:
        h, w = 32, 32
        depth = np.ones((h, w), dtype=np.float32)
        depth[:, :16] = 0.0
        depth[:, 16:] = 1.0
        mask = np.ones((h, w), dtype=bool)
        gap = 0.175
        mesh = create_layer_relief_vertical_plane_uv(
            1.0,
            1.0,
            depth,
            mask,
            z_base=0.0,
            depth_lo=0.0,
            depth_hi=1.0,
            mean_depth=0.5,
            depth_norm=0.0,
            inter_layer_gap_m=gap,
            relief_amplitude=0.9,
            segments_x=16,
            segments_y=16,
        )
        z = mesh.vertices.reshape(-1, 5)[:, 2]
        clamp = RELIEF_CLAMP_GAP_FRACTION * gap
        self.assertLessEqual(float(np.max(np.abs(z))), clamp + 1e-5)

    def test_flat_quad_when_relief_disabled_path(self) -> None:
        flat = create_vertical_plane_uv(1.0, 1.0, z=-0.1)
        z = flat.vertices.reshape(-1, 5)[:, 2]
        self.assertTrue(np.allclose(z, -0.1))


if __name__ == "__main__":
    unittest.main()
