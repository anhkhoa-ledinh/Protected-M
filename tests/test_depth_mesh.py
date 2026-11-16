"""Tests for depth-displaced content mesh generation."""

from __future__ import annotations

import unittest

import numpy as np

from src.content.depth_estimator import normalize_depth_map
from src.scene.primitives import create_displaced_vertical_plane_uv, sample_depth_map


class TestDepthMesh(unittest.TestCase):
    def test_sample_depth_map_bilinear(self) -> None:
        depth = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        self.assertAlmostEqual(sample_depth_map(depth, 0.0, 0.0), 0.0)
        self.assertAlmostEqual(sample_depth_map(depth, 1.0, 1.0), 1.0)

    def test_displaced_mesh_near_vertices_closer_to_viewer(self) -> None:
        depth = np.zeros((64, 64), dtype=np.float32)
        depth[:, :32] = 1.0
        depth[:, 32:] = 0.0
        mesh = create_displaced_vertical_plane_uv(
            1.28, 0.72, depth, depth_scale=0.2, segments_x=32, segments_y=16
        )
        verts = mesh.vertices.reshape(-1, 5)
        left_z = verts[verts[:, 0] < 0.0, 2]
        right_z = verts[verts[:, 0] > 0.0, 2]
        self.assertGreater(float(np.mean(left_z)), float(np.mean(right_z)))

    def test_displaced_mesh_has_more_than_one_quad(self) -> None:
        depth = np.linspace(0.0, 1.0, 16 * 16, dtype=np.float32).reshape(16, 16)
        mesh = create_displaced_vertical_plane_uv(
            1.0, 1.0, depth, depth_scale=0.1, segments_x=8, segments_y=8
        )
        self.assertGreater(mesh.vertex_count, 6)
        self.assertEqual(len(mesh.vertices) // 5, 9 * 9)

    def test_displaced_mesh_shows_curvature_not_flat(self) -> None:
        # Smooth radial depth bump -> surface must curve, not stay planar.
        grid = np.linspace(-1.0, 1.0, 64, dtype=np.float32)
        gx, gy = np.meshgrid(grid, grid)
        depth = np.clip(1.0 - np.sqrt(gx * gx + gy * gy), 0.0, 1.0).astype(np.float32)
        scale = 0.15
        mesh = create_displaced_vertical_plane_uv(
            1.28, 0.96, depth, depth_scale=scale, segments_x=64, segments_y=64
        )
        z = mesh.vertices.reshape(-1, 5)[:, 2]
        self.assertGreater(float(z.std()), 1e-3)
        self.assertLessEqual(float(z.max() - z.min()), scale + 1e-4)
        self.assertGreater(float(z.max() - z.min()), scale * 0.5)

    def test_normalize_depth_map_resizes(self) -> None:
        raw = np.ones((100, 200), dtype=np.float32) * 5.0
        raw[40:60, 80:120] = 100.0
        out = normalize_depth_map(raw, 64, 128)
        self.assertEqual(out.shape, (64, 128))
        self.assertGreaterEqual(float(out.max()), 0.9)
        self.assertLessEqual(float(out.min()), 0.1)


if __name__ == "__main__":
    unittest.main()
