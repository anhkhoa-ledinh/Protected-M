"""Tests for depth-map clustering into discrete alpha-masked layers."""

from __future__ import annotations

import unittest

import numpy as np

import cv2

from src.content.layer_clustering import (
    ATMOSPHERE_FALLBACK,
    LayerEdgeParams,
    LayerStagingParams,
    add_rim_light,
    apply_atmospheric_perspective,
    apply_color_bleed,
    assign_layer_z,
    classify_silhouette_edges,
    cluster_depth_layers,
    cluster_into_layers,
    estimate_atmosphere_color,
    extract_layer_rgba,
    feather_mask_alpha,
    stage_layer_color,
    staging_factors,
)


def _luma_std(rgb: np.ndarray, mask: np.ndarray) -> float:
    """Std-dev of luminance within a mask — a proxy for perceived contrast."""
    lum = (
        0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    ).astype(np.float64)
    return float(lum[mask].std())


def _color_bands() -> np.ndarray:
    rgb = np.zeros((64, 90, 3), dtype=np.uint8)
    rgb[:, 0:30] = (30, 60, 200)    # far band  - saturated blue
    rgb[:, 30:60] = (40, 180, 60)   # mid band  - saturated green
    rgb[:, 60:90] = (220, 80, 40)   # near band - saturated orange
    return rgb


def _plateau3() -> np.ndarray:
    d = np.zeros((64, 90), dtype=np.float32)
    d[:, 0:30] = 0.1
    d[:, 30:60] = 0.5
    d[:, 60:90] = 0.9
    return d


def _mean_saturation(rgb: np.ndarray, mask: np.ndarray) -> float:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    return float(hsv[..., 1][mask].mean())


def _plateau_depth() -> np.ndarray:
    """64x100 depth with five well-separated horizontal bands."""
    values = [0.1, 0.3, 0.5, 0.7, 0.9]
    rows = [np.full((64, 20), v, dtype=np.float32) for v in values]
    return np.concatenate(rows, axis=1)


class TestLayerClustering(unittest.TestCase):
    def test_centroids_sorted_and_labels_in_range(self) -> None:
        depth = _plateau_depth()
        labels, centroids = cluster_depth_layers(depth, n_clusters=5, seed=0)
        self.assertEqual(labels.shape, depth.shape)
        self.assertTrue(np.all(np.diff(centroids) > 0.0))
        self.assertGreaterEqual(int(labels.min()), 0)
        self.assertLess(int(labels.max()), 5)

    def test_five_plateaus_recovered_exactly(self) -> None:
        depth = _plateau_depth()
        labels, _ = cluster_depth_layers(depth, n_clusters=5, seed=0)
        # Each 20-column band should map to a single, distinct cluster id.
        band_ids = [int(np.unique(labels[:, i * 20:(i + 1) * 20]).item()) for i in range(5)]
        self.assertEqual(sorted(band_ids), [0, 1, 2, 3, 4])
        self.assertEqual(band_ids, [0, 1, 2, 3, 4])  # far(0.1)->near(0.9) ascending

    def test_extract_rgba_no_feather_matches_mask(self) -> None:
        rgb = np.full((8, 8, 3), 120, dtype=np.uint8)
        mask = np.zeros((8, 8), dtype=bool)
        mask[2:5, 3:6] = True
        rgba = extract_layer_rgba(rgb, mask, dilate_px=0, blur_px=0)
        self.assertEqual(rgba.shape, (8, 8, 4))
        self.assertTrue(np.array_equal(rgba[..., 3] > 0, mask))
        self.assertTrue(np.array_equal(rgba[..., :3], rgb))

    def test_feather_softens_edges(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[10:30, 10:30] = True
        alpha = feather_mask_alpha(mask, dilate_px=1, blur_px=5)
        # RGB kept full-frame; alpha must contain intermediate ramp values.
        intermediate = alpha[(alpha > 0) & (alpha < 255)]
        self.assertGreater(intermediate.size, 0)
        # Core stays fully opaque, far exterior stays fully transparent.
        self.assertEqual(int(alpha[20, 20]), 255)
        self.assertEqual(int(alpha[0, 0]), 0)

    def test_feather_dilate_grows_coverage(self) -> None:
        mask = np.zeros((40, 40), dtype=bool)
        mask[18:22, 18:22] = True
        no_feather = feather_mask_alpha(mask, dilate_px=0, blur_px=0)
        feathered = feather_mask_alpha(mask, dilate_px=2, blur_px=5)
        # Dilate + blur must expand the non-zero (visible) region.
        self.assertGreater(int((feathered > 0).sum()), int((no_feather > 0).sum()))

    def test_assign_layer_z_monotonic_far_to_near(self) -> None:
        z_far, z_near = -0.35, 0.0
        z0 = assign_layer_z(0, 5, z_near, z_far)
        z_last = assign_layer_z(4, 5, z_near, z_far)
        mids = [assign_layer_z(r, 5, z_near, z_far) for r in range(5)]
        self.assertAlmostEqual(z0, z_far)
        self.assertAlmostEqual(z_last, z_near)
        self.assertTrue(np.all(np.diff(mids) > 0.0))

    def test_cluster_into_layers_disjoint_and_complete(self) -> None:
        depth = _plateau_depth()
        rgb = (np.random.default_rng(1).random((64, 100, 3)) * 255).astype(np.uint8)
        layers = cluster_into_layers(rgb, depth, n_clusters=5, seed=0).layers
        self.assertEqual(len(layers), 5)

        # Back-to-front: mean depth strictly increases (far -> near).
        means = [lyr.mean_depth for lyr in layers]
        self.assertTrue(np.all(np.diff(means) > 0.0))

        # Masks partition the image: disjoint and cover every pixel exactly once.
        stacked = np.stack([lyr.mask for lyr in layers], axis=0).astype(np.int32)
        self.assertTrue(np.array_equal(stacked.sum(axis=0), np.ones(depth.shape, dtype=np.int32)))
        self.assertAlmostEqual(sum(lyr.coverage for lyr in layers), 1.0, places=5)

    def test_layer_depth_bands_do_not_overlap(self) -> None:
        # Clean silhouette boundaries: each band's max depth <= next band's min.
        depth = _plateau_depth()
        rgb = np.zeros((64, 100, 3), dtype=np.uint8)
        layers = cluster_into_layers(rgb, depth, n_clusters=5, seed=0).layers
        for near, far in zip(layers, layers[1:]):
            self.assertLessEqual(near.depth_hi, far.depth_lo + 1e-6)

    def test_estimate_atmosphere_color_is_mask_mean(self) -> None:
        rgb = _color_bands()
        mask = np.zeros((64, 90), dtype=bool)
        mask[:, 0:30] = True
        atmo = estimate_atmosphere_color(rgb, mask)
        self.assertEqual(tuple(round(c) for c in atmo), (30, 60, 200))

    def test_atmosphere_identity_at_zero_depth(self) -> None:
        rgb = _color_bands()
        out = apply_atmospheric_perspective(rgb, 0.0, (200.0, 200.0, 200.0))
        self.assertTrue(np.array_equal(out, rgb))

    def test_atmosphere_desaturates_and_pulls_far_color(self) -> None:
        rgb = np.full((16, 16, 3), (220, 80, 40), dtype=np.uint8)
        atmo = (180.0, 190.0, 200.0)
        graded = apply_atmospheric_perspective(rgb, 1.0, atmo)
        base_sat = _mean_saturation(rgb, np.ones((16, 16), bool))
        far_sat = _mean_saturation(graded, np.ones((16, 16), bool))
        self.assertLess(far_sat, base_sat)
        # Graded color must be closer to the atmosphere than the original.
        d_base = np.linalg.norm(rgb[0, 0].astype(float) - np.array(atmo))
        d_far = np.linalg.norm(graded[0, 0].astype(float) - np.array(atmo))
        self.assertLess(d_far, d_base)

    def test_rim_light_brightens_edge_not_interior(self) -> None:
        rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=bool)
        mask[8:32, 8:32] = True
        out = add_rim_light(rgb, mask, width_px=2, intensity=0.6)
        self.assertEqual(int(out[20, 20, 0]), 100)  # interior untouched
        # Inner silhouette band brightened toward white.
        eroded = cv2.erode(
            mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        rim = (mask) & (eroded == 0)
        self.assertGreater(float(out[..., 0][rim].mean()), 100.0)

    def test_layers_atmospheric_gradient(self) -> None:
        rgb, depth = _color_bands(), _plateau3()
        layers = cluster_into_layers(rgb, depth, n_clusters=3, seed=0).layers
        far, near = layers[0], layers[-1]
        self.assertAlmostEqual(far.depth_norm, 1.0)
        self.assertAlmostEqual(near.depth_norm, 0.0)
        # Far layer more desaturated than the (untouched) near layer.
        far_sat = _mean_saturation(far.rgba[..., :3], far.mask)
        near_sat = _mean_saturation(near.rgba[..., :3], near.mask)
        self.assertLess(far_sat, near_sat)

    def test_staging_factors_midpoint_is_neutral(self) -> None:
        params = LayerStagingParams()
        blend, sat, contrast = staging_factors(0.5, params)
        self.assertAlmostEqual(blend, 0.5 * params.atmosphere_blend_max)
        self.assertAlmostEqual(sat, 1.0)
        self.assertAlmostEqual(contrast, 1.0)

    def test_staging_factors_near_and_far_endpoints(self) -> None:
        params = LayerStagingParams()
        near_blend, near_sat, near_con = staging_factors(0.0, params)
        far_blend, far_sat, far_con = staging_factors(1.0, params)
        # Near: no atmosphere, full saturation, contrast boosted.
        self.assertAlmostEqual(near_blend, 0.0)
        self.assertAlmostEqual(near_sat, 1.0)
        self.assertAlmostEqual(near_con, params.near_contrast_boost)
        # Far: full atmosphere blend, desaturated, contrast reduced.
        self.assertAlmostEqual(far_blend, params.atmosphere_blend_max)
        self.assertAlmostEqual(far_sat, params.far_desaturation)
        self.assertAlmostEqual(far_con, params.far_contrast_reduce)

    def test_stage_disabled_is_identity(self) -> None:
        rgb = _color_bands()
        params = LayerStagingParams(enabled=False)
        out, report = stage_layer_color(rgb, 1.0, (184, 196, 208), params)
        self.assertTrue(np.array_equal(out, rgb))
        self.assertFalse(report.enabled)
        self.assertAlmostEqual(report.atmosphere_blend, 0.0)
        self.assertAlmostEqual(report.saturation_factor, 1.0)
        self.assertAlmostEqual(report.contrast_factor, 1.0)

    def test_stage_far_desaturates_and_pulls_color(self) -> None:
        rgb = np.full((16, 16, 3), (220, 80, 40), dtype=np.uint8)
        atmo = (184.0, 196.0, 208.0)
        params = LayerStagingParams()
        graded, report = stage_layer_color(rgb, 1.0, atmo, params)
        full = np.ones((16, 16), bool)
        self.assertLess(_mean_saturation(graded, full), _mean_saturation(rgb, full))
        d_base = np.linalg.norm(rgb[0, 0].astype(float) - np.array(atmo))
        d_far = np.linalg.norm(graded[0, 0].astype(float) - np.array(atmo))
        self.assertLess(d_far, d_base)
        self.assertAlmostEqual(report.atmosphere_blend, params.atmosphere_blend_max)
        self.assertAlmostEqual(report.saturation_factor, params.far_desaturation)

    def test_stage_near_boosts_contrast(self) -> None:
        rng = np.random.default_rng(3)
        rgb = (rng.random((32, 32, 3)) * 255).astype(np.uint8)
        params = LayerStagingParams()
        near, _ = stage_layer_color(rgb, 0.0, (184, 196, 208), params)
        full = np.ones((32, 32), bool)
        # Nearest layer should be punchier than the source.
        self.assertGreater(_luma_std(near, full), _luma_std(rgb, full))

    def test_estimate_atmosphere_dark_fallback(self) -> None:
        rgb = np.full((8, 8, 3), 5, dtype=np.uint8)  # near-black scene
        mask = np.ones((8, 8), dtype=bool)
        atmo = estimate_atmosphere_color(rgb, mask, dark_fallback=ATMOSPHERE_FALLBACK)
        self.assertEqual(atmo, ATMOSPHERE_FALLBACK)
        # Bright scene keeps the sampled mean (no fallback).
        bright = np.full((8, 8, 3), (30, 60, 200), dtype=np.uint8)
        sampled = estimate_atmosphere_color(
            bright, mask, dark_fallback=ATMOSPHERE_FALLBACK
        )
        self.assertEqual(tuple(round(c) for c in sampled), (30, 60, 200))

    def test_staged_layers_far_hazier_than_near(self) -> None:
        depth = _plateau3()
        # Textured bands (row gradient) so contrast changes are measurable.
        rgb = _color_bands().astype(np.int32)
        grad = np.linspace(-60, 60, 64, dtype=np.int32)[:, None, None]
        rgb = np.clip(rgb + grad, 0, 255).astype(np.uint8)
        layers = cluster_into_layers(
            rgb, depth, n_clusters=3, seed=0, staging=LayerStagingParams()
        ).layers
        far, near = layers[0], layers[-1]
        # Reports attached and ordered far(1.0) -> near(0.0).
        self.assertIsNotNone(far.grade_report)
        self.assertAlmostEqual(far.depth_norm, 1.0)
        self.assertAlmostEqual(near.depth_norm, 0.0)
        # Far layer: more desaturated + lower contrast than near.
        self.assertLess(
            _mean_saturation(far.rgba[..., :3], far.mask),
            _mean_saturation(near.rgba[..., :3], near.mask),
        )
        self.assertLess(
            _luma_std(far.rgba[..., :3], far.mask),
            _luma_std(near.rgba[..., :3], near.mask),
        )

    def test_staging_ab_toggle_changes_output(self) -> None:
        rgb, depth = _color_bands(), _plateau3()
        on = cluster_into_layers(
            rgb, depth, n_clusters=3, seed=0, staging=LayerStagingParams(enabled=True)
        ).layers
        off = cluster_into_layers(
            rgb, depth, n_clusters=3, seed=0, staging=LayerStagingParams(enabled=False)
        ).layers
        # OFF farthest layer is ungraded; ON differs (hazed/desaturated).
        self.assertFalse(np.array_equal(on[0].rgba[..., :3], off[0].rgba[..., :3]))
        self.assertTrue(on[0].grade_report.enabled)
        self.assertFalse(off[0].grade_report.enabled)

    def test_edge_feather_wider_than_base(self) -> None:
        mask = np.zeros((64, 64), dtype=bool)
        mask[20:44, 20:44] = True
        rgb = np.zeros((64, 64, 3), dtype=np.uint8)
        rgb[mask] = (40, 120, 200)
        rgb[~mask] = (220, 200, 180)
        edge_mask = classify_silhouette_edges(mask, rgb, gradient_threshold=0.02)
        base = feather_mask_alpha(mask, dilate_px=1, blur_px=5)
        wide = feather_mask_alpha(
            mask, dilate_px=1, blur_px=5, edge_mask=edge_mask, edge_blur_px=18
        )
        soft_base = int(((base > 10) & (base < 245)).sum())
        soft_wide = int(((wide > 10) & (wide < 245)).sum())
        self.assertGreater(soft_wide, soft_base)

    def test_color_bleed_toward_behind(self) -> None:
        fg = np.full((32, 32, 3), (200, 50, 50), dtype=np.uint8)
        behind = np.full((32, 32, 3), (50, 180, 80), dtype=np.uint8)
        alpha = np.zeros((32, 32), dtype=np.uint8)
        alpha[10:22, 10:22] = 255
        alpha[10, 10:22] = 128
        bled = apply_color_bleed(fg, alpha, behind, bleed_px=3)
        edge_px = bled[10, 16].astype(np.float32)
        fg_px = fg[10, 16].astype(np.float32)
        self.assertLess(float(edge_px[0]), float(fg_px[0]))

    def test_soft_rim_weaker_than_strong(self) -> None:
        rgb = np.full((40, 40, 3), 100, dtype=np.uint8)
        mask = np.zeros((40, 40), dtype=bool)
        mask[8:32, 8:32] = True
        soft = add_rim_light(rgb, mask, width_px=2, intensity=0.3)
        strong = add_rim_light(rgb, mask, width_px=2, intensity=0.6)
        eroded = cv2.erode(
            mask.astype(np.uint8) * 255,
            cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)),
        )
        rim = (mask) & (eroded == 0)
        self.assertLess(float(soft[rim].mean()), float(strong[rim].mean()))

    def test_determinism(self) -> None:
        depth = _plateau_depth()
        labels_a, _ = cluster_depth_layers(depth, n_clusters=5, seed=7)
        labels_b, _ = cluster_depth_layers(depth, n_clusters=5, seed=7)
        self.assertTrue(np.array_equal(labels_a, labels_b))


if __name__ == "__main__":
    unittest.main()
