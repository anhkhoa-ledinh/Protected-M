#!/usr/bin/env python3
"""One-shot K-policy + relief diagnostics for test images (no GUI)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.content.adaptive_k import select_optimal_k
from src.content.depth_estimator import DepthEstimator
from src.content.layer_clustering import cluster_into_layers, inter_layer_gap_m
from src.scene.primitives import create_layer_relief_vertical_plane_uv

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("diagnostic")


def run_image(path: Path, estimator: DepthEstimator) -> None:
    logger.info("=== %s ===", path.name)
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        logger.error("Failed to load %s", path)
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    depth = estimator.estimate(rgb)
    k, _report = select_optimal_k(depth, k_min=2, k_max=4, seed=0, k_policy="richness")
    result = cluster_into_layers(rgb, depth, n_clusters=k, seed=0)
    gap = inter_layer_gap_m(0.0, -0.35, len(result.layers))
    resolution = 96
    for idx, layer in enumerate(result.layers):
        create_layer_relief_vertical_plane_uv(
            1.28,
            0.72,
            depth,
            layer.mask,
            z_base=layer.z_m,
            depth_lo=layer.depth_lo,
            depth_hi=layer.depth_hi,
            mean_depth=layer.mean_depth,
            depth_norm=layer.depth_norm,
            inter_layer_gap_m=gap,
            relief_amplitude=0.25,
            relief_far_falloff=0.3,
            relief_smooth_sigma=1.5,
            segments_x=resolution,
            segments_y=resolution,
            layer_index=idx + 1,
            layer_count=len(result.layers),
        )


def main() -> None:
    images = [
        Path(r"D:\PycharmProjects\test_images\t15.png"),
        Path(r"D:\PycharmProjects\test_images\t14.png"),
    ]
    estimator = DepthEstimator("depth-anything-small")
    if not estimator.available:
        raise SystemExit(f"Depth model unavailable: {estimator.load_error}")
    for path in images:
        if path.exists():
            run_image(path, estimator)
        else:
            logger.warning("Skip missing image: %s", path)


if __name__ == "__main__":
    main()
