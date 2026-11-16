#!/usr/bin/env python3
"""
Verify GVF-threshold K selection + save K=2 vs K=3 A/B layer composites.

Composites are orthographic painter's-algorithm stacks (fixed frontal eye /
straight-on view — equivalent to eye at screen center looking along -Z).
"""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.content.adaptive_k import select_optimal_k
from src.content.depth_estimator import DepthEstimator
from src.content.layer_clustering import cluster_into_layers
from src.config.schema import ContentSettings
from src.content.semantic_depth import SemanticDepthParams

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT_DIR = PROJECT_ROOT / "artifacts" / "gvf_ab"
EYE_NOTE = "frontal orthographic composite (eye at screen center, looking along -Z)"


def _composite_layers(layers) -> np.ndarray:
    """Back-to-front alpha composite of DepthLayer RGBA textures."""
    h, w = layers[0].rgba.shape[:2]
    out = np.zeros((h, w, 3), dtype=np.float32)
    for layer in layers:  # far -> near
        rgba = layer.rgba.astype(np.float32)
        a = rgba[..., 3:4] / 255.0
        out = rgba[..., :3] * a + out * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def _semantic() -> SemanticDepthParams:
    d = ContentSettings()
    return SemanticDepthParams(
        enabled=True,
        depth_band_count=d.semantic_depth_bands,
        similarity_threshold=d.semantic_depth_similarity,
        separation_offset=d.semantic_depth_separation,
        max_segment_variance=d.semantic_max_segment_variance,
        min_segment_pixels=d.semantic_min_segment_pixels,
        subdivide_above_pixels=d.semantic_subdivide_pixels,
        max_work_dim=d.semantic_max_work_dim,
    )


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    estimator = DepthEstimator("depth-anything-small")
    if not estimator.available:
        raise SystemExit(estimator.load_error)

    images = [
        ("t15", Path(r"D:\PycharmProjects\test_images\t15.png")),
        ("t14", Path(r"D:\PycharmProjects\test_images\t14.png")),
        ("t5", Path(r"D:\PycharmProjects\test_images\t5.png")),
        ("t1", Path(r"D:\PycharmProjects\test_images\t1.png")),
    ]
    semantic = _semantic()
    print(f"A/B eye pose: {EYE_NOTE}")
    print(f"Output dir: {OUT_DIR}")

    for stem, path in images:
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        depth = estimator.estimate(rgb)
        depth_ms = (time.perf_counter() - t0) * 1000.0

        t1 = time.perf_counter()
        k, report = select_optimal_k(depth, k_min=2, k_max=4, seed=0, gvf_threshold=0.90)
        k_ms = (time.perf_counter() - t1) * 1000.0

        t2 = time.perf_counter()
        result = cluster_into_layers(rgb, depth, n_clusters=k, seed=0, semantic=semantic)
        clu_ms = (time.perf_counter() - t2) * 1000.0
        sem = result.timing_ms.get("semantic_adjust", 0.0)
        clu = result.timing_ms.get("clustering", 0.0)
        print(
            f"Load timing [{path.name}]: depth_inference={depth_ms:.1f}ms | "
            f"k_selection={k_ms:.1f}ms | semantic_adjust={sem:.1f}ms | "
            f"clustering={clu:.1f}ms | mesh_build=n/a | total={depth_ms+k_ms+clu_ms:.1f}ms"
        )
        print(
            f"  selected K={k} gvf_var={{{', '.join(f'{kk}:{vv:.3f}' for kk,vv in sorted(report.gvf_variance.items()))}}} "
            f"kmeans_fits={report.kmeans_fit_count}"
        )

        # A/B for images that move 2→3 under the new rule
        if stem in ("t15", "t14"):
            for forced_k, tag in ((2, "k2_before"), (3, "k3_after")):
                forced = cluster_into_layers(
                    rgb, depth, n_clusters=forced_k, seed=0, semantic=semantic
                )
                comp = _composite_layers(forced.layers)
                out = OUT_DIR / f"{stem}_{tag}.png"
                cv2.imwrite(str(out), cv2.cvtColor(comp, cv2.COLOR_RGB2BGR))
                print(f"  saved {out} (forced K={forced_k}, layers={len(forced.layers)})")


if __name__ == "__main__":
    main()
