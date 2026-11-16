#!/usr/bin/env python3
"""Report gvf_var/sil table + load timing for decision (no K-rule change)."""

from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.content.adaptive_k import select_optimal_k
from src.content.depth_estimator import DepthEstimator
from src.content.layer_clustering import cluster_into_layers
from src.config.schema import ContentSettings
from src.content.semantic_depth import SemanticDepthParams

logging.basicConfig(level=logging.INFO, format="%(message)s")


IMAGES = [
    ("t15 (starry village)", Path(r"D:\PycharmProjects\test_images\t15.png")),
    ("t14 (clock/rails)", Path(r"D:\PycharmProjects\test_images\t14.png")),
    ("t5 (clear fg subject)", Path(r"D:\PycharmProjects\test_images\t5.png")),
    ("t1 (traffic-light)", Path(r"D:\PycharmProjects\test_images\t1.png")),
]


def main() -> None:
    estimator = DepthEstimator("depth-anything-small")
    if not estimator.available:
        raise SystemExit(estimator.load_error)
    defaults = ContentSettings()
    semantic = SemanticDepthParams(
        enabled=True,
        depth_band_count=defaults.semantic_depth_bands,
        similarity_threshold=defaults.semantic_depth_similarity,
        separation_offset=defaults.semantic_depth_separation,
        max_segment_variance=defaults.semantic_max_segment_variance,
        min_segment_pixels=defaults.semantic_min_segment_pixels,
        subdivide_above_pixels=defaults.semantic_subdivide_pixels,
        max_work_dim=defaults.semantic_max_work_dim,
    )

    rows = []
    for label, path in IMAGES:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        t0 = time.perf_counter()
        depth = estimator.estimate(rgb)
        depth_ms = (time.perf_counter() - t0) * 1000.0
        t1 = time.perf_counter()
        k, report = select_optimal_k(depth, k_min=2, k_max=4, seed=0, k_policy="richness")
        k_ms = (time.perf_counter() - t1) * 1000.0
        t2 = time.perf_counter()
        result = cluster_into_layers(rgb, depth, n_clusters=k, seed=0, semantic=semantic)
        clu_ms = (time.perf_counter() - t2) * 1000.0
        sem = result.timing_ms.get("semantic_adjust", 0.0)
        clu = result.timing_ms.get("clustering", 0.0)
        total = depth_ms + k_ms + clu_ms
        print(
            f"Load timing [{path.name}]: depth_inference={depth_ms:.1f}ms | "
            f"k_selection={k_ms:.1f}ms | semantic_adjust={sem:.1f}ms | "
            f"clustering={clu:.1f}ms | mesh_build=n/a | total={total:.1f}ms"
        )
        print(
            f"  kmeans_fits={report.kmeans_fit_count} samples={report.sample_count} "
            f"gvf_var={ {kk: round(vv,3) for kk,vv in sorted(report.gvf_variance.items())} } "
            f"sil={ {kk: round(vv,3) for kk,vv in sorted(report.scores.items())} } K={k}"
        )
        rows.append(
            (
                label,
                report.gvf_variance.get(2, 0),
                report.gvf_variance.get(3, 0),
                report.gvf_variance.get(4, 0),
                report.scores.get(2, 0),
                report.scores.get(3, 0),
                report.scores.get(4, 0),
                k,
            )
        )

    print("\n| Image | gvf_var K=2 | gvf_var K=3 | gvf_var K=4 | sil K=2 | sil K=3 | sil K=4 | current selected K |")
    print("|-------|-------------|-------------|-------------|---------|---------|---------|--------------------|")
    for r in rows:
        print(
            f"| {r[0]} | {r[1]:.3f} | {r[2]:.3f} | {r[3]:.3f} | "
            f"{r[4]:.3f} | {r[5]:.3f} | {r[6]:.3f} | {r[7]} |"
        )


if __name__ == "__main__":
    main()
