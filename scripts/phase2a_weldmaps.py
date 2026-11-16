#!/usr/bin/env python3
"""Phase 2a: generate weldmap overlays (no Z-welding)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.content.adaptive_k import select_optimal_k
from src.content.boundary_classifier import classify_layer_seams, render_weldmap
from src.content.depth_estimator import DepthEstimator
from src.content.layer_clustering import cluster_into_layers
from src.config.schema import ContentSettings
from src.content.semantic_depth import SemanticDepthParams

logging.basicConfig(level=logging.INFO, format="%(message)s")

OUT = PROJECT_ROOT / "artifacts" / "phase2"

IMAGES = [
    ("t15", Path(r"D:\PycharmProjects\test_images\t15.png")),
    ("t16", Path(r"D:\PycharmProjects\test_images\t16.png")),  # cherry blossom
    ("t14", Path(r"D:\PycharmProjects\test_images\t14.png")),
]


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
    OUT.mkdir(parents=True, exist_ok=True)
    estimator = DepthEstimator("depth-anything-small")
    if not estimator.available:
        raise SystemExit(estimator.load_error)
    semantic = _semantic()

    for stem, path in IMAGES:
        if not path.exists():
            print(f"MISSING {path}")
            continue
        print(f"\n=== {stem} ({path.name}) ===")
        bgr = cv2.imread(str(path))
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        depth = estimator.estimate(rgb)  # raw pre-semantic
        k, _ = select_optimal_k(depth, k_min=2, k_max=4, seed=0, gvf_threshold=0.90)
        result = cluster_into_layers(rgb, depth, n_clusters=k, seed=0, semantic=semantic)
        # Classifier uses RAW depth + clustered masks (masks from post-semantic clustering).
        report = classify_layer_seams(
            depth, [ly.mask for ly in result.layers], rgb
        )
        weldmap = render_weldmap(rgb, report)
        out = OUT / f"{stem}_weldmap.png"
        cv2.imwrite(str(out), cv2.cvtColor(weldmap, cv2.COLOR_RGB2BGR))
        print(f"saved {out}  K={k} seams={len(report.seams)}")
        for s in report.seams:
            print(
                f"Seam {s.layer_a}|{s.layer_b}: hard={s.hard_frac*100:.0f}% "
                f"soft={s.soft_frac*100:.0f}% mixed={s.mixed_frac*100:.0f}% "
                f"length={s.length_px}px"
            )


if __name__ == "__main__":
    main()
