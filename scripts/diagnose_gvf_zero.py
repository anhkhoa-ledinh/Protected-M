#!/usr/bin/env python3
"""Diagnostic only: depth histogram + true WCSS/TSS GVF per K (read-only).

HISTORICAL: written to explain why the pre-GVF code reported gvf = 0.0 for
every K (it returned a silhouette *delta*, which is 0 when silhouettes tie).
That rule is gone — ``select_optimal_k`` now reports true ``1 - WCSS/TSS``.
The histogram / WCSS / silhouette dumps below are still useful for inspecting
how a depth map clusters, which is why this was repaired instead of dropped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config.schema import ContentSettings
from src.content.adaptive_k import (
    _assign_labels_1d,
    _downsample_depth_values,
    _silhouette_1d,
    select_optimal_k,
)
from src.content.depth_estimator import DepthEstimator
from src.content.layer_clustering import _kmeans_1d
from src.content.semantic_depth import SemanticDepthParams, adjust_depth_semantic


def _hist20(values: np.ndarray) -> list[tuple[float, int]]:
    hist, edges = np.histogram(values, bins=20, range=(0.0, 1.0))
    return [
        (float(0.5 * (edges[i] + edges[i + 1])), int(hist[i]))
        for i in range(len(hist))
    ]


def _wcss_tss(values: np.ndarray, centroids: np.ndarray, labels: np.ndarray) -> dict:
    mean = float(values.mean())
    tss = float(np.sum((values - mean) ** 2))
    wcss = 0.0
    for j, c in enumerate(centroids):
        members = values[labels == j]
        if members.size:
            wcss += float(np.sum((members - float(c)) ** 2))
    ratio = wcss / tss if tss > 1e-12 else float("nan")
    true_gvf = 1.0 - ratio if tss > 1e-12 else float("nan")
    return {
        "mean": mean,
        "std": float(values.std()),
        "len": int(values.size),
        "tss": tss,
        "wcss": wcss,
        "wcss_tss": ratio,
        "true_gvf": true_gvf,
    }


def diagnose(name: str, path: Path, estimator: DepthEstimator) -> None:
    print(f"\n========== {name} ==========")
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        print(f"FAILED to load {path}")
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    raw_depth = estimator.estimate(rgb)

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
    post_depth, sem_report = adjust_depth_semantic(rgb, raw_depth, semantic)

    # Pipeline fact: select_optimal_k receives RAW depth (pre-semantic).
    k_input = raw_depth
    samples = _downsample_depth_values(k_input)

    print("\n--- H3: depth fed to K-selection ---")
    print("source: RAW Depth-Anything (pre-semantic); semantic runs AFTER K-selection")
    raw_flat = raw_depth.reshape(-1).astype(np.float64)
    post_flat = post_depth.reshape(-1).astype(np.float64)
    print(
        f"raw:   unique~={len(np.unique(np.round(raw_flat, 4)))} "
        f"(rounded 4dp) exact_unique={len(np.unique(raw_flat))} "
        f"mean={raw_flat.mean():.4f} std={raw_flat.std():.4f}"
    )
    print(
        f"post:  unique~={len(np.unique(np.round(post_flat, 4)))} "
        f"(rounded 4dp) exact_unique={len(np.unique(post_flat))} "
        f"mean={post_flat.mean():.4f} std={post_flat.std():.4f} "
        f"semantic segs={sem_report.num_segments} flat={sem_report.segments_flattened}"
    )
    print(f"K-selection samples: len={samples.size} mean={samples.mean():.4f} std={samples.std():.4f}")
    print(f"K-selection hist20 (bin_center, count): {_hist20(samples)}")
    print(f"raw hist20: {_hist20(raw_flat)}")
    print(f"post-semantic hist20: {_hist20(post_flat)}")

    print("\n--- H1: _kmeans_1d at K=3 on K-selection samples ---")
    centroids = _kmeans_1d(samples, 3, seed=0)
    labels = _assign_labels_1d(samples, centroids)
    unique = np.unique(labels)
    counts = [int((labels == j).sum()) for j in range(3)]
    print(f"centers: {centroids.tolist()}")
    print(f"per-cluster counts: {counts}")
    print(f"unique label count: {int(unique.size)} labels={unique.tolist()}")

    print("\n--- H2: true WCSS/TSS GVF at K=3 (same array = samples) ---")
    stats = _wcss_tss(samples, centroids, labels)
    for key, val in stats.items():
        print(f"  {key}={val}")
    print("  confirmation: TSS and WCSS both computed on `samples` (downsampled raw depth)")

    print("\n--- Silhouette at K=2,3,4 (same samples) ---")
    for k in (2, 3, 4):
        c = _kmeans_1d(samples, k, seed=0)
        lab = _assign_labels_1d(samples, c)
        s = _silhouette_1d(samples, lab)
        true = _wcss_tss(samples, c, lab)
        print(
            f"  K={k}: sil_seed0={s:.6f} "
            f"true_gvf={true['true_gvf']:.6f} wcss/tss={true['wcss_tss']:.6f} "
            f"centers={[round(x,4) for x in c.tolist()]}"
        )

    print("\n--- Current code: true GVF = 1 - WCSS/TSS ---")
    chosen, report = select_optimal_k(k_input, k_min=2, k_max=4, seed=0)
    print(f"  AdaptiveKReport: chosen={chosen} gvf_variance={report.gvf_variance}")
    print(f"  threshold={report.gvf_threshold} rejections={report.rejections}")
    print(
        "  Values here should match the true_gvf column above (small drift is "
        "expected: select_optimal_k takes the best of n_init restarts)."
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
            diagnose(path.name, path, estimator)
        else:
            print(f"Skip missing: {path}")


if __name__ == "__main__":
    main()
