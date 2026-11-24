r"""
PB-070b: A/B ảnh giữa ``layer_method`` = "kmeans" và "kmeans_vote".

Không đổi file config — truyền ``layer_method`` bằng code. Ghi ảnh ghép lớp và
từng lớp riêng, kèm CSV: K, tỷ lệ pixel đổi nhãn, số vùng P10.

    .\.venv\Scripts\python.exe scripts\measure_layer_method_ab.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _real_image_metrics import params_stub, preview_png  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.content.adaptive_k import select_optimal_k  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.image_source import ImageSource  # noqa: E402
from src.content.layer_clustering import cluster_into_layers  # noqa: E402
from src.content.layer_regions import region_masks  # noqa: E402

IMAGES = Path(r"D:\PycharmProjects\test_images")
OUT_DIR = Path(r"D:\test_true3D space\task_done\SPIKE-15_kmeans_vote\ab")
NAMES = ("t25", "t22", "t24", "t4")
METHODS = ("kmeans", "kmeans_vote")
FIELDS = ["image", "layer_method", "K", "regions_p10", "relabel_pct",
          "coverage_pct", "error"]


def build(rgb, depth, cs, k, method):
    """``cluster_into_layers`` với một ``layer_method``; trả danh sách lớp."""
    stub = params_stub(cs)
    res = cluster_into_layers(
        rgb, depth, n_clusters=k, z_near_m=cs.layer_z_near_m,
        z_far_m=cs.layer_z_far_m, seed=int(cs.layer_seed),
        feather_dilate_px=int(cs.layer_feather_dilate_px),
        feather_blur_px=int(cs.layer_feather_blur_px),
        rim_light_px=int(cs.layer_rim_px),
        rim_intensity=float(cs.rim_light_intensity),
        staging=stub._staging_params(), semantic=stub._semantic_params(),
        edge=stub._edge_params(), disocclusion=stub._disocclusion_params(),
        layer_method=method)
    return res.layers


def run_one(name, rgb, depth, cs, k, method, out_dir, relabel_pct, n_regions):
    """Một hàng CSV + ảnh ghép + ảnh từng lớp."""
    row = {key: "" for key in FIELDS}
    row["image"], row["layer_method"] = name, method
    row["regions_p10"] = n_regions
    row["relabel_pct"] = relabel_pct
    try:
        layers = build(rgb, depth, cs, k, method)
        row["K"] = len(layers)
        row["coverage_pct"] = " / ".join(
            f"{100.0 * float(l.mask.mean()):.0f}" for l in layers)
        preview_png(layers, out_dir / f"{name}_{method}.png")
        for i, layer in enumerate(layers):
            cv2.imwrite(str(out_dir / f"{name}_{method}_L{i}.png"),
                        cv2.cvtColor(layer.rgba, cv2.COLOR_RGBA2BGRA))
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="PB-070b layer_method A/B")
    ap.add_argument("--images", default=str(IMAGES))
    ap.add_argument("--out", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    folder, out_dir = Path(args.images), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    cs = ContentSettings.from_dict(json.loads(
        (ROOT / "assets/config/content.demo.json").read_text(encoding="utf-8")))
    estimator = DepthEstimator(str(cs.depth_model), str(cs.depth_backend))
    rows = []
    for name in NAMES:
        rgb = ImageSource(import_photo(folder / f"{name}.png")).tick(0.0)
        depth = estimator.estimate(rgb)
        k, _ = select_optimal_k(
            depth, k_min=int(cs.adaptive_k_min), k_max=int(cs.adaptive_k_max),
            seed=int(cs.layer_seed), k_policy=str(cs.k_policy),
            gvf_threshold=float(cs.gvf_threshold),
            log_silhouette_comparison=False)
        # Đo riêng tỷ lệ đổi nhãn và số vùng, để điền cho cả hai hàng.
        from src.content.layer_clustering import cluster_depth_layers
        from src.content.layer_vote import vote_labels

        labels, centroids = cluster_depth_layers(depth, k,
                                                 seed=int(cs.layer_seed))
        masks = region_masks(rgb, depth)
        _, frac = vote_labels(labels, centroids, depth, masks)
        for method in METHODS:
            row = run_one(name, rgb, depth, cs, k, method, out_dir,
                          round(100.0 * frac, 2), len(masks))
            rows.append(row)
            print(f"{name} {method:12s} K={row['K']} regions={len(masks)} "
                  f"relabel={row['relabel_pct']}% cov={row['coverage_pct']} "
                  f"{row['error']}", flush=True)
    csv_path = out_dir / "layer_method_ab.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} hang -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
