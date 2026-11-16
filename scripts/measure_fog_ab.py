r"""
PB-069: A/B sương trên lớp xa — 3 mức × 4 ảnh, chỉ ảnh, KHÔNG đổi file config.

Ba mức truyền bằng code: A = 0.35/0.6/0.75 (mặc định đang chạy), B =
0.175/0.8/0.875 (nửa liều), C = tắt ``atmospheric_enabled``. Thứ tự ba số là
``atmosphere_blend_max`` / ``far_desaturation`` / ``far_contrast_reduce``.

Chỉ số, định nghĩa dựng lại từ `MEMORY.md` mục 5 dòng 27/10 (lúc đó đo tay
trên PNG của seg_lab, repo chưa có script), cả hai tỷ lệ tính TRÊN CÙNG tập
pixel **chỉ** thuộc lớp xa: ``contrast_pct`` = 100 × std(luminance lớp) /
std(luminance ảnh gốc); ``saturation_pct`` = 100 × mean(S lớp) / mean(S gốc)
(HSV); ``halo_dL`` dùng lại ``rings_dl`` của ``_layer_report.py``.

Mốc tự kiểm: mức A trên t25 phải ra ≈ 46% và ≈ 39% (số 27/10).

    .\.venv\Scripts\python.exe scripts\measure_fog_ab.py
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

from _fog_metrics import (LEVELS, far_only_mask, nearer_mask,  # noqa: E402
                          ratios, staging_for)
from _layer_report import rings_dl  # noqa: E402
from _real_image_metrics import params_stub, preview_png  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.content.adaptive_k import select_optimal_k  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.image_source import ImageSource  # noqa: E402
from src.content.layer_clustering import cluster_into_layers  # noqa: E402

IMAGES = Path(r"D:\PycharmProjects\test_images")
OUT_DIR = Path(r"D:\test_true3D space\task_done\PB-069_fog_ab")
NAMES = ("t25", "t22", "t24", "t4")
FIELDS = ["image", "level", "far_px", "contrast_pct", "saturation_pct",
          "halo_dL", "K", "error"]





def run_level(name, rgb, depth, cs, k, level, out_dir):
    """
    Một hàng CSV + một PNG cho (ảnh, mức).

    Gọi ``cluster_into_layers`` trực tiếp như ``measure_layers.py``: chỉ đường
    đó trả về ``DepthLayer`` có ``mask`` và ``rgba`` — thứ ``rings_dl`` và phép
    đo tỷ lệ cần. ``_apply_layers`` chỉ để lại ``_LayerRenderable`` (có
    ``alpha``, không có ``mask``/``rgba``) nên không dùng được ở đây.
    """
    row = {key: "" for key in FIELDS}
    row["image"], row["level"] = name, level
    try:
        stub = params_stub(cs)
        res = cluster_into_layers(
            rgb, depth, n_clusters=k, z_near_m=cs.layer_z_near_m,
            z_far_m=cs.layer_z_far_m, seed=int(cs.layer_seed),
            feather_dilate_px=int(cs.layer_feather_dilate_px),
            feather_blur_px=int(cs.layer_feather_blur_px),
            rim_light_px=int(cs.layer_rim_px),
            rim_intensity=float(cs.rim_light_intensity),
            staging=staging_for(cs, level), semantic=stub._semantic_params(),
            edge=stub._edge_params(), disocclusion=stub._disocclusion_params())
        layers = res.layers
        row["K"] = len(layers)
        far = far_only_mask(layers)
        row["far_px"] = int(far.sum())
        contrast, saturation = ratios(layers[0], rgb, far)
        row["contrast_pct"] = round(contrast, 2)
        row["saturation_pct"] = round(saturation, 2)
        nearer = nearer_mask(layers)
        row["halo_dL"] = round(
            float(rings_dl(layers[0], nearer, 0.0, 3.0, 6.0, 12.0, False)), 2)
        preview_png(layers, out_dir / f"{name}_{level}.png")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="PB-069 fog A/B (images only)")
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
        path = folder / f"{name}.png"
        rgb = ImageSource(path).tick(0.0)
        depth = estimator.estimate(rgb)          # MỘT lần, dùng cho cả 3 mức
        # K cũng chốt một lần: staging chỉ đổi MÀU, không đổi cách cắt cụm, nên
        # cả ba mức phải có cùng K mới so được với nhau.
        k, _ = select_optimal_k(
            depth, k_min=int(cs.adaptive_k_min), k_max=int(cs.adaptive_k_max),
            seed=int(cs.layer_seed), k_policy=str(cs.k_policy),
            gvf_threshold=float(cs.gvf_threshold),
            log_silhouette_comparison=False)
        for level in LEVELS:
            row = run_level(name, rgb, depth, cs, k, level, out_dir)
            rows.append(row)
            print(f"{name} {level}: far_px={row['far_px']} "
                  f"contrast={row['contrast_pct']}% "
                  f"sat={row['saturation_pct']}% halo_dL={row['halo_dL']} "
                  f"K={row['K']} {row['error']}", flush=True)
    csv_path = out_dir / "fog_ab.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} hang -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
