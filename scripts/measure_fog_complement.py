r"""
PB-071: A/B sương B, B− và B−K (màu bù) trên t25/t22/t24/t4. Chỉ ảnh.

Không đổi file config — cường độ truyền bằng code. Ba biến thể:

| biến thể | s | blend | saturation | contrast |
|---|---|---|---|---|
| B | 0.50 | 0.1750 | 0.80 | 0.8750 |
| B− | 0.35 | 0.1225 | 0.86 | 0.9125 |
| B−K | 0.35 | — | — | 0.9125 | ← blend + desaturate thay bằng màu bù

B−K dùng **đúng** bước contrast của B− (gọi lại ``stage_layer_color`` với
blend = 0 và saturation = 1, nên dùng chính code của app), rồi thay hai bước
sau bằng ``complement_mix``; ``t`` chia đôi cho tới khi bão hoà trung bình của
lớp bằng bão hoà lớp đó ở B−.

    .\.venv\Scripts\python.exe scripts\measure_fog_complement.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _fog_complement import (complement_mix, mean_lab_l,  # noqa: E402
                             mean_saturation, solve_t_for_saturation, strength)
from _fog_metrics import nearer_mask, ratios  # noqa: E402
from _layer_report import rings_dl  # noqa: E402
from _real_image_metrics import params_stub, preview_png  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.content.adaptive_k import select_optimal_k  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.image_source import ImageSource  # noqa: E402
from src.content.layer_clustering import (LayerStagingParams,  # noqa: E402
                                          cluster_into_layers)

IMAGES = Path(r"D:\PycharmProjects\test_images")
OUT_DIR = Path(r"D:\test_true3D space\task_done\PB-071_fog_complement")
NAMES = ("t25", "t22", "t24", "t4")
VARIANTS = {"B": 0.50, "B-": 0.35, "B-K": 0.35}
FIELDS = ["image", "variant", "layer", "contrast_pct", "saturation_pct",
          "lab_L", "halo_dL", "t", "error"]


def staging(cs, s, contrast_only=False):
    """``LayerStagingParams`` của cường độ ``s``; ``contrast_only`` cho B−K."""
    blend, sat, contrast = strength(s)
    return LayerStagingParams(
        enabled=True,
        atmosphere_blend_max=0.0 if contrast_only else blend,
        far_desaturation=1.0 if contrast_only else sat,
        near_contrast_boost=float(cs.near_contrast_boost),
        far_contrast_reduce=contrast)


def build(rgb, depth, cs, k, stage):
    """``cluster_into_layers`` với một bộ staging; trả danh sách lớp."""
    stub = params_stub(cs)
    res = cluster_into_layers(
        rgb, depth, n_clusters=k, z_near_m=cs.layer_z_near_m,
        z_far_m=cs.layer_z_far_m, seed=int(cs.layer_seed),
        feather_dilate_px=int(cs.layer_feather_dilate_px),
        feather_blur_px=int(cs.layer_feather_blur_px),
        rim_light_px=int(cs.layer_rim_px),
        rim_intensity=float(cs.rim_light_intensity),
        staging=stage, semantic=stub._semantic_params(),
        edge=stub._edge_params(), disocclusion=stub._disocclusion_params())
    return res.layers


def rows_for(name, variant, layers, rgb, ts):
    """Một hàng mỗi lớp."""
    out = []
    for i, layer in enumerate(layers):
        row = {key: "" for key in FIELDS}
        row["image"], row["variant"], row["layer"] = name, variant, i
        mask = layer.mask.astype(bool)
        contrast, saturation = ratios(layer, rgb, mask)
        row["contrast_pct"] = round(contrast, 2)
        row["saturation_pct"] = round(saturation, 2)
        row["lab_L"] = round(mean_lab_l(layer.rgba[..., :3], mask), 2)
        nearer = nearer_mask(layers) if i == 0 else np.zeros(mask.shape, bool)
        dl = rings_dl(layer, nearer, 0.0, 3.0, 6.0, 12.0, False) if i == 0 \
            else float("nan")
        row["halo_dL"] = "" if not np.isfinite(dl) else round(float(dl), 2)
        row["t"] = "" if ts is None else round(float(ts[i]), 4)
        out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="PB-071 fog complement A/B")
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
        bminus = build(rgb, depth, cs, k, staging(cs, VARIANTS["B-"]))
        for variant, s in VARIANTS.items():
            if variant != "B-K":
                layers, ts = build(rgb, depth, cs, k, staging(cs, s)), None
            else:
                layers = build(rgb, depth, cs, k,
                               staging(cs, s, contrast_only=True))
                ts = []
                for i, layer in enumerate(layers):
                    mask = layer.mask.astype(bool)
                    target = mean_saturation(bminus[i].rgba[..., :3], mask)
                    t = solve_t_for_saturation(layer.rgba[..., :3], mask, target)
                    ts.append(t)
                    layer.rgba[..., :3] = complement_mix(
                        layer.rgba[..., :3], t)
            rows.extend(rows_for(name, variant, layers, rgb, ts))
            preview_png(layers, out_dir / f"{name}_{variant}.png")
            shown = "" if ts is None else " t=" + ",".join(f"{v:.3f}" for v in ts)
            print(f"{name} {variant:4s} K={len(layers)}{shown}", flush=True)
    csv_path = out_dir / "fog_k.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} hang -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
