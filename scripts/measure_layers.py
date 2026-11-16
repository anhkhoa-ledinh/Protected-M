"""
Đo trạng thái lớp TRƯỚC khi sửa — PB-045 Phase L0. Chỉ đọc, không đụng ``src/``.
Sao y đường của app: ``DepthEstimator`` → ``select_optimal_k`` trên depth **thô**
(content_plane.py:801, đó là PB-022) → ``cluster_into_layers`` với đúng bộ tham số
``_apply_layers`` truyền. Preset qua dòng lệnh; KHÔNG sửa content.json.
Phần in bảng nằm ở ``_layer_report.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from _layer_report import report  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.content.adaptive_k import select_optimal_k  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.layer_clustering import (  # noqa: E402
    adjust_depth_semantic, cluster_into_layers)

CONTENT_JSON = PROJECT_ROOT / "assets" / "config" / "content.json"


def parse_args(argv):
    p = argparse.ArgumentParser(description="Đo lớp trước khi sửa (L0)")
    p.add_argument("images", nargs="+")
    p.add_argument("--eye-distance", type=float, nargs="+",
                   default=[325.0, 380.0, 450.0, 600.0])
    p.add_argument("--out", default=str(PROJECT_ROOT / "artifacts" / "layers"))
    p.add_argument("--plane-z", type=float)
    p.add_argument("--plane-width", type=float)
    p.add_argument("--z-far", type=float)
    p.add_argument("--design-distance", type=float)
    return p.parse_args(argv)


def build_stub(cs):
    """``ContentPlaneRenderer`` không gọi ``__init__`` — chỉ mượn 4 hàm params."""
    from src.renderer.content_plane import ContentPlaneRenderer
    s = ContentPlaneRenderer.__new__(ContentPlaneRenderer)
    s._content = cs
    s._disocclusion_runtime_enabled = bool(cs.disocclusion_fill_enabled)
    s._mist_runtime_enabled = bool(cs.mist_enabled)
    s._atmospheric_runtime_enabled = bool(cs.atmospheric_enabled)
    s._relief_runtime_enabled = bool(cs.relief_enabled)
    return s


def run_pipeline(path, cs, stub, estimator):
    """Trả (rgb, depth, layers, gvf_thô, gvf_adjusted, timing)."""
    rgb = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    depth = estimator.estimate(rgb)
    kw = dict(k_min=int(cs.adaptive_k_min), k_max=int(cs.adaptive_k_max),
              seed=int(cs.layer_seed), k_policy=str(cs.k_policy),
              gvf_threshold=float(cs.gvf_threshold), log_silhouette_comparison=False)
    k, rep = select_optimal_k(depth, **kw)          # depth THÔ, như app
    adj, _ = adjust_depth_semantic(rgb, depth, stub._semantic_params())
    _, rep_adj = select_optimal_k(adj, **kw)
    res = cluster_into_layers(
        rgb, depth, n_clusters=k, z_near_m=cs.layer_z_near_m,
        z_far_m=cs.layer_z_far_m, seed=int(cs.layer_seed),
        feather_dilate_px=int(cs.layer_feather_dilate_px),
        feather_blur_px=int(cs.layer_feather_blur_px),
        rim_light_px=int(cs.layer_rim_px), rim_intensity=float(cs.rim_light_intensity),
        staging=stub._staging_params(), semantic=stub._semantic_params(),
        edge=stub._edge_params(), disocclusion=stub._disocclusion_params())
    return rgb, depth, res.layers, rep.gvf, rep_adj.gvf, res.timing_ms


def main(argv=None):
    args = parse_args(argv)
    cs = ContentSettings.from_dict(json.loads(CONTENT_JSON.read_text(encoding="utf-8")))
    if args.plane_z is not None:
        cs.plane_position_m = [0.0, 0.0, args.plane_z]
    if args.plane_width is not None:
        cs.plane_width_m = args.plane_width
    if args.z_far is not None:
        cs.layer_z_far_m = args.z_far
    stub = build_stub(cs)
    print("params dung cho clustering:")
    for n in ("_staging_params", "_semantic_params", "_edge_params",
              "_disocclusion_params"):
        print(f"  {n}() = {getattr(stub, n)()}")
    estimator = DepthEstimator(str(cs.depth_model))   # nap model MOT lan cho ca lot
    for img in args.images:
        path = Path(img)
        report(path, *run_pipeline(path, cs, stub, estimator), cs, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
