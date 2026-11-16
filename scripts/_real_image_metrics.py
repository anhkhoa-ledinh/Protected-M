r"""
PB-059 (2): phần đo cho ``measure_real_images.py`` — stub renderer, seam, preview.

Tách ra để mỗi file dưới 150 dòng, theo đúng lối ``_layer_report.py`` /
``_depth_ab_common.py`` đã dùng trong ``scripts/``.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import cv2
import numpy as np

from src.content.adaptive_k import select_optimal_k
from src.content.boundary_classifier import classify_layer_seams
from src.content.layer_clustering import cluster_into_layers
from src.renderer import content_plane as cp

_DUMMY_MESH = type("M", (), {"STRIDE": 5, "__init__": lambda s, d: None})


def build_stub(cs, rgb):
    """``ContentPlaneRenderer`` không gọi ``__init__`` — mượn đường vẽ thật."""
    h, w = rgb.shape[:2]
    r = cp.ContentPlaneRenderer.__new__(cp.ContentPlaneRenderer)
    r._content, r._layers, r._mist = cs, [], None
    r._position = tuple(cs.plane_position_m)
    r._screen_w_mm, r._screen_h_mm = 344.0, 194.0
    r._plane_width = float(cs.plane_width_m)
    r._plane_height = 0.0
    r._tex_width, r._tex_height = w, h
    r._layer_shader, r._depth_worker, r._adaptive_k_report = object(), None, None
    r._disocclusion_runtime_enabled = bool(cs.disocclusion_fill_enabled)
    r._mist_runtime_enabled = bool(cs.mist_enabled)
    r._atmospheric_runtime_enabled = bool(cs.atmospheric_enabled)
    r._relief_runtime_enabled = bool(cs.relief_enabled)
    r._last_depth_rgb = rgb
    return r


def apply_layers(stub, depth, name):
    """Gọi ``_apply_layers`` THẬT với OpenGL bị stub (như ``measure_props.py``)."""
    with mock.patch.object(cp, "TexturedMesh", _DUMMY_MESH), \
         mock.patch.object(cp.ContentPlaneRenderer, "_create_rgba_texture",
                           lambda self, a: 0):
        stub._apply_layers(depth, name)


def params_stub(cs):
    """Stub chỉ để mượn 4 hàm ``_*_params`` của renderer."""
    stub = cp.ContentPlaneRenderer.__new__(cp.ContentPlaneRenderer)
    stub._content = cs
    for name in ("_disocclusion_runtime_enabled", "_mist_runtime_enabled",
                 "_atmospheric_runtime_enabled", "_relief_runtime_enabled"):
        setattr(stub, name, False)
    return stub


def seam_hard_frac(depth, rgb, cs, layer_method="kmeans"):
    """
    ``(frac, layers)``: trung bình ``hard_frac`` có trọng số ``length_px``.

    Đúng công thức Thành dùng ở ``task_done\\Work2_Phase 02\\source\\
    benchmark_metrics.py`` dòng 398–404, nhưng gọi thẳng hàm của repo
    (``classify_layer_seams``) chứ không copy script. ``frac`` là ``None`` khi
    ảnh không có seam nào (K = 1).
    """
    stub = params_stub(cs)
    k, _ = select_optimal_k(
        depth, k_min=int(cs.adaptive_k_min), k_max=int(cs.adaptive_k_max),
        seed=int(cs.layer_seed), k_policy=str(cs.k_policy),
        gvf_threshold=float(cs.gvf_threshold), log_silhouette_comparison=False)
    res = cluster_into_layers(
        rgb, depth, n_clusters=k, z_near_m=cs.layer_z_near_m,
        z_far_m=cs.layer_z_far_m, seed=int(cs.layer_seed),
        semantic=stub._semantic_params(), edge=stub._edge_params(),
        layer_method=str(layer_method))
    out = classify_layer_seams(depth, [l.mask for l in res.layers], rgb)
    total = sum(s.length_px for s in out.seams)
    if total <= 0:
        return None, res.layers
    frac = sum(s.hard_frac * s.length_px for s in out.seams) / total
    return float(frac), res.layers


def preview_png(layers, path: Path):
    """Ghép các lớp back-to-front thành một PNG xem nhanh."""
    if not layers:
        return
    h, w = layers[0].mask.shape
    canvas = np.zeros((h, w, 3), dtype=np.float32)
    for layer in layers:
        rgba = layer.rgba.astype(np.float32)
        alpha = rgba[:, :, 3:4] / 255.0
        canvas = canvas * (1.0 - alpha) + rgba[:, :, :3] * alpha
    cv2.imwrite(str(path), cv2.cvtColor(
        np.clip(canvas, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
