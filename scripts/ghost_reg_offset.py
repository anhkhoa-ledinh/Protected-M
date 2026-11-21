r"""
TASK_ghost_ab_1111 Phase 1 — reg_offset_px: lệch đăng ký giữa hai lớp kề nhau,
tại MỘT mắt (0, 0, 600), toàn màn hình. KHÔNG mở GL — chỉ đọc bộ lớp đã nướng
(cache, không đổi theo biến thể vì overscan_mode/overscan_design_eye_mm/
layer_z_far_m không nằm trong ``bake.BAKE_KEYS``) và dùng ĐÚNG các hàm hình
học của renderer (``layer_geometry``, ``pad_content_scale``,
``assign_layer_z``) — không dựng lại công thức.

Định nghĩa: lấy 20 điểm trên đường bao (alpha ~ biên) của lớp GẦN hơn, tính
toạ độ (u, v) của từng điểm trong khung ảnh gốc (0..1, dùng chung cho mọi lớp
vì mặt nạ đến từ CÙNG một lượt nướng). Với cùng (u, v) đó, suy ra vị trí thế
giới qua hình học của lớp GẦN và qua hình học của lớp XA kế tiếp — hai công
thức dùng chung ``layer_geometry`` nhưng khác ``z_m``/``centre``/hệ số co nội
dung. Chiếu cả hai về pixel thiết kế qua ``project_to_design`` (eye cố định),
rồi đo khoảng cách. Khoảng cách này > 0 nghĩa là CÙNG một điểm ảnh gốc xuất
hiện ở hai chỗ khác nhau trên màn — đúng định nghĩa "ghost" của kịch bản.
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

from layer_review import PROFILES, review_set  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.content.bake import bake  # noqa: E402
from src.content.layer_clustering import assign_layer_z  # noqa: E402
from src.renderer.content_plane import (SCREEN_Z_MM,  # noqa: E402
                                        ContentPlaneRenderer, layer_geometry,
                                        pad_content_scale)
from src.game.viewport_filter import project_to_design  # noqa: E402

N_POINTS = 20
SEED = 0


class _RendererStub:
    """Đủ cho ``ContentPlaneRenderer._overscan_scale`` (chỉ đọc ``_content``)."""

    def __init__(self, content: ContentSettings) -> None:
        self._content = content


def boundary_points(mask: np.ndarray, n: int, seed: int = SEED):
    """``n`` điểm (row, col) trên biên của ``mask`` (dilate XOR erode)."""
    mask_u8 = np.where(mask, np.uint8(255), np.uint8(0))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = (cv2.dilate(mask_u8, kernel) > 0) & (cv2.erode(mask_u8, kernel) == 0)
    rows, cols = np.nonzero(boundary)
    if rows.size == 0:
        return []
    rng = np.random.default_rng(seed)
    idx = rng.choice(rows.size, size=min(n, rows.size), replace=False)
    return [(int(rows[i]), int(cols[i])) for i in idx]


def world_pos_mm(content: ContentSettings, plane_z_m: float, layer_z_m: float,
                 base_w_m: float, base_h_m: float, screen_w_mm: float,
                 screen_h_mm: float, u: float, v: float, pad_mode: bool):
    """Vị trí thế giới (mm) của điểm ``(u, v)`` trên quad của MỘT lớp."""
    stub = _RendererStub(content)
    overscan_scale = ContentPlaneRenderer._overscan_scale.__get__(stub)
    geom = layer_geometry(overscan_scale, content, plane_z_m, layer_z_m,
                          base_w_m, base_h_m, screen_w_mm, screen_h_mm)
    x_content_m = (u - 0.5) * base_w_m
    y_content_m = (0.5 - v) * base_h_m
    if pad_mode:
        t_plane_mm = SCREEN_Z_MM - float(plane_z_m) * 1000.0
        s = pad_content_scale(geom.t_mm, t_plane_mm,
                              content.overscan_design_distance_mm)
    else:
        s = geom.scale
    x_m = geom.centre_x_m + x_content_m * s
    y_m = geom.centre_y_m + y_content_m * s
    z_m = float(plane_z_m) + float(layer_z_m)
    return (x_m * 1000.0, y_m * 1000.0, z_m * 1000.0)


def reg_offsets_for(image_path: Path, content: ContentSettings,
                    screen_w_mm: float, screen_h_mm: float):
    """Danh sách khoảng cách (px thiết kế) của mọi điểm biên, mọi cặp lớp kề."""
    result = bake(image_path, content, ROOT)
    layers = result.layers
    count = len(layers)
    for rank, layer in enumerate(layers):
        layer.z_m = assign_layer_z(rank, count, content.layer_z_near_m,
                                   content.layer_z_far_m)
    base_w_m = float(content.plane_width_m)
    base_h_m = base_w_m  # ty le anh: chieu cao suy tu aspect that cua texture
    h_px, w_px = layers[0].mask.shape
    base_h_m = base_w_m * (h_px / max(w_px, 1))
    plane_z_m = float(content.plane_position_m[2])
    pad_mode = str(content.overscan_mode).lower().strip() == "pad"
    eye_mm = (0.0, 0.0, 600.0)
    offsets = []
    for i in range(count - 1):
        far_layer, near_layer = layers[i], layers[i + 1]
        points = boundary_points(near_layer.mask, N_POINTS)
        for row, col in points:
            u = col / max(w_px - 1, 1)
            v = row / max(h_px - 1, 1)
            near_mm = world_pos_mm(content, plane_z_m, near_layer.z_m,
                                   base_w_m, base_h_m, screen_w_mm,
                                   screen_h_mm, u, v, pad_mode)
            far_mm = world_pos_mm(content, plane_z_m, far_layer.z_m,
                                  base_w_m, base_h_m, screen_w_mm,
                                  screen_h_mm, u, v, pad_mode)
            p_near = project_to_design(near_mm, eye_mm,
                                       (screen_w_mm, screen_h_mm))
            p_far = project_to_design(far_mm, eye_mm,
                                      (screen_w_mm, screen_h_mm))
            if p_near is None or p_far is None:
                continue
            dist = float(np.hypot(p_near[0] - p_far[0], p_near[1] - p_far[1]))
            offsets.append(dist)
    return offsets, result.k


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", default="ghost_v0,ghost_v1,ghost_v2,ghost_v3")
    ap.add_argument("--only", default="u2_cat,u1_anime_room,t16")
    ap.add_argument("--out", default="results/layers_1110/ghost_ab")
    args = ap.parse_args(argv)

    screen_json = json.loads((ROOT / "assets/config/screen.json")
                             .read_text(encoding="utf-8"))
    screen_w_mm = float(screen_json.get("screen_width_m", 0.344)) * 1000.0
    screen_h_mm = float(screen_json.get("screen_height_m", 0.194)) * 1000.0

    images = review_set("eight")
    names = [s.strip() for s in args.only.split(",") if s.strip()]
    variants = [s.strip() for s in args.variants.split(",") if s.strip()]

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in names:
        path = images[name]
        for variant in variants:
            base, keys = PROFILES[variant]
            data = json.loads((ROOT / base).read_text(encoding="utf-8"))
            data["content_path"] = str(path)
            data.update(keys)
            content = ContentSettings.from_dict(data)
            offsets, k = reg_offsets_for(path, content, screen_w_mm, screen_h_mm)
            row = {
                "image": name, "variant": variant, "K": k,
                "n_points": len(offsets),
                "reg_offset_median_px": round(float(np.median(offsets)), 2)
                if offsets else "",
                "reg_offset_max_px": round(float(np.max(offsets)), 2)
                if offsets else "",
            }
            rows.append(row)
            print(f"{name:16} {variant:10} K={k} n={len(offsets):3} "
                  f"median={row['reg_offset_median_px']:>8} "
                  f"max={row['reg_offset_max_px']:>8}", flush=True)

    fields = ["image", "variant", "K", "n_points", "reg_offset_median_px",
             "reg_offset_max_px"]
    with (out_dir / "reg_offset.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "reg_offset.json").write_text(json.dumps(rows, indent=1),
                                             encoding="utf-8")
    print(f"-> {out_dir / 'reg_offset.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
