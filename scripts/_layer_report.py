"""In bảng đo lớp cho ``measure_layers.py`` — PB-045. Chỉ đọc. Mọi công thức ở
đây là **sao y** code chạy thật, có ghi địa chỉ; sửa bên kia phải sửa ở đây."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from src.content.disocclusion import disocclusion_band
from src.content.layer_clustering import relief_depth_scale
from src.scene.primitives import RELIEF_CLAMP_GAP_FRACTION, _sample_bilinear_grid

SW, SH = 344.0, 194.0          # màn hình vật lý, mm


def overscan_scale(t_mm, cw, ch, e_max, dd):
    """Sao y ``ContentPlaneRenderer._overscan_scale`` (content_plane.py:293-325)."""
    if e_max <= 0.0 or t_mm <= 1e-6:
        return 1.0
    d = max(dd, 1.0)
    need = lambda s: s * 0.5 * (d + t_mm) / d + e_max * t_mm / d  # noqa: E731
    return max(1.0, need(SW) / (cw / 2), need(SH) / (ch / 2))


def relief_stats(layer, depth, cs, gap_m, z_world_m, res):
    """Dải relief của mesh một lớp theo world Z (mm). Sao y primitives.py:256-302."""
    span = max(float(layer.depth_hi - layer.depth_lo), 1e-6)
    mean_norm = float(np.clip((layer.mean_depth - layer.depth_lo) / span, 0.0, 1.0))
    scale = relief_depth_scale(layer.depth_norm, float(cs.relief_far_falloff))
    u, v = np.meshgrid(np.arange(res + 1) / res, np.arange(res + 1) / res)
    local = np.clip((_sample_bilinear_grid(depth, u, v) - layer.depth_lo) / span, 0, 1)
    mask_w = _sample_bilinear_grid(layer.mask.astype(np.float32), u, v)
    r = ((local - mean_norm) * (float(cs.relief_amplitude) * gap_m)
         * scale * mask_w).astype(np.float32)
    sigma = max(0.0, float(cs.relief_smooth_sigma))
    if sigma > 0.0:
        r = cv2.GaussianBlur(r, (max(3, int(sigma * 6.0) | 1),) * 2, sigma)
    clamp = RELIEF_CLAMP_GAP_FRACTION * gap_m
    zw = z_world_m * 1000.0 + np.clip(r, -clamp, clamp) * 1000.0
    b, ring = max(1, int(res * 0.05)), np.zeros(zw.shape, dtype=bool)
    ring[:b, :] = ring[-b:, :] = ring[:, :b] = ring[:, -b:] = True
    return zw.min(), zw.max(), 100.0 * (zw > 0.0).mean(), zw[ring].max()


def band_seam_min(layer, nearer, fill_px, feather_px, ring_px=2):
    """
    Alpha nhỏ nhất của **dải disocclusion** nằm trong ``ring_px`` px quanh mask
    của chính lớp đó — định nghĩa ở L1 acceptance.

    Dải lấy từ ``disocclusion_band``, KHÔNG từ ``alpha > 0 & ~own``: cái sau gộp
    cả váy feather riêng của lớp (18 px), vốn ramp xuống là đúng, và che mất
    hiệu ứng của bản sửa. **Trừ** phần dải trong ``feather_px`` tính từ mép
    NGOÀI (dốc ở đó là cố ý); khoảng cách gieo trên ``band ∪ own`` nên mép áp
    đường bao riêng không tính là biên. Trả ``-1`` khi không có điểm.
    """
    own, alpha = layer.mask.astype(bool), layer.rgba[..., 3]
    band = disocclusion_band(own, nearer, int(fill_px))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(ring_px) + 1,) * 2)
    near = cv2.dilate(own.astype(np.uint8), k) > 0
    seed = np.where(band | own, np.uint8(255), np.uint8(0))
    dist = cv2.distanceTransform(seed, cv2.DIST_L2, 3)
    sel = near & band & (dist > float(feather_px))
    return int(alpha[sel].min()) if bool(sel.any()) else -1


def rings_dl(layer, nearer, a_lo, a_hi, b_lo, b_hi, a_in):
    """
    |luminance trung bình vành A − vành B|, hai vành định theo mép lỗ.

    ``a_in`` True → A nằm TRONG lỗ = ``fill_leak_dL`` (màu lấp lệch so với nền
    thật). False → A nằm NGOÀI lỗ = ``halo_dL`` (viền vật còn sót trên tấm nền,
    là những điểm biên lai không được inpaint mà cũng không bị thay). B luôn là
    điểm riêng của lớp, đủ xa mép để chính nó không làm từ các điểm lai đó.
    """
    own, hole = layer.mask.astype(bool), nearer & ~layer.mask.astype(bool)
    if not bool(hole.any()):
        return float("nan")
    rgb = layer.rgba[..., :3].astype(np.float64)
    lum = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    h8 = np.where(hole, np.uint8(255), np.uint8(0))
    d_out = cv2.distanceTransform(255 - h8, cv2.DIST_L2, 3)
    if a_in:
        ring_a = hole & (cv2.distanceTransform(h8, cv2.DIST_L2, 3) <= a_hi)
    else:
        ring_a = ~hole & (d_out >= a_lo) & (d_out <= a_hi)
    ring_b = own & ~hole & (d_out >= b_lo) & (d_out <= b_hi)
    if not (bool(ring_a.any()) and bool(ring_b.any())):
        return float("nan")
    return abs(float(lum[ring_a].mean()) - float(lum[ring_b].mean()))


def print_layer(rank, L, ts, tr, kr, rel, zw_m, cw, ch, eyes, nearer, cs):
    """Một lớp: hình học, độ phủ, relief, rồi M và e_void theo từng khoảng cách."""
    lo, hi, front, bmax = rel
    bsm = band_seam_min(L, nearer, cs.disocclusion_fill_px, cs.disocclusion_feather_px)
    print(f"  lop {rank}: z_world={zw_m * 1000:+8.1f}  t_signed={ts:+8.1f}  "
          f"t_renderer_abs={tr:8.1f}  k_render={kr:.3f}  "
          f"coverage={100.0 * L.mask.mean():5.1f}%"
          f"  alpha>=250={100.0 * (L.rgba[..., 3] >= 250).mean():5.1f}%")
    print(f"         relief {lo:+.2f}..{hi:+.2f}mm  truoc kinh {front:.2f}%  "
          f"z max vanh mep 5% {bmax:+.2f}mm  "
          f"band_seam_min={'n/a' if bsm < 0 else bsm}  "
          f"fill_leak_dL={rings_dl(L, nearer, 0.0, 6.0, 3.0, 9.0, True):.2f}  "
          f"halo_dL={rings_dl(L, nearer, 0.0, 3.0, 6.0, 12.0, False):.2f}")
    for d in eyes:
        m = kr * d / (d + ts) if abs(d + ts) > 1e-9 else float("nan")
        if abs(ts) < 1e-6:
            ex = ey = "    inf"
        else:
            ex = f"{(cw / 2 * kr - SW / 2 * (d + ts) / d) * d / ts:7.1f}"
            ey = f"{(ch / 2 * kr - SH / 2 * (d + ts) / d) * d / ts:7.1f}"
        print(f"         D={d:5.0f}mm  M={m:5.3f}  e_void X={ex} Y={ey}")


def report(path, rgb, depth, layers, gvf, gvf_adj, timing, cs, args):
    """In toàn bộ bảng của một ảnh và ghi PNG từng lớp + composite_alpha."""
    plane_z, cw = float(cs.plane_position_m[2]), float(cs.plane_width_m) * 1000.0
    ch = cw / (rgb.shape[1] / rgb.shape[0])
    dd = args.design_distance if args.design_distance is not None else 600.0
    src = "--design-distance" if args.design_distance is not None else "mac dinh 600"
    gap = (cs.layer_z_near_m - cs.layer_z_far_m) / max(len(layers) - 1, 1)
    e_max, mres = float(cs.overscan_e_max_mm), max(8, int(cs.depth_mesh_resolution))
    r4 = lambda d: {k: round(v, 4) for k, v in sorted(d.items())}  # noqa: E731
    print(f"\n{'=' * 78}\n{path.name}  {rgb.shape[1]}x{rgb.shape[0]}  K = {len(layers)}")
    print(f"  GVF tho={r4(gvf)}  GVF adjusted={r4(gvf_adj)}")
    print(f"  timing ms={ {k: round(v, 1) for k, v in sorted(timing.items())} }")
    print(f"  plane_z={plane_z:+.3f}m width={cw:.1f}mm height={ch:.1f}mm "
          f"gap={gap * 1000:.1f}mm e_max={e_max:.0f}mm design_distance={dd:.0f}mm ({src})")
    out = Path(args.out) / path.stem
    out.mkdir(parents=True, exist_ok=True)
    comp = np.ones(layers[0].rgba.shape[:2], np.float64)
    for rank, L in enumerate(layers):
        comp *= 1.0 - L.rgba[..., 3].astype(np.float64) / 255.0
        zw_m = plane_z + float(L.z_m)
        kr = overscan_scale(abs(zw_m) * 1000.0, cw, ch, e_max, dd)
        nearer = np.zeros_like(L.mask, dtype=bool)
        for o in layers[rank + 1:]:
            nearer |= o.mask.astype(bool)
        print_layer(rank, L, -zw_m * 1000.0, abs(zw_m) * 1000.0, kr,
                    relief_stats(L, depth, cs, gap, zw_m, mres),
                    zw_m, cw, ch, args.eye_distance, nearer, cs)
        cv2.imwrite(str(out / f"layer{rank}.png"),
                    cv2.cvtColor(L.rgba, cv2.COLOR_RGBA2BGRA))
    a0, composite = layers[0].rgba[..., 3], 1.0 - comp
    print(f"  plate_open_pct={100.0 * (a0 < 250).mean():.2f}%  "
          f"seam_pct={100.0 * (composite < 0.98).mean():.2f}%  -> {out}")
    cv2.imwrite(str(out / "composite_alpha.png"),
                (np.clip(composite, 0, 1) * 255).astype(np.uint8))
