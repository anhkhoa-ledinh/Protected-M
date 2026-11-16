r"""
PB-074: các chỉ số đọc từ **khung hình đã dựng** (không phải từ texture lớp).

Cách đánh dấu: xem ``src/content/debug_marks.py``. Lượt A tô void/backdrop =
ĐỎ, đệm = LỤC, lấp khe = LAM; lượt B tô cả ba = ĐEN. Hiệu A−B tách theo kênh
cho đúng ba nguồn, kể cả ở điểm pha mờ.
"""
from __future__ import annotations

import cv2
import numpy as np

MARK_A = {"void": (255, 0, 0), "pad": (0, 255, 0), "fill": (0, 0, 255)}
MARK_B = {"void": (0, 0, 0), "pad": (0, 0, 0), "fill": (0, 0, 0)}

#: Dư sáng tối thiểu (0–255) để coi một kênh là "có nguồn đánh dấu". 8 nằm
#: trên nhiễu làm tròn 8-bit của hai lượt dựng (đo được: nền ảnh thật lệch
#: 0–2 mức) và dưới mức một điểm feather mờ nhất còn nhìn thấy.
MARK_MIN = 8


def source_masks(frame_a, frame_b, min_level=MARK_MIN):
    """``{"void": mask, "pad": mask, "fill": mask}`` từ hai lượt dựng."""
    delta = frame_a.astype(np.int16) - frame_b.astype(np.int16)
    return {"void": delta[..., 0] >= min_level,
            "pad": delta[..., 1] >= min_level,
            "fill": delta[..., 2] >= min_level}


def share_pct(mask):
    """Phần trăm điểm ảnh của khung hình thuộc ``mask``."""
    return round(100.0 * float(np.count_nonzero(mask)) / float(mask.size), 4)


def _lap_var(gray, mask):
    """Phương sai Laplacian trên ``mask``; ``nan`` khi mask rỗng."""
    if not bool(mask.any()):
        return float("nan")
    lap = cv2.Laplacian(gray, cv2.CV_64F, ksize=3)
    return float(lap[mask].var())


def fill_tex_ratio(frame, fill_mask, ring_px=15):
    """
    Độ "có vân" của vùng lấp so với ảnh thật quanh nó. 1,0 = giống hệt.

    Vành so sánh là ``ring_px`` điểm quanh vùng lấp, trừ chính vùng lấp — tức
    là ảnh thật ngay cạnh, cùng vùng sáng và cùng loại nội dung.
    """
    if not bool(fill_mask.any()):
        return float("nan")
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    outside = np.where(fill_mask, np.uint8(0), np.uint8(255))
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    ring = (distance > 0) & (distance <= float(ring_px))
    inside = _lap_var(gray, fill_mask)
    around = _lap_var(gray, ring)
    if not np.isfinite(inside) or not np.isfinite(around) or around <= 0.0:
        return float("nan")
    return round(inside / around, 4)


def rim_dl(frame, near_mask, rings=(3.0, 6.0, 12.0)):
    """
    Vệt sáng **bên trong** đường bao lớp gần (viền sáng ``add_rim_light``).

    Cùng kiểu vành như :func:`halo_dl` nhưng đo phía TRONG mask: L\\* của vành
    0–3 px trong trừ L\\* của vành 6–12 px trong. Số dương = có dải sáng viền.
    ``halo_dl`` đo phía ngoài nên không bắt được viền sáng này.
    """
    if not bool(near_mask.any()):
        return float("nan")
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float64)
    lab *= 100.0 / 255.0
    inside = np.where(near_mask, np.uint8(255), np.uint8(0))
    distance = cv2.distanceTransform(inside, cv2.DIST_L2, 3)
    edge_ring = (distance > 0) & (distance <= float(rings[0]))
    core_ring = (distance > float(rings[1])) & (distance <= float(rings[2]))
    if not edge_ring.any() or not core_ring.any():
        return float("nan")
    return round(float(lab[edge_ring].mean() - lab[core_ring].mean()), 3)


def halo_dl(frame, near_mask, rings=(3.0, 6.0, 12.0)):
    """
    Vệt sáng/tối còn sót quanh đường bao lớp gần, đo trên KHUNG HÌNH.

    Lấy L\\* trung bình của vành 0–3 px ngoài đường bao trừ L\\* của vành
    6–12 px (vùng "nền sạch"). Số dương = quanh vật sáng hơn nền, tức còn
    quầng. Cùng định nghĩa vành với ``_layer_report.rings_dl`` nhưng nguồn là
    ảnh đã dựng, nên so được giữa các cách vẽ khác nhau.
    """
    if not bool(near_mask.any()):
        return float("nan")
    lab = cv2.cvtColor(frame, cv2.COLOR_RGB2Lab)[..., 0].astype(np.float64)
    lab *= 100.0 / 255.0
    outside = np.where(near_mask, np.uint8(0), np.uint8(255))
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, 3)
    near_ring = (distance > 0) & (distance <= float(rings[0]))
    far_ring = (distance > float(rings[1])) & (distance <= float(rings[2]))
    if not near_ring.any() or not far_ring.any():
        return float("nan")
    return round(float(lab[near_ring].mean() - lab[far_ring].mean()), 3)
