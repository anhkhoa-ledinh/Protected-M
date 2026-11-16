r"""
PB-074: tô màu đánh dấu cho PHÉP ĐO. Không đặt biến môi trường thì im lặng.

Phase 0b cần biết mỗi điểm trên khung hình đến từ đâu: nền void/backdrop, dải
đệm bịa ra ở mép ảnh (``BORDER_REPLICATE``), hay vùng Telea lấp khe. Cách đo:
dựng cùng một cảnh **hai lượt** — lượt A tô ba nguồn đó bằng ĐỎ / LỤC / LAM,
lượt B tô cả ba bằng ĐEN — rồi lấy hiệu từng kênh.

Vì sao hiệu từng kênh là đúng: điểm cuối cùng trên màn =
``α·màu_nguồn·ánh_sáng + (1−α)·phần_còn_lại``. Phần ``(1−α)`` giống hệt nhau ở
hai lượt nên triệt tiêu, còn ``ánh_sáng`` chỉ nhân theo từng kênh. Hiệu A−B
do đó tỉ lệ với màu đánh dấu: dư ở kênh đỏ = void, lục = đệm, lam = lấp khe —
đúng cả khi điểm bị pha mờ ở biên feather, không cần ngưỡng mò.

| Biến | Tác dụng |
|---|---|
| ``HCPD_MARK_PAD=<r,g,b>`` | tô dải đệm quanh texture lớp |
| ``HCPD_MARK_FILL=<r,g,b>`` | tô vùng lấp của **lớp giữa** (dải 64 px) |
| ``HCPD_MARK_PLATE=<r,g,b>`` | tô vùng lấp của **tấm nền** (PB-075b) |
| ``HCPD_MARK_TOP=<r,g,b>`` | tô mask của lớp GẦN NHẤT |
"""

from __future__ import annotations

import os

import numpy as np

ENV_PAD = "HCPD_MARK_PAD"
ENV_FILL = "HCPD_MARK_FILL"
ENV_TOP = "HCPD_MARK_TOP"
ENV_PLATE = "HCPD_MARK_PLATE"


def _colour(name):
    """``(r, g, b)`` từ biến môi trường ``name``, hoặc ``None`` khi không đặt."""
    raw = (os.environ.get(name) or "").strip()
    if not raw:
        return None
    parts = raw.split(",")
    if len(parts) != 3:
        return None
    try:
        return tuple(int(float(p)) & 0xFF for p in parts)
    except ValueError:
        return None


def pad_colour():
    """Màu đánh dấu dải đệm, hoặc ``None`` = không đánh dấu."""
    return _colour(ENV_PAD)


def fill_colour():
    """Màu đánh dấu vùng lấp khe, hoặc ``None`` = không đánh dấu."""
    return _colour(ENV_FILL)


def plate_fill_colour():
    """
    Màu đánh dấu vùng lấp của **tấm nền**, hoặc ``None`` = không đánh dấu.

    PB-075b tách riêng khỏi ``fill_colour`` để trả lời đúng một câu: trong
    phần vùng lấp NHÌN THẤY được, bao nhiêu là của tấm nền và bao nhiêu là của
    dải lớp giữa. Không đặt biến thì vẫn như cũ — cả hai cùng một màu.
    """
    return _colour(ENV_PLATE) or _colour(ENV_FILL)


def top_layer_colour():
    """Màu đánh dấu lớp gần nhất, hoặc ``None`` = không đánh dấu."""
    return _colour(ENV_TOP)


def paint_border(rgba, pad_px, colour):
    """Tô vành ``pad_px`` quanh ``rgba`` (tại chỗ) bằng ``colour``; trả lại nó."""
    if colour is None or int(pad_px) <= 0:
        return rgba
    pad = int(pad_px)
    mark = np.asarray(colour, dtype=rgba.dtype)
    rgba[:pad, :, :3] = mark
    rgba[-pad:, :, :3] = mark
    rgba[:, :pad, :3] = mark
    rgba[:, -pad:, :3] = mark
    return rgba


def paint_zone(rgb, zone, colour):
    """Tô các điểm ``zone`` của ``rgb`` (tại chỗ) bằng ``colour``; trả lại nó."""
    if colour is None:
        return rgb
    rgb[zone] = np.asarray(colour, dtype=rgb.dtype)
    return rgb
