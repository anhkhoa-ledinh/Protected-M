r"""
PB-071: xám hoá bằng **màu bù** (kỹ thuật của hoạ sĩ) thay vì trộn lớp xám.

Ý người dùng 29/10: muốn xám một màu thì thêm màu bù của nó, ra xám tối tự
nhiên, chứ không phải phủ một tấm xám lên.

Công thức, trộn **trong linear RGB** (trộn bột màu là phép trừ, nên dùng trung
bình **hình học**, nó vừa xám vừa tối đi — khác hẳn lerp tuyến tính sang xám):

    c' = (max(c) + min(c)) − c        (bù theo từng pixel, trên 3 kênh)
    out = c^(1−t) · c'^t              (theo từng kênh)

``t`` tìm bằng chia đôi cho tới khi **bão hoà trung bình** của lớp bằng bão hoà
của bản B− — khớp bão hoà để phép A/B trả lời "xám KIỂU NÀO", không phải "xám
BAO NHIÊU".
"""
from __future__ import annotations

import cv2
import numpy as np

#: Thang cường độ so với mức A (blend 0.35 / sat 0.60 / contrast 0.75).
A_BLEND, A_SAT_DROP, A_CONTRAST_DROP = 0.35, 0.4, 0.25
BISECT_STEPS = 24


def strength(s):
    """``(blend, saturation, contrast)`` của cường độ ``s`` so với mức A."""
    return (A_BLEND * float(s), 1.0 - A_SAT_DROP * float(s),
            1.0 - A_CONTRAST_DROP * float(s))


def _srgb_to_linear(x):
    """sRGB [0,1] → linear."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    """linear → sRGB [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def complement_mix(rgb_u8, t):
    """
    Trộn ảnh với màu bù của chính nó, hệ số ``t`` ∈ [0, 1], trong linear RGB.

    ``t = 0`` trả lại ảnh gốc; ``t = 1`` là màu bù hoàn toàn. Trung bình hình
    học nên càng trộn càng TỐI, đúng như trộn bột màu.
    """
    lin = _srgb_to_linear(np.asarray(rgb_u8, dtype=np.float64) / 255.0)
    hi = lin.max(axis=2, keepdims=True)
    lo = lin.min(axis=2, keepdims=True)
    comp = np.clip((hi + lo) - lin, 0.0, 1.0)
    eps = 1e-6
    mixed = np.exp((1.0 - float(t)) * np.log(lin + eps)
                   + float(t) * np.log(comp + eps))
    return np.clip(_linear_to_srgb(mixed) * 255.0, 0, 255).astype(np.uint8)


def mean_saturation(rgb_u8, mask):
    """Bão hoà trung bình (HSV của OpenCV) trên ``mask``."""
    sat = cv2.cvtColor(np.ascontiguousarray(rgb_u8, dtype=np.uint8),
                       cv2.COLOR_RGB2HSV)[..., 1].astype(np.float64)
    return float(sat[mask].mean()) if bool(mask.any()) else float("nan")


#: Chỉ dò ``t`` trong [0, T_MAX]. Bão hoà theo ``t`` có dạng chữ U, KHÔNG giảm
#: đều: ``t = 1`` là **màu bù nguyên chất**, mà bù của một màu rực thì cũng rực
#: y như nó (đỏ → lơ), nên bão hoà quay về gần mức gốc. Điểm xám nhất nằm ở
#: giữa, chỗ trung bình hình học của màu và bù của nó triệt tiêu. Vì vậy chỉ
#: nửa [0, 0.5] là đơn điệu giảm và dò được bằng chia đôi.
T_MAX = 0.5


def solve_t_for_saturation(rgb_u8, mask, target_sat):
    """
    ``t`` ∈ [0, 0.5] làm bão hoà trung bình bằng ``target_sat``, bằng chia đôi.

    Trả đúng ``T_MAX`` khi ngay ở điểm xám nhất bão hoà vẫn còn cao hơn đích;
    lúc đó cột ``saturation_pct`` sẽ không khớp B− và phải báo cáo như thế, chứ
    không nới ``t`` ra ngoài khoảng đơn điệu.
    """
    if not bool(mask.any()) or not np.isfinite(target_sat):
        return 0.0
    if mean_saturation(complement_mix(rgb_u8, T_MAX), mask) > target_sat:
        return T_MAX
    lo, hi = 0.0, T_MAX
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if mean_saturation(complement_mix(rgb_u8, mid), mask) > target_sat:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def mean_lab_l(rgb_u8, mask):
    """L* trung bình (CIELAB) trên ``mask``, thang 0–100."""
    lab = cv2.cvtColor(np.ascontiguousarray(rgb_u8, dtype=np.uint8),
                       cv2.COLOR_RGB2Lab)[..., 0].astype(np.float64)
    if not bool(mask.any()):
        return float("nan")
    return float(lab[mask].mean()) * 100.0 / 255.0
