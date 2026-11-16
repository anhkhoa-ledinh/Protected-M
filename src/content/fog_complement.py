r"""
PB-071: xám hoá lớp xa bằng **màu bù** thay vì trộn một tấm xám.

Chép (không import) từ ``scripts/_fog_complement.py`` của bản đo A/B ngày
29/10 (commit ``188df40``); đo thì ở ``scripts/``, app thì không được nhập từ
``scripts/``. Công thức giữ nguyên từng chữ.

Ý người dùng 29/10: muốn xám một màu thì thêm màu bù của nó, ra xám tối tự
nhiên, chứ không phải phủ một tấm xám lên. Trộn **trong linear RGB** bằng
trung bình **hình học** (trộn bột màu là phép trừ, nên vừa xám vừa tối đi —
khác hẳn lerp tuyến tính sang xám):

    c' = (max(c) + min(c)) − c        (bù theo từng pixel, trên 3 kênh)
    out = c^(1−t) · c'^t              (theo từng kênh)
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

BISECT_STEPS = 24

#: Chỉ dò ``t`` trong [0, T_MAX]. Bão hoà theo ``t`` có dạng chữ U, KHÔNG giảm
#: đều: ``t = 1`` là **màu bù nguyên chất**, mà bù của một màu rực thì cũng rực
#: y như nó (đỏ → lơ), nên bão hoà quay về gần mức gốc. Điểm xám nhất nằm ở
#: giữa, chỗ trung bình hình học của màu và bù của nó triệt tiêu. Vì vậy chỉ
#: nửa [0, 0.5] là đơn điệu giảm và dò được bằng chia đôi.
T_MAX = 0.5


#: Mọi phép tính chạy ở ``float32``. Bản đầu dùng ``float64``: cùng công thức
#: nhưng gấp đôi băng thông bộ nhớ cho **không** một mức màu 8-bit nào khác.
WORK_DTYPE = np.float32
EPS = np.float32(1e-6)


def _srgb_to_linear(x):
    """sRGB [0,1] → linear."""
    return np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)


def _linear_to_srgb(x):
    """linear → sRGB [0,1]."""
    x = np.clip(x, 0.0, 1.0)
    return np.where(x <= 0.0031308, x * 12.92,
                    1.055 * np.power(x, 1.0 / 2.4) - 0.055)


def prepare(rgb_u8):
    """
    ``(log(lin), log(bù))`` của một ảnh — phần KHÔNG phụ thuộc ``t``.

    Tách ra vì chia đôi gọi phép trộn 24 lần: bản đầu giải mã sRGB → linear và
    dựng lại màu bù ở **mỗi** bước, tức làm lại 24 lần một việc chỉ cần làm
    một lần. Lấy sẵn logarit luôn, vì trộn hình học chính là nội suy tuyến
    tính trong miền log.
    """
    lin = _srgb_to_linear(np.asarray(rgb_u8, dtype=WORK_DTYPE) / 255.0)
    hi = lin.max(axis=2, keepdims=True)
    lo = lin.min(axis=2, keepdims=True)
    comp = np.clip((hi + lo) - lin, 0.0, 1.0)
    return np.log(lin + EPS), np.log(comp + EPS)


def mix_prepared(prepared, t):
    """Trộn từ bản đã ``prepare``; trả ảnh sRGB uint8."""
    log_lin, log_comp = prepared
    k = WORK_DTYPE(t)
    mixed = np.exp(log_lin + k * (log_comp - log_lin))
    return np.clip(_linear_to_srgb(mixed) * 255.0, 0, 255).astype(np.uint8)


def complement_mix(rgb_u8, t):
    """
    Trộn ảnh với màu bù của chính nó, hệ số ``t`` ∈ [0, 1], trong linear RGB.

    ``t = 0`` trả lại ảnh gốc; ``t = 1`` là màu bù hoàn toàn. Trung bình hình
    học nên càng trộn càng TỐI, đúng như trộn bột màu.
    """
    return mix_prepared(prepare(rgb_u8), t)


def mean_saturation(rgb_u8: npt.NDArray[np.uint8],
                    mask: npt.NDArray[np.bool_] | None = None) -> float:
    """Bão hoà trung bình (HSV của OpenCV) trên ``mask``; ``None`` = cả ảnh."""
    sat = cv2.cvtColor(np.ascontiguousarray(rgb_u8, dtype=np.uint8),
                       cv2.COLOR_RGB2HSV)[..., 1].astype(np.float64)
    if mask is None:
        return float(sat.mean())
    return float(sat[mask].mean()) if bool(mask.any()) else float("nan")


def solve_t_for_saturation(rgb_u8: npt.NDArray[np.uint8],
                           mask: npt.NDArray[np.bool_] | None,
                           target_sat: float) -> float:
    """
    ``t`` ∈ [0, 0.5] làm bão hoà trung bình bằng ``target_sat``, bằng chia đôi.

    Trả đúng ``T_MAX`` khi ngay ở điểm xám nhất bão hoà vẫn còn cao hơn đích;
    lúc đó lớp sẽ xám ít hơn bản B− và phải báo cáo như thế, chứ không nới
    ``t`` ra ngoài khoảng đơn điệu.
    """
    if mask is not None and not bool(mask.any()):
        return 0.0
    if not np.isfinite(target_sat):
        return 0.0
    prepared = prepare(rgb_u8)                  # giải mã sRGB đúng MỘT lần

    def sat_at(t_value):
        return mean_saturation(mix_prepared(prepared, t_value), mask)

    if sat_at(T_MAX) > target_sat:
        return T_MAX
    lo, hi = 0.0, T_MAX
    for _ in range(BISECT_STEPS):
        mid = 0.5 * (lo + hi)
        if sat_at(mid) > target_sat:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def _layer_pixels(rgb_u8, mask):
    """Các điểm của lớp xếp thành ảnh ``N×1×3`` cho OpenCV; ``None`` = cả ảnh."""
    px = rgb_u8[mask] if mask is not None else rgb_u8.reshape(-1, 3)
    return np.ascontiguousarray(px.reshape(-1, 1, 3), dtype=np.uint8)


def solve_t_for_layer(base_u8, atmo_u8, mask=None):
    """
    ``t`` để lớp ``base_u8`` (mới chỉnh contrast) xám bằng bản atmo ``atmo_u8``.

    Dò trên **đúng các điểm của lớp**, không xấp xỉ: đích bão hoà và phép dò
    dùng cùng tập điểm đó. Đắt hơn dò trên bản thu nhỏ nhưng cho đúng ``t``
    của bản dựng ảnh A/B mà người dùng đã chọn bằng mắt (xem README PB-074
    Phase 1: thu nhỏ về 256 px làm ``t`` của lớp giữa lệch 0,033 vì
    ``INTER_AREA`` trộn cả điểm ngoài mask vào — lớp giữa vốn vụn).

    Trả ``0.0`` khi lớp rỗng.
    """
    if mask is not None and not bool(mask.any()):
        return 0.0
    target = mean_saturation(_layer_pixels(atmo_u8, mask), None)
    return solve_t_for_saturation(_layer_pixels(base_u8, mask), None, target)
