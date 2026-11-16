"""
Đo mắt nghỉ ngay lúc chạy — PB-015 C2, bổ sung 2026-10-15. Thuần numpy.

Một hằng số không khớp được người ngồi. ``overscan_design_eye_mm = [-20, 100]``
lấy từ một phiên (``run2.log``); phiên sau chủ sở hữu ngồi hẳn chỗ khác (ảnh
chụp 2: mắt (+40, 25, 541)). Góc mở nắp, người và ghế đổi mỗi lần ngồi, nên
chỉnh hằng số chỉ là khớp vào vài phiên đã qua. ``G`` vì thế đo tại chỗ.

Các ngưỡng dưới đây CHỈ lọc "đã ngồi yên chưa". Biên đưa võng *trong một lần
ngồi* vẫn do ``REST_BOX_*`` của C1b lo, và không đổi.
"""

from __future__ import annotations

import numpy as np

#: Thời gian gom mẫu khi bấm G, giây.
REST_MEASURE_S = 2.0
#: Dưới ngần này khung CÓ MẶT thì mẫu quá thưa để tin.
REST_MIN_FRAMES = 20
#: Dải p10–p90 rộng hơn mức này nghĩa là còn đang cử động, không phải ngồi yên.
REST_MAX_SPREAD_MM = 40.0
#: Chặn z mắt nghỉ trước khi đưa vào ``place_props``.
Z_REST_MIN_MM, Z_REST_MAX_MM = 350.0, 600.0


def spread_mm(samples, axis):
    """Dải p10–p90 của một trục, mm."""
    values = np.asarray(samples, dtype=np.float64)[:, axis]
    return float(np.percentile(values, 90) - np.percentile(values, 10))


def rest_eye(samples):
    """
    Trả ``((x0, y0, z0), None)`` khi mẫu đạt, hoặc ``(None, "lý do")``.

    ``samples`` là các mắt (mm) của những khung TÌM THẤY MẶT trong cửa sổ đo;
    khung mất mặt phải bị loại từ phía người gọi, vì lúc đó mắt là giá trị giữ
    lại của khung trước chứ không phải số đo mới.
    """
    if len(samples) < REST_MIN_FRAMES:
        return None, f"chi {len(samples)}/{REST_MIN_FRAMES} khung co mat"
    for axis, name in ((0, "x"), (1, "y")):
        width = spread_mm(samples, axis)
        if width > REST_MAX_SPREAD_MM:
            return None, f"{name} dao dong {width:.0f}mm > {REST_MAX_SPREAD_MM:.0f}mm"
    median = np.median(np.asarray(samples, dtype=np.float64), axis=0)
    return (float(median[0]), float(median[1]), float(median[2])), None


def z_rest_mm(z0):
    """z mắt nghỉ đã chặn, dùng làm ``z_rest`` cho ``place_props``."""
    return float(np.clip(float(z0), Z_REST_MIN_MM, Z_REST_MAX_MM))
