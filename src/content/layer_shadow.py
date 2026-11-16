r"""
PB-073 "Dimensional Layering" — bóng của lớp gần đổ lên lớp xa hơn.

Ý người dùng 22/10: cái làm thiệp giấy nhiều lớp trông có chiều sâu không phải
là bản thân chỗ cắt, mà là **bóng** mỗi lớp hắt xuống lớp dưới. Ở đây mỗi lớp
(trừ lớp xa nhất) là vật đổ bóng, lớp xa hơn kế tiếp là mặt nhận.

Hình học rút gọn, chốt trong `TASK_look3d_1031.md` Phase 3: bóng = đường bao
vật **dời** một đoạn rồi **làm nhoè**, cả hai tỉ lệ với khoảng cách Δz giữa
hai lớp. Không dựng tia sáng thật: với lớp phẳng song song và nguồn sáng xa
thì phép chiếu đúng *chính là* một phép dời, nên công thức rút gọn này không
phải xấp xỉ cho trường hợp đang dùng.

Mặt nhận có alpha riêng, và bóng được nhân với alpha đó nên **không bao giờ**
có bóng rơi vào khoảng trống (void) — điều kiện này có test.
"""

from __future__ import annotations

import cv2
import numpy as np
import numpy.typing as npt

#: Δz tham chiếu (m) để quy đổi ``*_frac`` ra pixel: khoảng cách hai lớp ở
#: cấu hình demo (``layer_z_far_m`` −0.35 với K = 3). Đổi K thì Δz đổi theo và
#: bóng tự giãn/co đúng tỉ lệ.
REFERENCE_GAP_M = 0.175


def shadow_geometry(width_px, delta_z_m, offset_frac, blur_frac,
                    reference_gap_m=REFERENCE_GAP_M):
    """``(độ dời px, sigma nhoè px)`` của một cặp lớp cách nhau ``delta_z_m``."""
    span = abs(float(delta_z_m)) / float(reference_gap_m)
    return (float(offset_frac) * float(width_px) * span,
            float(blur_frac) * float(width_px) * span)


def _shift(alpha, dx, dy):
    """Dời ảnh alpha đi ``(dx, dy)`` pixel, ngoài biên coi như trong suốt."""
    matrix = np.array([[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)]])
    return cv2.warpAffine(alpha, matrix, (alpha.shape[1], alpha.shape[0]),
                          flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def _blur(alpha, sigma):
    """Gaussian với bán kính suy ra từ ``sigma``; ``sigma <= 0`` thì bỏ qua."""
    if sigma <= 0.0:
        return alpha
    radius = max(1, int(round(3.0 * sigma)))
    size = 2 * radius + 1
    return cv2.GaussianBlur(alpha, (size, size), sigma)


def resample_to(alpha, src_w_m, src_cx_m, dst_shape, dst_w_m, dst_cx_m,
                src_h_m=None, src_cy_m=None, dst_h_m=None, dst_cy_m=None):
    """
    Đưa alpha của vật đổ bóng sang **hệ toạ độ texture của mặt nhận**.

    Hai lớp có quad rộng hẹp khác nhau (overscan phóng mỗi lớp một hệ số) và
    tâm lệch nhau, nên chồng thẳng hai mảng là sai. Phép đổi là affine thuần
    theo trục: điểm world ``x`` ứng với cột ``u`` của mặt nhận, rồi từ ``x``
    suy ngược ra cột của vật đổ bóng.
    """
    height, width = int(dst_shape[0]), int(dst_shape[1])
    src_h_m = src_w_m if src_h_m is None else src_h_m
    dst_h_m = dst_w_m if dst_h_m is None else dst_h_m
    src_cy_m = 0.0 if src_cy_m is None else src_cy_m
    dst_cy_m = 0.0 if dst_cy_m is None else dst_cy_m
    sx = float(dst_w_m) / float(src_w_m) * (alpha.shape[1] / float(width))
    sy = float(dst_h_m) / float(src_h_m) * (alpha.shape[0] / float(height))
    # Cột 0 của mặt nhận nằm ở đâu trong lưới của vật đổ bóng:
    tx = (0.5 + (float(dst_cx_m) - float(src_cx_m)) / float(src_w_m)
          ) * alpha.shape[1] - sx * width * 0.5
    ty = (0.5 - (float(dst_cy_m) - float(src_cy_m)) / float(src_h_m)
          ) * alpha.shape[0] - sy * height * 0.5
    matrix = np.array([[sx, 0.0, tx], [0.0, sy, ty]])
    return cv2.warpAffine(alpha, matrix, (width, height),
                          flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def shadow_alpha(
    caster_alpha: npt.NDArray[np.uint8],
    receiver_alpha: npt.NDArray[np.uint8],
    *,
    offset_px: float,
    sigma_px: float,
    angle_deg: float,
    opacity: float,
) -> npt.NDArray[np.uint8]:
    """
    Alpha của bóng, **cùng hệ toạ độ với mặt nhận**.

    ``angle_deg`` 90° = nguồn sáng ở trên, bóng đổ thẳng xuống. Trục y của ảnh
    hướng xuống nên dời xuống là ``+dy``.
    """
    if caster_alpha.shape != receiver_alpha.shape:
        raise ValueError(
            f"caster {caster_alpha.shape} khac receiver {receiver_alpha.shape}")
    radians = np.deg2rad(float(angle_deg))
    moved = _shift(caster_alpha.astype(np.float32),
                   float(offset_px) * np.cos(radians),
                   float(offset_px) * np.sin(radians))
    soft = _blur(moved, float(sigma_px))
    gated = soft * (receiver_alpha.astype(np.float32) / 255.0)
    return np.clip(gated * float(opacity), 0.0, 255.0).astype(np.uint8)


def shadow_rgba(caster_alpha, receiver_alpha, **kwargs):
    """Texture RGBA đen, alpha = :func:`shadow_alpha`."""
    alpha = shadow_alpha(caster_alpha, receiver_alpha, **kwargs)
    rgba = np.zeros((*alpha.shape, 4), dtype=np.uint8)
    rgba[..., 3] = alpha
    return np.ascontiguousarray(rgba)
