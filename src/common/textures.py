"""Texture bleed helper shared by ``renderer`` and ``fx`` — moved out of
``src.ui_album.textures`` (TASK_arch_refactor_1120 R2) so neither has to
depend on ``ui_album`` for it."""

from __future__ import annotations

import cv2
import numpy as np

#: TASK_game_feel_1118 P2: số lần lặp box-blur của ``bleed_transparent`` —
#: gần giống Gauss (rẻ hơn, ``cv2.blur`` là box-filter thuần), loang màu
#: ra xa dần theo số lần.
BLEED_ITERATIONS = 6
BLEED_KERNEL = 3


def bleed_transparent(array: np.ndarray, iterations: int = BLEED_ITERATIONS,
                      ksize: int = BLEED_KERNEL) -> np.ndarray:
    """
    Loang màu vào các điểm TRONG SUỐT HOÀN TOÀN (alpha=0) trước khi lên
    mipmap.

    PNG alpha thẳng (không premultiply) thường giữ RGB=đen ở chỗ alpha=0
    (không ai nhìn thấy màu đó — cho tới khi mipmap TRUNG BÌNH CỘNG nó
    vào viền, ra quầng tối quanh hình). Cách: premultiply (``RGB·a``),
    làm mờ CẢ ``RGB·a`` LẪN ``a`` bằng cùng một nhân hộp lặp
    ``iterations`` lần, rồi ở những điểm ``a=0`` gán
    ``RGB = blur(RGB·a) / max(blur(a), eps)`` — màu trung bình có trọng
    số của các điểm KHÔNG trong suốt gần đó. Alpha giữ NGUYÊN — chỉ RGB
    của điểm alpha=0 đổi.
    """
    if array.ndim != 3 or array.shape[2] != 4:
        return array
    alpha = array[..., 3]
    transparent = alpha == 0
    if not np.any(transparent):
        return array           # đặc hoàn toàn (hoặc không điểm nào a=0): không có gì để loang
    # TASK_game_feel_1118 P3: ``dst=`` dùng LẠI đúng bộ đệm mỗi lần lặp
    # (``cv2.blur`` mặc định cấp một mảng MỚI mỗi lần gọi) — atlas lớn
    # nhất (hit_kill, 1987x3548) tốn ~80 MB một bản float32 RGB; không
    # dùng lại thì N lần lặp = N bản sống cùng lúc, dễ ``MemoryError``
    # khi vài atlas lớn giải mã song song (đo được khi chạy test).
    a = alpha.astype(np.float32)
    a *= 1.0 / 255.0
    rgb_pm = array[..., :3].astype(np.float32)
    rgb_pm *= a[..., None]
    for _ in range(max(1, iterations)):
        rgb_pm = cv2.blur(rgb_pm, (ksize, ksize), dst=rgb_pm)
        a = cv2.blur(a, (ksize, ksize), dst=a)
    np.divide(rgb_pm, np.maximum(a[..., None], 1e-6), out=rgb_pm)
    np.clip(rgb_pm, 0.0, 255.0, out=rgb_pm)
    out = array.copy()
    # Index BEFORE converting dtype — ``rgb_pm[transparent]`` is only as
    # big as the transparent REGION; ``rgb_pm.astype(uint8)[transparent]``
    # would convert and hold the WHOLE image first (a 1920x1080 UI
    # background mostly-opaque-but-not-entirely blew well past 15 MiB
    # just for that one throwaway copy, observed under full-suite memory
    # pressure).
    out[..., :3][transparent] = rgb_pm[transparent].astype(np.uint8)
    return out
