"""
Mặt phẳng Kooima = hình chữ nhật **vật lý của cửa sổ** trên màn hình (D2).

Phép chiếu off-axis của HCPD lấy một hình chữ nhật thật trong không gian để
dựng khối nhìn. Khi app chạy toàn màn hình thì hình ấy đúng là cả màn hình, và
đó là thứ ``screen.json`` đo. Chạy trong một cửa sổ thì không còn đúng: cửa sổ
chỉ chiếm một phần màn, và nó **không nằm giữa**. Dùng nguyên hiệu chỉnh của
cả màn cho một cửa sổ kê bên trái nghĩa là nói với phép chiếu rằng tấm kính
rộng hơn và ở chỗ khác so với tấm kính người dùng đang nhìn — ảnh sẽ trượt
khi dời cửa sổ, đúng cái mà USER NEEDED 2 đi kiểm.

Hàm ở đây thuần số: không GL, không GLFW, nên ``tests/test_window_plane.py``
chấm được nó trực tiếp.
"""

from __future__ import annotations

import numpy as np


def pixel_size_m(screen_width_m: float, screen_height_m: float,
                 monitor_size) -> tuple[float, float]:
    """Mét trên mỗi pixel của màn hình, theo từng trục."""
    width_px = max(1.0, float(monitor_size[0]))
    height_px = max(1.0, float(monitor_size[1]))
    return (float(screen_width_m) / width_px, float(screen_height_m) / height_px)


def window_plane(screen_width_m: float, screen_height_m: float, screen_center_m,
                 monitor_pos, monitor_size, window_pos, window_size):
    """
    ``(width_m, height_m, center_m)`` của vùng khách cửa sổ.

    ``dy`` đếm theo pixel **xuôi xuống** (quy ước của hệ điều hành), còn trục
    Y của HCPD hướng **lên**, nên nó vào ``center_m`` với dấu trừ. Để nguyên
    dấu thì dời cửa sổ lên lại làm cảnh dịch xuống.

    Toàn màn hình (``window_pos == monitor_pos`` và cỡ bằng chế độ hình) cho
    ra đúng hiệu chỉnh gốc: ``dx = dy = 0``, ``width_m = screen_width_m``.
    """
    screen_w_px, screen_h_px = float(monitor_size[0]), float(monitor_size[1])
    sx, sy = pixel_size_m(screen_width_m, screen_height_m, monitor_size)
    window_w, window_h = float(window_size[0]), float(window_size[1])
    dx = (float(window_pos[0]) - float(monitor_pos[0])) + window_w * 0.5 - screen_w_px * 0.5
    dy = (float(window_pos[1]) - float(monitor_pos[1])) + window_h * 0.5 - screen_h_px * 0.5
    centre = np.asarray(screen_center_m, dtype=np.float64).reshape(3).copy()
    centre = centre + np.array([dx * sx, -dy * sy, 0.0], dtype=np.float64)
    return (window_w * sx, window_h * sy, centre)
