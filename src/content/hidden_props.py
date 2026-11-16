"""
Đặt vật ẩn giữa các lớp — PB-015 phần 1. Thuần numpy, không OpenGL.

Vật nằm trong khe giữa hai lớp, bị lớp gần hơn che ở tư thế NGHỈ, lộ khi nghiêng
đầu. Lớp xa nhất là tấm nền đục toàn phần (PB-045 L1) nên không gì nấp sau nó.
Phép kiểm dùng **mặt phẳng phẳng** tại z đã shift và bỏ qua relief trong lớp:
trần đã biết. Bù lại khoảng hở lấy theo bao relief THẬT (``z_min``/``z_max`` ghi
từ mesh), không phải chặn 0.4 × gap.

Phần tìm kiếm chỗ đặt nằm ở ``hidden_props_search`` (PB-015 C1b) để mỗi file ≤ 150
dòng. Module này KHÔNG tái xuất ``place_props``: nó là tầng dưới, không được phép
import tầng trên. Hãy lấy ``place_props`` trực tiếp từ ``hidden_props_search``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Số tia (trên 5) phải thông để coi là thấy vật; alpha từ ALPHA_BLOCK là che.
RAYS_CLEAR_MIN = 3
ALPHA_BLOCK = 0.5


@dataclass(frozen=True)
class LayerPlane:
    """Một lớp đã vẽ, rút gọn cho phép kiểm che khuất."""

    alpha: np.ndarray                      # H x W uint8
    z_m: float                             # world z mặt phẳng, sau relief shift
    quad_w_m: float
    quad_h_m: float
    centre_x_m: float
    centre_y_m: float
    z_min_m: float                         # bao relief thật, không phải chặn
    z_max_m: float


def layer_planes(records):
    """Đổi ``_LayerRenderable`` của renderer thành bản ghi thuần, xa → gần."""
    return tuple(
        LayerPlane(r.alpha, r.z_world_m, r.quad_w_m, r.quad_h_m,
                   r.centre_x_m, r.centre_y_m, r.relief_z_min_m, r.relief_z_max_m)
        for r in records if r.alpha is not None
    )


def alpha_at(plane, x_mm, y_mm):
    """Alpha (0..1) của lớp tại điểm trên mặt phẳng của nó; 0 nếu ra ngoài quad."""
    w, h = plane.quad_w_m * 1000.0, plane.quad_h_m * 1000.0
    if w <= 0.0 or h <= 0.0:
        return 0.0
    u = (x_mm - (plane.centre_x_m * 1000.0 - w / 2)) / w
    # v = 0 ở MÉP TRÊN: glTexImage2D coi hàng 0 của mảng là đỉnh ảnh.
    v = ((plane.centre_y_m * 1000.0 + h / 2) - y_mm) / h
    if not (0.0 <= u <= 1.0 and 0.0 <= v <= 1.0):
        return 0.0
    hh, ww = plane.alpha.shape[:2]
    return float(plane.alpha[int(v * (hh - 1)), int(u * (ww - 1))]) / 255.0


def _blocked(eye_mm, point_mm, planes, threshold: float = ALPHA_BLOCK):
    """
    Tia mắt → điểm có bị lớp nào che không.

    ``threshold`` mặc định ``ALPHA_BLOCK`` (0,5) cho đúng hành vi cũ của
    ``is_visible`` (đặt chỗ / ẩn-hiện) — KHÔNG đổi. TASK_hit_0200 truyền
    ``content.layer_alpha_threshold`` (0,05) khi cần khớp đúng ngưỡng mà
    shader lớp dùng để vẽ/bỏ (``video_rgba.frag``), vì đó mới là "có vẽ đè
    lên màn hay không" — khác câu hỏi "có nên tính là ẩn hay không".
    """
    ex, ey, ez = eye_mm
    px, py, pz = point_mm
    if abs(pz - ez) < 1e-9:
        return False
    for pl in planes:
        s = (pl.z_m * 1000.0 - ez) / (pz - ez)      # MỌI thứ ở đây tính bằng mm
        if not (0.0 < s < 1.0):          # lớp không nằm giữa mắt và vật
            continue
        if alpha_at(pl, ex + s * (px - ex), ey + s * (py - ey)) >= threshold:
            return True
    return False


def is_visible(eye_mm, prop_mm, radius_mm, nearer_layers):
    """Thấy vật khi ít nhất ``RAYS_CLEAR_MIN`` trên 5 tia (tâm + 4 rìa) thông."""
    x, y, z = prop_mm
    r = float(radius_mm)
    points = ((x, y, z), (x - r, y, z), (x + r, y, z), (x, y - r, z), (x, y + r, z))
    clear = sum(0 if _blocked(eye_mm, p, nearer_layers) else 1 for p in points)
    return clear >= RAYS_CLEAR_MIN


def _nearer(planes, z_m):
    """Các lớp nằm GẦN người xem hơn vật (z lớn hơn)."""
    return tuple(p for p in planes if p.z_m > z_m + 1e-9)


def _clearance_ok(planes, z_m, radius_mm):
    """Vật phải nằm ngoài bao relief của MỌI lớp, cách ít nhất bán kính."""
    r = radius_mm / 1000.0
    return all(z_m + r <= p.z_min_m or z_m - r >= p.z_max_m for p in planes)
