"""
Tìm chỗ đặt vật ẩn — PB-015 C1b. Tách khỏi ``hidden_props`` để mỗi file ≤ 150
dòng. Đây là cửa vào duy nhất của ``place_props``; chiều import chỉ một hướng
(search → hidden_props) nên không có vòng.

Ba thay đổi so với C1, đều từ số đo:
- **Hộp nghỉ.** C1 chỉ kiểm "ẩn" ở MỘT điểm mắt. Mắt người ngồi yên vẫn đưa
  võng: ``run2.log`` (24 khung z ≥ 420) cho x p10 −32 / p90 −10 và y p10 84 /
  p90 101, tức ~22 mm ngang và ~17 mm dọc. Vật vì thế tự lộ ra mà người chơi
  không hề nghiêng đầu.
- **Ngưỡng lộ.** 4/14 vật ở bảng C1 lộ chỉ sau 20–30 mm — nằm trong chính biên
  đưa võng đó.
- **Xếp hạng.** C1 lấy ứng viên đầu tiên gặp; giờ gom hết rồi chọn.
"""

from __future__ import annotations

import numpy as np

from src.content.hidden_props import _clearance_ok, _nearer, is_visible

#: Nửa bề rộng hộp nghỉ, mm — biên đưa võng đo được khi ngồi yên.
REST_BOX_X_MM = 20.0
REST_BOX_Y_MM = 15.0
#: Lộ sớm hơn mức này thì vật tự hiện, không tính là trò chơi.
MIN_REVEAL_MM = 40.0
#: Khoảng lộ dễ chịu và điểm ngắm khi phải chọn ngoài khoảng đó.
PREFER_LO, PREFER_HI, PREFER_MID = 50.0, 120.0, 85.0


def _rest_eyes(x0, y0, z):
    """Lưới 3 × 3 phủ hộp nghỉ quanh mắt nghỉ."""
    return [
        (x0 + dx, y0 + dy, z)
        for dx in (-REST_BOX_X_MM, 0.0, REST_BOX_X_MM)
        for dy in (-REST_BOX_Y_MM, 0.0, REST_BOX_Y_MM)
    ]


def _hidden_in_box(prop, radius_mm, near, x0, y0, zs):
    """Phải ẩn ở MỌI mắt trong hộp nghỉ, ở mọi khoảng cách trong ``zs``."""
    for z in zs:
        for eye in _rest_eyes(x0, y0, z):
            if is_visible(eye, prop, radius_mm, near):
                return False
    return True


def _reveal_offset(planes, prop, radius_mm, eye0, span, axis, step=10.0):
    """Offset nhỏ nhất (có dấu) theo một trục làm vật lộ ra. None nếu không."""
    near = _nearer(planes, prop[2] / 1000.0)
    for d in np.arange(step, span + step, step):
        for sgn in (-1.0, 1.0):
            eye = list(eye0)
            eye[axis] += sgn * d
            if is_visible(tuple(eye), prop, radius_mm, near):
                return float(sgn * d)
    return None


def _candidate(planes, prop, radius_mm, rest_xy, z_rest, z_near, spans):
    """Một ứng viên hợp lệ, hoặc None. ``prop`` tính bằng mm."""
    x0, y0 = rest_xy
    span_x, span_y = spans
    near = _nearer(planes, prop[2] / 1000.0)
    if not _hidden_in_box(prop, radius_mm, near, x0, y0, (z_rest, z_near)):
        return None
    axis, r450 = 0, _reveal_offset(
        planes, prop, radius_mm, (x0, y0, z_rest), span_x, 0)
    if r450 is None:                      # không lộ theo x thì thử theo y
        axis, r450 = 1, _reveal_offset(
            planes, prop, radius_mm, (x0, y0, z_rest), span_y, 1)
    if r450 is None or abs(r450) < MIN_REVEAL_MM:
        return None
    r330 = _reveal_offset(
        planes, prop, radius_mm, (x0, y0, z_near),
        span_x if axis == 0 else span_y, axis)
    return {"pos_mm": prop, "radius_mm": radius_mm,
            "reveal_axis": "x" if axis == 0 else "y",
            "reveal_450_mm": r450, "reveal_330_mm": r330}


def _rank(cands, n, radius_mm):
    """Ưu tiên lộ trong [50, 120] mm, rồi gần 85 mm nhất; giữ giãn cách 3 × r."""
    def key(c):
        r = abs(c["reveal_450_mm"])
        return (0 if PREFER_LO <= r <= PREFER_HI else 1, abs(r - PREFER_MID))

    out = []
    for c in sorted(cands, key=key):
        x, y = c["pos_mm"][0], c["pos_mm"][1]
        if any(abs(p["pos_mm"][0] - x) < 3 * radius_mm
               and abs(p["pos_mm"][1] - y) < 3 * radius_mm for p in out):
            continue
        out.append(c)
        if len(out) >= n:
            break
    return out


def place_props(planes, n, radius_mm, rest_eye_mm, screen_mm=(344.0, 194.0),
                grid=9, z_rest=450.0, z_near=330.0, span_x=120.0, span_y=60.0):
    """
    Tối đa ``n`` vật: ẩn trong cả hộp nghỉ, lộ khi nghiêng đầu thật sự. Trả ít
    hơn ``n`` (có thể rỗng) với ảnh không giấu được — không bao giờ lỗi.
    """
    x0, y0 = rest_eye_mm
    sw, sh = screen_mm
    cands = []
    for i in range(len(planes) - 1):
        lo, hi = planes[i].z_max_m, planes[i + 1].z_min_m
        if hi - lo < 2.0 * radius_mm / 1000.0:
            continue
        for frac in (1.0 / 3.0, 0.5, 2.0 / 3.0):
            z = lo + (hi - lo) * frac
            if not _clearance_ok(planes, z, radius_mm):
                continue
            for gx in np.linspace(-sw / 2 * 0.8, sw / 2 * 0.8, grid):
                for gy in np.linspace(-sh / 2 * 0.8, sh / 2 * 0.8, grid):
                    got = _candidate(
                        planes, (float(gx), float(gy), z * 1000.0), radius_mm,
                        (x0, y0), z_rest, z_near, (span_x, span_y))
                    if got is not None:
                        cands.append(got)
    return _rank(cands, n, radius_mm)
