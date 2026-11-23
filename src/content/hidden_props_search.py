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

import random

import numpy as np

from src.content.hidden_props import _clearance_ok, _nearer, is_visible
from src.game.levels import LEGACY_PREFER

#: Nửa bề rộng hộp nghỉ, mm — biên đưa võng đo được khi ngồi yên.
REST_BOX_X_MM = 20.0
REST_BOX_Y_MM = 15.0
#: Lộ sớm hơn mức này thì vật tự hiện, không tính là trò chơi.
MIN_REVEAL_MM = 40.0
#: Khoảng lộ dễ chịu mặc định (PvP); solo truyền khoảng của cấp (levels.py).
PREFER_LO, PREFER_HI = LEGACY_PREFER
#: Seeded runs shift each depth fraction by up to this much of the layer gap.
FRAC_JITTER = 0.06


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


def _rank(cands, n, radius_mm, rng=None, prefer=LEGACY_PREFER):
    """Ưu tiên lộ trong ``prefer`` (mm), rồi gần giữa khoảng nhất; giữ giãn cách 3 × r.

    ``prefer`` is a preference only: a candidate outside it is still kept
    when the preferred ones run out, so it never lowers the count.

    ``rng`` (a ``random.Random``) shuffles BEFORE the stable sort, so only
    candidates with an equal preference key change order; the preference
    order itself never does. ``None`` = the old deterministic order.
    """
    lo, hi = prefer
    mid = (lo + hi) / 2.0

    def key(c):
        r = abs(c["reveal_450_mm"])
        return (0 if lo <= r <= hi else 1, abs(r - mid))

    cands = list(cands)
    if rng is not None:
        rng.shuffle(cands)
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


#: TASK_hud_fix_1119 Fix 3: side of the eye-position grid used to double
#: check a KEPT prop is actually exposable, independent of
#: ``_candidate``'s own single-axis reveal scan above (which only walks
#: ONE axis at a time, not the full 2D peek range at once). 5x5 = 25
#: positions, matching the task's own number.
EXPOSURE_GRID = 5


def exposure_grid(x0, y0, z, span_x=120.0, span_y=60.0, grid=EXPOSURE_GRID):
    """``grid`` x ``grid`` eye positions (mm), evenly covering
    ``[x0-span_x, x0+span_x] x [y0-span_y, y0+span_y]`` at depth ``z`` —
    the peek range a player can lean through at the screen distance."""
    return [(x0 + dx, y0 + dy, z)
           for dx in np.linspace(-span_x, span_x, grid)
           for dy in np.linspace(-span_y, span_y, grid)]


def exposure_count(planes, prop_mm, radius_mm, rest_xy, z,
                   span_x=120.0, span_y=60.0, grid=EXPOSURE_GRID) -> int:
    """How many of ``exposure_grid``'s eye positions expose ``prop_mm``
    (``is_visible``, same ``ALPHA_BLOCK=0.5`` default as gameplay's own
    "is this prop exposed right now" check) — 0 means a spot that can
    NEVER be found, the TASK_hud_fix_1119 Fix 3 rejection case."""
    near = _nearer(planes, prop_mm[2] / 1000.0)
    x0, y0 = rest_xy
    return sum(1 for eye in exposure_grid(x0, y0, z, span_x, span_y, grid)
              if is_visible(eye, prop_mm, radius_mm, near))


def place_props(planes, n, radius_mm, rest_eye_mm, screen_mm=(344.0, 194.0),
                grid=9, z_rest=450.0, z_near=330.0, span_x=120.0, span_y=60.0,
                seed=None, prefer=LEGACY_PREFER):
    """
    Tối đa ``n`` vật: ẩn trong cả hộp nghỉ, lộ khi nghiêng đầu thật sự. Trả ít
    hơn ``n`` (có thể rỗng) với ảnh không giấu được — không bao giờ lỗi.

    ``seed`` (any hashable, e.g. the per-match int) jitters every grid point
    by up to half a cell and the depth fraction slightly, and breaks ties in
    ``_rank`` at random. Every jittered point still goes through the same
    ``_candidate`` test (hidden in the rest box, revealed on a lean), so no
    rule is loosened. Same seed, same result; ``None`` = fixed grid, as before.
    """
    rng = None if seed is None else random.Random(f"place:{seed}")
    x0, y0 = rest_eye_mm
    sw, sh = screen_mm
    cell_x = sw * 0.8 / max(grid - 1, 1)
    cell_y = sh * 0.8 / max(grid - 1, 1)
    cands = []
    for i in range(len(planes) - 1):
        lo, hi = planes[i].z_max_m, planes[i + 1].z_min_m
        if hi - lo < 2.0 * radius_mm / 1000.0:
            continue
        for frac in (1.0 / 3.0, 0.5, 2.0 / 3.0):
            if rng is not None:
                frac += rng.uniform(-FRAC_JITTER, FRAC_JITTER)
            z = lo + (hi - lo) * frac
            if not _clearance_ok(planes, z, radius_mm):
                continue
            for gx in np.linspace(-sw / 2 * 0.8, sw / 2 * 0.8, grid):
                for gy in np.linspace(-sh / 2 * 0.8, sh / 2 * 0.8, grid):
                    if rng is not None:
                        gx_j = gx + rng.uniform(-0.5, 0.5) * cell_x
                        gy_j = gy + rng.uniform(-0.5, 0.5) * cell_y
                    else:
                        gx_j, gy_j = gx, gy
                    got = _candidate(
                        planes, (float(gx_j), float(gy_j), z * 1000.0), radius_mm,
                        (x0, y0), z_rest, z_near, (span_x, span_y))
                    if got is not None:
                        cands.append(got)
    return _rank(cands, n, radius_mm, rng, prefer)
