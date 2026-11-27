"""Virtual room behind the photo: pure geometry and colours (no GL).

All lengths are millimetres. The screen plane is z = 0 and behind it is z < 0.
The room is a box whose open front is the screen rectangle (half sizes
``hx``, ``hy``, centred on the origin) and whose back wall is at ``z = -D``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

#: Grid cell = screen width / GRID_CELLS_ACROSS, on every wall.
GRID_CELLS_ACROSS = 16
#: Largest channel of a line / fill colour.
LINE_PEAK = 215.0
FILL_PEAK = 26.0
#: Share of rows, at the top and at the bottom, that give the sky / ground colour.
BAND_FRACTION = 0.08

S_MIN = 0.6
S_DEFAULT = 1.0
S_STEP = 1.1


# ---- sizes -----------------------------------------------------------------

def s_max(e0_mm: float, depth_mm: float) -> float:
    """Largest photo scale: the photo has today's size at the centre."""
    return (e0_mm + depth_mm) / e0_mm


def layer_size_mm(screen_w_mm: float, screen_h_mm: float, s: float,
                  e0_mm: float, depth_mm: float, z_mm: float) -> tuple[float, float]:
    """Physical size of a layer at depth ``z_mm`` (<= 0) inside the room."""
    f = s * (e0_mm + abs(z_mm)) / (e0_mm + depth_mm)
    return screen_w_mm * f, screen_h_mm * f


def clip_rect_mm(z_mm: float, eye_mm, hx: float, hy: float
                 ) -> tuple[float, float, float, float]:
    """Room cross-section at depth ``z_mm`` projected on the screen plane.

    Returns ``(x0, x1, y0, y1)``. Its centre is ``e * (1 - E / (E + |z|))``
    and its half size is the room half size times ``E / (E + |z|)``.
    """
    ex, ey, ez = (float(v) for v in eye_mm)
    k = ez / (ez + abs(z_mm))
    cx, cy = ex * (1.0 - k), ey * (1.0 - k)
    return cx - hx * k, cx + hx * k, cy - hy * k, cy + hy * k


# ---- grid ------------------------------------------------------------------

def grid_cell_mm(screen_w_mm: float) -> float:
    return screen_w_mm / GRID_CELLS_ACROSS


def grid_divisions(screen_w_mm: float, screen_h_mm: float, depth_mm: float
                   ) -> tuple[int, int, int]:
    """Cells along x, y and the depth direction."""
    cell = grid_cell_mm(screen_w_mm)
    return (GRID_CELLS_ACROSS,
            max(1, round(screen_h_mm / cell)),
            max(1, round(depth_mm / cell)))


def _steps(lo: float, hi: float, n: int) -> list[float]:
    return [lo + (hi - lo) * i / n for i in range(n + 1)]


def grid_lines(hx: float, hy: float, depth_mm: float) -> dict[str, list[tuple]]:
    """Grid segments ``((x, y, z), (x, y, z))`` per wall, edges included."""
    nx, ny, nz = grid_divisions(2 * hx, 2 * hy, depth_mm)
    xs, ys, zs = _steps(-hx, hx, nx), _steps(-hy, hy, ny), _steps(0.0, -depth_mm, nz)
    z0, zf = 0.0, -depth_mm
    walls: dict[str, list[tuple]] = {}
    for name, y in (("floor", -hy), ("ceiling", hy)):
        walls[name] = ([((x, y, z0), (x, y, zf)) for x in xs]
                       + [((-hx, y, z), (hx, y, z)) for z in zs])
    for name, x in (("left", -hx), ("right", hx)):
        walls[name] = ([((x, y, z0), (x, y, zf)) for y in ys]
                       + [((x, -hy, z), (x, hy, z)) for z in zs])
    walls["back"] = ([((x, -hy, zf), (x, hy, zf)) for x in xs]
                     + [((-hx, y, zf), (hx, y, zf)) for y in ys])
    return walls


# ---- colours ---------------------------------------------------------------

def _scaled(rgb, peak: float) -> np.ndarray:
    """Scale ``rgb`` so its largest channel is ``peak`` (grey if it is black)."""
    rgb = np.asarray(rgb, dtype=np.float64)
    top = float(rgb.max())
    if top < 1.0:
        return np.full(3, peak)
    return rgb * (peak / top)


@dataclass(frozen=True)
class RoomColors:
    """Sky (``ceiling``) and ground (``floor``) mean colours of a photo, 0..255."""

    ceiling: tuple[float, float, float]
    floor: tuple[float, float, float]

    def _pick(self, wall: str, t: float, peak: float) -> tuple[float, float, float]:
        ceil = _scaled(self.ceiling, peak)
        floor = _scaled(self.floor, peak)
        if wall == "ceiling":
            out = ceil
        elif wall == "floor":
            out = floor
        elif wall == "side":            # t = 0 at the bottom, 1 at the top
            t = min(max(float(t), 0.0), 1.0)
            out = floor + (ceil - floor) * t
        elif wall == "back":
            out = (ceil + floor) / 2.0
        else:
            raise ValueError(f"unknown wall {wall!r}")
        return tuple(float(v) for v in out)

    def line(self, wall: str, t: float = 0.5) -> tuple[float, float, float]:
        """Grid line colour of ``wall`` (largest channel 215)."""
        return self._pick(wall, t, LINE_PEAK)

    def fill(self, wall: str, t: float = 0.5) -> tuple[float, float, float]:
        """Wall fill colour (largest channel 26)."""
        return self._pick(wall, t, FILL_PEAK)


def room_colors(photo) -> RoomColors:
    """Ceiling / floor colours from the original photo (H x W x 3 or 4, uint8)."""
    arr = np.asarray(photo)
    rows = arr.shape[0]
    n = max(1, int(round(rows * BAND_FRACTION)))
    top = arr[:n, :, :3].reshape(-1, 3).mean(axis=0)
    bottom = arr[rows - n:, :, :3].reshape(-1, 3).mean(axis=0)
    return RoomColors(tuple(float(v) for v in top), tuple(float(v) for v in bottom))
