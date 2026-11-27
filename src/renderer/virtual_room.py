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


# ---- state (keys) ----------------------------------------------------------

class RoomState:
    """Room on/off and the photo scale ``s``. Only ``--demo`` can use it:
    in the game (``demo=False``) the room stays off and every key is a no-op."""

    def __init__(self, demo: bool) -> None:
        self.demo = bool(demo)
        self.on = self.demo
        self.s = S_DEFAULT

    @property
    def shown(self) -> bool:
        return self.demo and self.on

    def toggle(self) -> bool:
        if self.demo:
            self.on = not self.on
        return self.shown

    def clamp(self, s_max_value: float) -> None:
        self.s = min(max(self.s, S_MIN), max(S_MIN, s_max_value))


    def zoom_in(self, s_max_value: float) -> None:
        if self.demo:
            self.s *= S_STEP
            self.clamp(s_max_value)

    def zoom_out(self, s_max_value: float) -> None:
        if self.demo:
            self.s /= S_STEP
            self.clamp(s_max_value)

    def reset(self) -> None:
        if self.demo:
            self.s = S_DEFAULT


# ---- one frame's placement of the layers -----------------------------------

@dataclass(frozen=True)
class RoomLayout:
    """Everything the draw pass needs to put the layers inside the room.

    ``hx``/``hy``: Kooima screen half sizes; ``centre_mm``: the screen centre;
    ``eye_mm``: eye relative to that centre (z = distance to the glass).
    """

    hx: float
    hy: float
    depth_mm: float
    e0_mm: float
    s: float
    eye_mm: tuple
    centre_mm: tuple
    width_px: int
    height_px: int

    def layer_size_mm(self, depth_mm: float) -> tuple[float, float]:
        z = -max(0.0, float(depth_mm))
        return layer_size_mm(2 * self.hx, 2 * self.hy, self.s, self.e0_mm,
                             self.depth_mm, z)

    def layer_matrix(self, depth_mm: float, quad_w_m: float, quad_h_m: float,
                     centre_x_m: float, centre_y_m: float,
                     plane_z_m: float) -> np.ndarray:
        """Model matrix replacing the plane's translation for one layer.

        The quad is rescaled (same factor in x and y) to fit the room size of
        its depth: it fills the width, or the height if the photo is taller
        than the screen, so it never spills out of the walls. It is moved to
        the screen centre and keeps its z (relief included). This also
        cancels today's overscan.
        """
        w_mm, h_mm = self.layer_size_mm(depth_mm)
        r = min((w_mm / 1000.0) / max(quad_w_m, 1e-9),
                (h_mm / 1000.0) / max(quad_h_m, 1e-9))
        m = np.eye(4)
        m[0, 0] = m[1, 1] = r
        m[0, 3] = self.centre_mm[0] / 1000.0 - centre_x_m * r
        m[1, 3] = self.centre_mm[1] / 1000.0 - centre_y_m * r
        m[2, 3] = plane_z_m
        return m

    def scissor_px(self, depth_mm: float) -> tuple[int, int, int, int]:
        """``(x, y, w, h)`` of the room cross-section at ``depth_mm`` in pixels."""
        x0, x1, y0, y1 = clip_rect_mm(-max(0.0, float(depth_mm)), self.eye_mm,
                                      self.hx, self.hy)
        px0 = (x0 + self.hx) / (2 * self.hx) * self.width_px
        px1 = (x1 + self.hx) / (2 * self.hx) * self.width_px
        py0 = (y0 + self.hy) / (2 * self.hy) * self.height_px
        py1 = (y1 + self.hy) / (2 * self.hy) * self.height_px
        x, y = max(0, int(np.floor(px0))), max(0, int(np.floor(py0)))
        x2 = min(self.width_px, int(np.ceil(px1)))
        y2 = min(self.height_px, int(np.ceil(py1)))
        return x, y, max(0, x2 - x), max(0, y2 - y)
