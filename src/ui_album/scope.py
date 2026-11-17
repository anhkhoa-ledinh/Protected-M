"""
TASK_game_feel_1118 Phase 6: Precision-gun scope (aim-down-sights) — the
Kooima sub-window magnifier, finer aim, steadied eye. Pure Python: no GL,
no ``VirtualCamera``/numpy dependency, so every number here is directly
testable against the task's own 0.5 design-px projection tolerance.

The magnification itself — warping the camera's screen corners and
steadying the eye — lives on ``src/renderer/camera.py``'s ``VirtualCamera``
(it already owns ``eye_position_mm`` and the Kooima corners, and both
rendering and the hit test must read the SAME numbers from one place).
This module only owns the INPUT side: the zoom spring, the right-mouse
hold state, and the finer-aim crosshair math — plus the pure glass-mm
conversion both sides need.
"""

from __future__ import annotations

from src.fx.anim import Spring

#: Spring(k=300, d=24), rest 1.0, scoped target 2.5 — same spring class
#: Phase 4's crosshair kick uses.
ZOOM_SPRING_K, ZOOM_SPRING_D = 300.0, 24.0
ZOOM_REST, ZOOM_MAX = 1.0, 2.5

DESIGN_SIZE = (1280.0, 720.0)

#: Look (drawn by ``v3.py``).
VIGNETTE_RADIUS_PX = 288.0
VIGNETTE_COL = "#0F1C2B"
VIGNETTE_ALPHA = 0.94
VIGNETTE_SOFT_PX = 6.0
RETICLE_COL = "#24415F"
RETICLE_THICK_PX = 1.5
RETICLE_GAP_PX = 6.0
RETICLE_ARM_PX = 28.0

#: Below this, treat the zoom as settled — avoids both a divide-by-
#: near-zero in the finer-aim/steadied-eye formulas and a camera that
#: keeps "scope_active" (and the Kooima warp) on for an imperceptible
#: residual zoom forever.
ZOOM_SETTLED_EPS = 0.01


def clamp(value: float, lo: float, hi: float) -> float:
    return min(hi, max(lo, value))


def design_to_glass_m(point, glass_mm, centre_mm) -> tuple[float, float]:
    """Design px -> glass point, in METRES, on the glass plane — the exact
    inverse of ``viewport_filter.project_to_design``/``hit.aim_point_mm``
    (same formula, those return mm; this returns m to match
    ``VirtualCamera``'s ``pa``/``pb``/``pc``)."""
    width_mm, height_mm = glass_mm
    sx = (point[0] / DESIGN_SIZE[0] - 0.5) * width_mm + centre_mm[0]
    sy = (0.5 - point[1] / DESIGN_SIZE[1]) * height_mm + centre_mm[1]
    return (sx * 0.001, sy * 0.001)


def scoped_kooima_corners(pa, pb, pc, c, z):
    """``pa' = c + (pa - c)/z``, same for ``pb``/``pc`` — ``c``/``pa``/
    ``pb``/``pc`` any 3-sequence of the same length (tuple or ndarray);
    returns plain tuples."""
    def warp(p):
        return tuple(c[i] + (p[i] - c[i]) / z for i in range(len(c)))
    return warp(pa), warp(pb), warp(pc)


def steadied_eye(eye, anchor, z):
    """``eye_at_scope_start + (eye - eye_at_scope_start)/z`` — keeps head
    jitter from being magnified by the scope. Rendering and the hit test
    must both use this (see module docstring)."""
    return tuple(anchor[i] + (eye[i] - anchor[i]) / z for i in range(len(anchor)))


class ScopeState:
    """Input-side scope state: right-mouse hold, the zoom spring, and the
    finer-aim crosshair. One instance lives as long as the ``V3UI`` that
    owns it (like ``ShotFeel``/``HoverFx``) — ``app.py`` drives it, never
    reaches into its internals."""

    def __init__(self) -> None:
        self.active = False
        self.zoom = Spring(ZOOM_REST, ZOOM_SPRING_K, ZOOM_SPRING_D)
        self.zoom.to = ZOOM_REST
        self._prev_mouse: tuple | None = None

    @property
    def z(self) -> float:
        return self.zoom.x

    @property
    def visually_active(self) -> bool:
        """``True`` while the zoom is still meaningfully above 1.0, even
        a moment after the trigger was released — the camera keeps
        rendering the (shrinking) magnified view smoothly instead of
        cutting it off the instant ``active`` goes false."""
        return self.zoom.x > ZOOM_REST + ZOOM_SETTLED_EPS

    def enter(self, mouse: tuple) -> None:
        self.active = True
        self.zoom.to = ZOOM_MAX
        self._prev_mouse = (float(mouse[0]), float(mouse[1]))

    def exit(self) -> None:
        self.active = False
        self.zoom.to = ZOOM_REST
        self._prev_mouse = None

    def step(self, dt: float) -> None:
        self.zoom.step(dt)

    def update_crosshair(self, crosshair: tuple, mouse: tuple) -> tuple:
        """Finer aim: a PER-FRAME mouse delta, scaled by the CURRENT zoom
        and added to the crosshair's running position — not a single
        distance-from-scope-entry divide, which would make the crosshair
        jump every frame the zoom spring is still settling (same raw
        total mouse movement would map to a different offset as z
        changes). Clamped to the design area."""
        if self._prev_mouse is None:
            self._prev_mouse = (float(mouse[0]), float(mouse[1]))
            return (float(crosshair[0]), float(crosshair[1]))
        z = max(1e-6, self.z)
        dx = (float(mouse[0]) - self._prev_mouse[0]) / z
        dy = (float(mouse[1]) - self._prev_mouse[1]) / z
        self._prev_mouse = (float(mouse[0]), float(mouse[1]))
        return (clamp(crosshair[0] + dx, 0.0, DESIGN_SIZE[0]),
               clamp(crosshair[1] + dy, 0.0, DESIGN_SIZE[1]))
