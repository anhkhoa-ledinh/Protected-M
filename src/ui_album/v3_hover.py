"""
Hover/press feedback for v3 (TASK_ui_sakuga_1117 Phase 5): "Today a player
cannot tell what is clickable, where the pointer is, or whether a click
registered." Every screen gets it EXCEPT the three match screens
(``PLAY``/``PVP_HUNT``/``PVP_HIDE``), which keep today's plain behaviour —
they also have no hotspots at all, so this is mostly a documented no-op
there, not a real behaviour change.

A ``HoverFx`` instance holds the only state that must survive across
frames: one spring per currently-hovered-or-recently-left hotspot, plus a
cache of each hotspot's sampled background luminance and measured text
bounds (``underline`` kind) — read from ``base.png`` with PIL ONCE per
hotspot, never per frame. ``cursor_for`` is the one pure, directly
testable piece: screen/hover/state in, a cursor-shape string out.
"""

from __future__ import annotations

from collections import namedtuple

from PIL import Image

from src.fx import screens as fx_screens
from src.fx.anim import Spring
from src.ui_album.v3 import FOCUS_INSET, FOCUS_PX, INK
from src.ui_album.v3_hit import (hotspot_rect, hotspot_shape,
                                 measure_ink_bounds, ring_draw_radius)
from src.ui_album.v3_painter import rgba
from src.ui_album.v3_text_field import CARET_COL

#: Hover spring: 0 (idle) -> 1 (fully hovered) and back, stepped with real dt.
HOVER_SPRING_K, HOVER_SPRING_D = 420.0, 26.0

#: Ring: 2px #3B72B4, growing 2px outward as it fades in.
RING_PX, RING_GROW_PX, RING_ALPHA = 2.0, 2.0, 0.85
#: Inside tint: white on a dark element, ink (#24415F) on a light one.
TINT_DARK_ALPHA, TINT_LIGHT_ALPHA = 0.12, 0.05
#: An already-selected row/tab still hovers, just at half ring alpha.
SELECTED_RING_FACTOR = 0.5

#: underline kind: a thin line under the measured text, no ring.
UNDERLINE_PX = 1.5

#: thumb kind: a soft halo around the slider knob at its CURRENT position.
THUMB_HALO_ALPHA = 0.18
#: ``state.sprites`` key v3_values.py positions the volume knob sprite at.
VOLUME_KNOB_SPRITE = "volume_knob"

#: Press: tint turns ink-coloured and stronger, ring sits inset instead of
#: grown outward. Clicks fire on press (v3_routes.py), so this is what's
#: on screen at the same instant a Sakuga burst spawns (Phase 6).
PRESS_TINT_COL, PRESS_TINT_ALPHA, PRESS_RING_INSET = "#24415F", 0.10, 1.0

#: Luminance below this (0-255, sRGB-weighted) counts as a "dark" element.
DARK_LUMINANCE_THRESHOLD = 128.0

_Sample = namedtuple("_Sample", ("is_dark", "ink_rect"))


def cursor_for(screen: str, hover: str | None, state) -> str:
    """OS cursor shape for the current frame: ``"hand"`` over an enabled
    clickable hotspot, ``"ibeam"`` over a text field, ``"arrow"``
    everywhere else (including every match screen, and disabled
    elements, if ``state`` ever carries a ``disabled`` set). Pure —
    no GL/GLFW — so it's unit-testable on its own."""
    if state.shot or fx_screens.is_match_screen(screen) or hover is None:
        return "arrow"
    if hover in getattr(state, "disabled", ()):
        return "arrow"
    if hover.startswith("field_"):
        return "ibeam"
    return "hand"


def _find_hotspot(record: dict, name: str) -> dict | None:
    for hotspot in record.get("hotspots", ()):
        if hotspot["n"] == name:
            return hotspot
    return None


class HoverFx:
    """Per-hotspot hover springs + a small per-hotspot pixel-sample cache.
    One instance per running app (``V3UI`` owns it); construct a fresh one
    in tests that need isolation."""

    def __init__(self) -> None:
        self._springs: dict[str, Spring] = {}
        self._sampled: dict[tuple, _Sample] = {}

    # ---- spring bookkeeping ------------------------------------------------
    def step(self, screen: str, hover: str | None, dt: float) -> None:
        """Advance every tracked spring toward 1 (hovered) or 0 (not);
        drop ones that reached ~0 and aren't the current hover."""
        if fx_screens.is_match_screen(screen):
            self._springs.clear()
            return
        for name in list(self._springs):
            spring = self._springs[name]
            spring.to = 1.0 if name == hover else 0.0
            spring.step(dt)
            if spring.x <= 0.001 and name != hover:
                del self._springs[name]
        if hover is not None and hover not in self._springs:
            spring = Spring(0.0, HOVER_SPRING_K, HOVER_SPRING_D)
            spring.to = 1.0
            self._springs[hover] = spring

    def value(self, name: str) -> float:
        spring = self._springs.get(name)
        return 0.0 if spring is None else max(0.0, min(1.0, spring.x))

    # ---- one-time pixel sampling -----------------------------------------
    def _sample(self, manifest, screen: str, record: dict, name: str,
                rect, need_ink: bool) -> _Sample:
        key = (screen, name)
        cached = self._sampled.get(key)
        if cached is not None and (cached.ink_rect is not None or not need_ink):
            return cached
        is_dark, ink_rect = True, None
        path = manifest.path(record.get("base"))
        if path is not None and path.exists():
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    scale = manifest.image_scale
                    cx = rect[0] + rect[2] * 0.5
                    cy = rect[1] + rect[3] * 0.5
                    ix = min(rgb.width - 1, max(0, int(round(cx * scale))))
                    iy = min(rgb.height - 1, max(0, int(round(cy * scale))))
                    r, g, b = rgb.getpixel((ix, iy))
                    is_dark = (0.2126 * r + 0.7152 * g + 0.0722 * b) < \
                        DARK_LUMINANCE_THRESHOLD
                    if need_ink:
                        ink_rect = measure_ink_bounds(rgb, rect, scale)
            except (OSError, ValueError):
                pass
        sample = _Sample(is_dark, ink_rect)
        self._sampled[key] = sample
        return sample

    # ---- drawing -----------------------------------------------------
    def draw(self, painter, manifest, screen: str, record: dict, state) -> None:
        if state.shot or fx_screens.is_match_screen(screen):
            return
        for hotspot in record.get("hotspots", ()):
            name = hotspot["n"]
            h = self.value(name)
            if h <= 0.0:
                continue
            rect = hotspot_rect(screen, hotspot)
            shape = hotspot_shape(name, rect)
            pressed = bool(state.pressed) and name == state.hover
            selected = name in state.selected
            self._draw_one(painter, manifest, screen, record, state, name,
                           rect, shape, h, pressed, selected)
        if state.focus is not None:
            self._draw_focus(painter, manifest, screen, record, state, state.focus)

    def _draw_one(self, painter, manifest, screen, record, state, name, rect,
                 shape, h, pressed, selected) -> None:
        if shape.kind == "none":
            return
        if shape.kind == "underline":
            sample = self._sample(manifest, screen, record, name, rect, True)
            x, y, w, _h = sample.ink_rect if sample.ink_rect is not None else rect
            painter.fill((x, rect[1] + rect[3] - 2.0, w, UNDERLINE_PX),
                         rgba(CARET_COL, h))
            return
        if shape.kind == "thumb":
            self._draw_thumb(painter, state, THUMB_HALO_ALPHA * h, RING_GROW_PX * h)
            return
        radius = ring_draw_radius(shape, rect) or 0.0
        alpha = RING_ALPHA * h * (SELECTED_RING_FACTOR if selected else 1.0)
        if pressed:
            inset = PRESS_RING_INSET
            ring_rect = (rect[0] + inset, rect[1] + inset,
                        max(0.0, rect[2] - 2.0 * inset),
                        max(0.0, rect[3] - 2.0 * inset))
            ring_radius = max(0.0, radius - inset)
            tint_col, tint_alpha = PRESS_TINT_COL, PRESS_TINT_ALPHA
        else:
            grow = RING_GROW_PX * h
            ring_rect = (rect[0] - grow, rect[1] - grow,
                        rect[2] + 2.0 * grow, rect[3] + 2.0 * grow)
            ring_radius = radius + grow
            sample = self._sample(manifest, screen, record, name, rect, False)
            tint_col = "#FFFFFF" if sample.is_dark else INK
            tint_alpha = (TINT_DARK_ALPHA if sample.is_dark else TINT_LIGHT_ALPHA) * h
        painter.rounded(rect, radius, rgba(tint_col, tint_alpha))
        painter.rounded_frame(ring_rect, rgba(CARET_COL, alpha), RING_PX, ring_radius)

    def _draw_thumb(self, painter, state, alpha: float, grow: float) -> None:
        knob = state.sprites.get(VOLUME_KNOB_SPRITE)
        if not isinstance(knob, (tuple, list)) or alpha <= 0.0:
            return
        kx, ky, kw, kh = (float(v) for v in knob)
        cx, cy = kx + kw * 0.5, ky + kh * 0.5
        radius = max(kw, kh) * 0.5 + grow
        painter.disc((cx - radius, cy - radius, radius * 2.0, radius * 2.0),
                    rgba(CARET_COL, alpha))

    # ---- keyboard focus (same shape table, no spring) ---------------------
    def _draw_focus(self, painter, manifest, screen, record, state, name) -> None:
        hotspot = _find_hotspot(record, name)
        if hotspot is None:
            return
        rect = hotspot_rect(screen, hotspot)
        shape = hotspot_shape(name, rect)
        if shape.kind == "none":
            return
        if shape.kind == "underline":
            sample = self._sample(manifest, screen, record, name, rect, True)
            x, y, w, _h = sample.ink_rect if sample.ink_rect is not None else rect
            painter.fill((x, rect[1] + rect[3] - 2.0, w, FOCUS_PX), rgba(CARET_COL))
            return
        if shape.kind == "thumb":
            self._draw_thumb(painter, state, 1.0, FOCUS_INSET)
            return
        radius = ring_draw_radius(shape, rect) or 0.0
        ring_rect = (rect[0] - FOCUS_INSET, rect[1] - FOCUS_INSET,
                    rect[2] + 2.0 * FOCUS_INSET, rect[3] + 2.0 * FOCUS_INSET)
        painter.rounded_frame(ring_rect, rgba(CARET_COL), FOCUS_PX,
                             radius + FOCUS_INSET if radius else 0.0)
