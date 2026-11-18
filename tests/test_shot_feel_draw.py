"""
TASK_game_feel_1118 Phase 4: the draw-side of shot feel —
``_draw_hit_markers``/``_draw_crit_impact`` in ``src/ui_album/v3.py``,
plus the screen-shake offset applied to the 3D scene viewport rect in
``_draw_photos``. Fake painters only, no GL — same style as
``tests/test_aim_cross.py``.

TASK_game_feel_fix_1118 Fix 6: the tracer and its tests (``_draw_tracers``
and everything below "tracers") are gone — removed outright per the
user's playtest note, not disabled.
"""

from __future__ import annotations

import pytest

from src.ui_album.shot_feel import CRIT_IMPACT_ALPHA, HITMARK_CRIT_COL, HITMARK_OUTLINE_COL
from src.ui_album.v3 import (DESIGN_SIZE, V3State, _draw_crit_impact,
                             _draw_hit_markers, _draw_photos, rgba)


class FakeWhiteCache:
    def white(self):
        return ("white-tex", 1, 1)


class FakePainter:
    """Records every flat-coloured quad (``fill``) and arbitrary-corner
    quad (``quad_corners``, used for anything not axis-aligned)."""

    def __init__(self) -> None:
        self.fills: list = []
        self.quads: list = []
        self.cache = FakeWhiteCache()

    def fill(self, rect, colour) -> None:
        self.fills.append((tuple(float(v) for v in rect), colour))

    def quad_corners(self, corners, texture, tint=(1.0, 1.0, 1.0, 1.0),
                     uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        self.quads.append((tuple(corners), texture, tint))


# ---- hit markers -------------------------------------------------------
def test_no_hit_markers_drawn_when_list_is_empty():
    state = V3State()
    painter = FakePainter()
    _draw_hit_markers(painter, state)
    assert painter.quads == []


def test_hit_marker_draws_4_ticks_in_2_passes():
    state = V3State()
    state.hit_markers = [(100.0, 200.0, 0.0, False)]
    painter = FakePainter()
    _draw_hit_markers(painter, state)
    assert len(painter.quads) == 8  # 4 ticks x (outline + core)


def test_hit_marker_crit_colour_differs_from_normal():
    state = V3State()
    state.hit_markers = [(100.0, 200.0, 0.0, True)]
    painter = FakePainter()
    _draw_hit_markers(painter, state)
    core_cols = {q[2] for q in painter.quads}
    assert rgba(HITMARK_CRIT_COL, 1.0) in core_cols


def test_hit_marker_outline_colour_present_both_normal_and_crit():
    for crit in (False, True):
        state = V3State()
        state.hit_markers = [(100.0, 200.0, 0.0, crit)]
        painter = FakePainter()
        _draw_hit_markers(painter, state)
        cols = {q[2] for q in painter.quads}
        assert rgba(HITMARK_OUTLINE_COL, 1.0) in cols


def test_expired_hit_marker_is_not_drawn():
    state = V3State()
    state.hit_markers = [(100.0, 200.0, 0.18, False)]  # age >= life
    painter = FakePainter()
    _draw_hit_markers(painter, state)
    assert painter.quads == []


def test_crit_hit_marker_is_bigger_than_a_normal_one():
    normal_state = V3State()
    normal_state.hit_markers = [(0.0, 0.0, 0.0, False)]
    normal_painter = FakePainter()
    _draw_hit_markers(normal_painter, normal_state)

    crit_state = V3State()
    crit_state.hit_markers = [(0.0, 0.0, 0.0, True)]
    crit_painter = FakePainter()
    _draw_hit_markers(crit_painter, crit_state)

    def farthest(quads):
        return max(abs(c) for corners, *_ in quads for p in corners for c in p)

    assert farthest(crit_painter.quads) > farthest(normal_painter.quads)


# ---- crit impact frame -------------------------------------------------
def test_no_crit_impact_frame_when_alpha_is_zero():
    state = V3State()
    painter = FakePainter()
    _draw_crit_impact(painter, state)
    assert painter.fills == []


def test_crit_impact_frame_covers_the_full_design_box():
    state = V3State()
    state.crit_impact_alpha = CRIT_IMPACT_ALPHA
    painter = FakePainter()
    _draw_crit_impact(painter, state)
    assert len(painter.fills) == 1
    rect, colour = painter.fills[0]
    assert rect == (0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1])
    assert colour[3] == pytest.approx(CRIT_IMPACT_ALPHA)


def test_shot_mode_draws_no_crit_impact_frame():
    state = V3State()
    state.shot = True
    state.crit_impact_alpha = CRIT_IMPACT_ALPHA
    painter = FakePainter()
    _draw_crit_impact(painter, state)
    assert painter.fills == []


# ---- scene shake moves the viewport rect, nothing else ------------------
class _RecordingManifest:
    pass


def test_viewport_rect_is_shifted_by_scene_shake():
    manifest = _RecordingManifest()
    record = {"photos": [{"kind": "viewport", "rect": (100.0, 50.0, 400.0, 300.0)}]}
    state = V3State()
    state.scene_shake = (3.0, -2.0)
    seen = []
    state.viewport_draw = lambda rect: seen.append(rect)
    _draw_photos(None, manifest, record, state)
    assert seen == [(103.0, 48.0, 400.0, 300.0)]


def test_viewport_rect_unshifted_when_shake_is_zero():
    manifest = _RecordingManifest()
    record = {"photos": [{"kind": "viewport", "rect": (100.0, 50.0, 400.0, 300.0)}]}
    state = V3State()
    seen = []
    state.viewport_draw = lambda rect: seen.append(rect)
    _draw_photos(None, manifest, record, state)
    assert seen == [(100.0, 50.0, 400.0, 300.0)]
