"""
TASK_game_feel_fix_1118 Fix 5: scene-anchored overlays (shot marks, the
Sakuga hit_kill/hit_crit/hit_block fx) track their GLASS POINT while
scoped, not the design-px spot they were drawn at the moment they
appeared. The 1118 report's "fixed point" argument only covers a mark at
the CURRENT crosshair — this is the fix for every OTHER mark.
"""

from __future__ import annotations

import math

import pytest

from src.fx.player import FxPlayer
from src.fx.spec import load_spec
from src.ui_album.scope import VIGNETTE_RADIUS_PX, screen_scope_transform
from src.ui_album.v3 import V3State
from src.ui_album.v3_screen_draw import _draw_hit_markers, _draw_shots


# ---- pure transform ------------------------------------------------------
def test_unscoped_shot_then_scoping_elsewhere_puts_the_mark_at_b_plus_z_a_minus_b():
    """The task's own scenario: a shot at A, THEN scope with the crosshair
    at B — the mark must render at B + z(A - B)."""
    a = (500.0, 300.0)
    b = (700.0, 400.0)
    z = 2.0
    result = screen_scope_transform(a, b, z)
    expected = (b[0] + z * (a[0] - b[0]), b[1] + z * (a[1] - b[1]))
    assert result == pytest.approx(expected)


def test_crosshair_moving_by_delta_moves_an_active_mark_by_z_times_delta():
    """``s = c + z(g - c)``: differentiating w.r.t. a crosshair shift
    delta (g, z fixed) gives ``ds = delta - z*delta = (1 - z)*delta`` —
    the task's own "moves by z*delta relative to the new crosshair" is
    this same statement in the moving frame: relative to c (which moved
    by delta), the mark's position ``s - c = z(g - c)`` changes by
    ``-z*delta`` for a fixed g — i.e. the mark moves z*delta OPPOSITE the
    crosshair's own motion, AS SEEN FROM the crosshair's new frame."""
    a = (500.0, 300.0)
    z = 2.5
    c1 = (640.0, 360.0)
    delta = (30.0, -10.0)
    c2 = (c1[0] + delta[0], c1[1] + delta[1])
    s1 = screen_scope_transform(a, c1, z)
    s2 = screen_scope_transform(a, c2, z)
    moved = (s2[0] - s1[0], s2[1] - s1[1])
    assert moved == pytest.approx(((1.0 - z) * delta[0], (1.0 - z) * delta[1]))
    # Relative to the crosshair (which itself moved by delta), the mark
    # moved by -z*delta — exactly the task's "moves by z*delta" (in
    # magnitude, opposite in direction to the crosshair's own shift).
    moved_relative_to_crosshair = (moved[0] - delta[0], moved[1] - delta[1])
    assert moved_relative_to_crosshair == pytest.approx((-z * delta[0], -z * delta[1]))


def test_crosshair_itself_is_still_a_fixed_point():
    c = (640.0, 360.0)
    assert screen_scope_transform(c, c, 2.5) == pytest.approx(c)


def test_identity_when_z_is_1():
    a = (111.0, 222.0)
    c = (640.0, 360.0)
    assert screen_scope_transform(a, c, 1.0) == pytest.approx(a)


# ---- shot marks / hit markers drawing ------------------------------------
class FakeWhiteCache:
    def white(self):
        return ("white-tex", 1, 1)


class FakePainter:
    def __init__(self) -> None:
        self.fills: list = []
        self.discs: list = []
        self.quads: list = []
        self.cache = FakeWhiteCache()

    def fill(self, rect, colour) -> None:
        self.fills.append((tuple(float(v) for v in rect), colour))

    def disc(self, rect, colour) -> None:
        self.discs.append((tuple(float(v) for v in rect), colour))

    def quad_corners(self, corners, texture, tint=(1.0, 1.0, 1.0, 1.0),
                     uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        self.quads.append((tuple(corners), texture, tint))


def _disc_centre_xs(painter: FakePainter) -> list:
    return [rect[0] + rect[2] * 0.5 for rect, _c in painter.discs]


def test_shot_mark_unscoped_draws_at_its_own_stored_position():
    state = V3State()
    state.shots = [(500.0, 300.0, 0.0, True)]
    painter = FakePainter()
    _draw_shots(painter, state)
    assert painter.discs
    assert any(abs(x - 500.0) < 1.0 for x in _disc_centre_xs(painter))


def test_shot_mark_scoped_moves_to_b_plus_z_times_a_minus_b():
    state = V3State()
    a = (600.0, 350.0)
    state.shots = [(a[0], a[1], 0.0, True)]
    state.aim = (650.0, 370.0)          # close enough that the result stays inside the scope ring
    state.scope_zoom = 1.5
    painter = FakePainter()
    _draw_shots(painter, state)
    expected = screen_scope_transform(a, state.aim, state.scope_zoom)
    xs = _disc_centre_xs(painter)
    assert xs
    assert any(abs(x - expected[0]) < 1.0 for x in xs)
    # and NOT at the original unscoped spot
    assert not any(abs(x - a[0]) < 2.0 for x in xs)


def test_shot_mark_hidden_outside_the_scope_circle():
    state = V3State()
    far_a = (0.0, 0.0)          # far from the crosshair — will land outside the ring once zoomed
    state.shots = [(far_a[0], far_a[1], 0.0, True)]
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    painter = FakePainter()
    _draw_shots(painter, state)
    assert painter.discs == []


def test_hit_marker_scoped_moves_the_same_way():
    state = V3State()
    a = (620.0, 340.0)
    state.hit_markers = [(a[0], a[1], 0.0, False)]
    state.aim = (660.0, 380.0)          # close enough that the result stays inside the scope ring
    state.scope_zoom = 1.8
    painter = FakePainter()
    _draw_hit_markers(painter, state)
    expected = screen_scope_transform(a, state.aim, state.scope_zoom)
    # average of all quad corners should cluster around the expected centre
    all_x = [p[0] for corners, _t, _c in painter.quads for p in corners]
    assert any(abs(x - expected[0]) < 30.0 for x in all_x)


# ---- FxPlayer.draw_list(transform=...) ------------------------------------
@pytest.fixture
def spec():
    return load_spec(".")


def test_draw_list_transform_applied_to_every_corner(spec):
    player = FxPlayer(spec, seed=0)
    player.spawn("ui_tap", 500.0, 300.0, words=False)
    plain = player.draw_list()
    assert plain

    def shift(p):
        return (p[0] + 50.0, p[1] - 20.0)

    shifted = player.draw_list(transform=shift)
    assert len(shifted) == len(plain)
    for plain_cmd, shifted_cmd in zip(plain, shifted):
        for (px, py), (sx, sy) in zip(plain_cmd.corners, shifted_cmd.corners):
            assert (sx, sy) == pytest.approx((px + 50.0, py - 20.0))


def test_draw_list_scope_transform_scales_the_sprite_with_the_scene(spec):
    """Applying the scope transform PER CORNER (not just to an anchor)
    scales a live effect's size along with the rest of the magnified
    scene, not just its position."""
    player = FxPlayer(spec, seed=0)
    player.spawn("ui_tap", 500.0, 300.0, words=False)
    plain = player.draw_list()[0]
    crosshair, z = (640.0, 360.0), 2.0
    scoped = player.draw_list(
        transform=lambda p: screen_scope_transform(p, crosshair, z))[0]

    def span(corners):
        xs = [p[0] for p in corners]
        return max(xs) - min(xs)

    assert span(scoped.corners) == pytest.approx(span(plain.corners) * z, rel=1e-6)
