"""
TASK_game_feel_1118 Phase 6: the scope's look — ``_draw_scope_vignette``/
``_draw_scope_reticle`` in ``src/ui_album/v3.py``, and that the normal
crosshair (``_draw_aim``) steps aside for the reticle while scoped. Fake
painter, no GL — same style as ``tests/test_aim_cross.py``.
"""

from __future__ import annotations

from src.ui_album.v3 import (V3State, _draw_aim, _draw_scope_reticle,
                             _draw_scope_vignette)


class FakeWhiteCache:
    def white(self):
        return ("white-tex", 1, 1)


class FakePilCache:
    def get(self, key, build):
        array, _meta = build()
        return ("vignette-tex", array.shape[1], array.shape[0], None)


class FakePainter:
    def __init__(self, scale: float = 1.0) -> None:
        self.fills: list = []
        self.quads: list = []
        self.cache = FakeWhiteCache()
        self.pil = FakePilCache()
        self.scale = scale

    def fill(self, rect, colour) -> None:
        self.fills.append((tuple(float(v) for v in rect), colour))

    def quad(self, rect, texture, tint=(1.0, 1.0, 1.0, 1.0), uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        self.quads.append((tuple(float(v) for v in rect), texture))


# ---- vignette ---------------------------------------------------------
def test_no_vignette_when_not_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    painter = FakePainter()
    _draw_scope_vignette(painter, state)
    assert painter.fills == []
    assert painter.quads == []


def test_vignette_draws_strips_and_a_square_when_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    painter = FakePainter()
    _draw_scope_vignette(painter, state)
    assert len(painter.fills) == 4            # top/bottom/left/right strips
    assert len(painter.quads) == 1             # the gradient square


def test_vignette_square_centred_on_the_crosshair():
    state = V3State()
    state.aim = (900.0, 250.0)
    state.scope_zoom = 2.5
    painter = FakePainter()
    _draw_scope_vignette(painter, state)
    (x, y, w, h), _tex = painter.quads[0]
    assert x + w / 2.0 == 900.0
    assert y + h / 2.0 == 250.0


def test_no_vignette_in_shot_mode():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    state.shot = True
    painter = FakePainter()
    _draw_scope_vignette(painter, state)
    assert painter.fills == [] and painter.quads == []


# ---- reticle ------------------------------------------------------------
def test_no_reticle_when_not_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    painter = FakePainter()
    _draw_scope_reticle(painter, state)
    assert painter.fills == []


def test_reticle_draws_four_arms_with_a_centre_gap():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    painter = FakePainter()
    _draw_scope_reticle(painter, state)
    assert len(painter.fills) == 4
    for (rx, ry, rw, rh), _colour in painter.fills:
        covers_centre = rx < 640.0 < rx + rw and ry < 360.0 < ry + rh
        assert not covers_centre


# ---- the normal crosshair steps aside while scoped -----------------------
def test_normal_crosshair_hidden_while_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    painter = FakePainter()
    _draw_aim(painter, state)
    assert painter.fills == []


def test_normal_crosshair_shown_when_not_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    painter = FakePainter()
    _draw_aim(painter, state)
    assert len(painter.fills) == 8
