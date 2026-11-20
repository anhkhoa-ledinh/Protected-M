"""
TASK_game_feel_1118 Phase 6: the scope's look — ``_draw_scope_vignette``/
``_draw_scope_reticle`` in ``src/ui_album/v3.py``, and that the normal
crosshair (``_draw_aim``) steps aside for the reticle while scoped. Fake
painter, no GL — same style as ``tests/test_aim_cross.py``.
"""

from __future__ import annotations

import pytest

from src.ui_album.v3 import (V3State, _draw_aim, _draw_scope_mini_readout,
                             _draw_scope_reticle, _draw_scope_vignette)


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


# ---- Fix 10.3: scope mini readout (ammo/timer/kills just outside the ring) -

class FakeReadoutPainter:
    def __init__(self) -> None:
        self.labels: list = []

    def outlined_label(self, centre, text, font_spec, fg, outline_col,
                       outline_px, alpha: float = 1.0) -> None:
        self.labels.append((tuple(float(v) for v in centre), text))


def test_no_mini_readout_when_not_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.texts = {"ammo": ["9", "/12"], "timer": "02:20", "kills": ["1", "/4"]}
    painter = FakeReadoutPainter()
    _draw_scope_mini_readout(painter, state)
    assert painter.labels == []


def test_mini_readout_shows_ammo_timer_kills_when_scoped():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    state.texts = {"ammo": ["9", "/12"], "timer": "02:20", "kills": ["1", "/4"]}
    painter = FakeReadoutPainter()
    _draw_scope_mini_readout(painter, state)
    texts = {text for _centre, text in painter.labels}
    assert texts == {"9/12", "02:20", "1/4"}


def test_mini_readout_timer_sits_directly_above_the_crosshair():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    state.texts = {"timer": "02:20"}
    painter = FakeReadoutPainter()
    _draw_scope_mini_readout(painter, state)
    assert len(painter.labels) == 1
    (cx, cy), text = painter.labels[0]
    assert text == "02:20"
    assert cx == pytest.approx(640.0)
    assert cy < 360.0  # above the crosshair


def test_mini_readout_ammo_and_kills_are_mirrored_left_right():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    state.texts = {"ammo": ["9", "/12"], "kills": ["1", "/4"]}
    painter = FakeReadoutPainter()
    _draw_scope_mini_readout(painter, state)
    by_text = {text: centre for centre, text in painter.labels}
    ammo_c, kills_c = by_text["9/12"], by_text["1/4"]
    assert ammo_c[0] > 640.0   # lower-right
    assert kills_c[0] < 640.0  # lower-left
    assert ammo_c[1] == pytest.approx(kills_c[1])  # mirrored -> same height
    assert ammo_c[1] > 360.0   # both below the crosshair


def test_mini_readout_uses_kills_you_for_pvp_hunt():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    state.texts = {"kills_you": ["0", "/4"]}
    painter = FakeReadoutPainter()
    _draw_scope_mini_readout(painter, state)
    texts = {text for _centre, text in painter.labels}
    assert "0/4" in texts


def test_no_mini_readout_in_shot_mode():
    state = V3State()
    state.aim = (640.0, 360.0)
    state.scope_zoom = 2.5
    state.shot = True
    state.texts = {"timer": "02:20"}
    painter = FakeReadoutPainter()
    _draw_scope_mini_readout(painter, state)
    assert painter.labels == []


# ---- outlined_label: real GL, actually renders without crashing ----------

def test_outlined_label_renders_on_real_gl(gl_context):
    from pathlib import Path

    from src.ui_album.v3 import V3Manifest, V3Painter

    manifest = V3Manifest(Path(".").resolve())
    painter = V3Painter(manifest, Path(".").resolve() / "assets" / "shaders")
    painter.initialize()
    painter.set_frame((1280, 720))
    painter.begin()
    painter.outlined_label((640.0, 360.0), "9/12",
                           {"ff": "Nunito", "fw": 800, "fs": 18.0},
                           "#FFFFFF", "#24415F", 2.0)
    painter.end()
    painter.release()
