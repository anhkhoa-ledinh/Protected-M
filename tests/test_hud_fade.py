"""
TASK_game_feel_fix_1118 Fix 10.2/10.3: HUD card fade (proximity and
scoped).

- Fix 10.2: a card whose (grown) rect contains the crosshair fades to
  0.22 via Spring(k=420, d=26), unscoped, every gun.
- Fix 10.3: scoped (Precision gun), every card fades to 0 instead.

Pure-Python where possible (Spring convergence); the proximity/scoped
fade tests need a real GL ``V3UI`` (card rects come from the real
``top.png`` assets) but run with ``HCPD_NO_FX=1`` to skip the heavy fx
preload entirely — this file does not need fx at all, and
test_v3ui_fx_hooks.py already covers that path.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from src.fx.anim import Spring

ROOT = Path(".").resolve()


# ---- Spring convergence (pure) --------------------------------------------

def _settle(spring: Spring, to: float, seconds: float = 1.0, step: float = 1 / 60) -> float:
    spring.to = to
    elapsed = 0.0
    while elapsed < seconds:
        spring.step(step)
        elapsed += step
    return spring.x


def test_spring_converges_to_the_proximity_alpha_target():
    spring = Spring(1.0, 420.0, 26.0)
    assert _settle(spring, 0.22) == pytest.approx(0.22, abs=0.01)


def test_spring_converges_to_zero_when_scoped():
    spring = Spring(1.0, 420.0, 26.0)
    assert _settle(spring, 0.0) == pytest.approx(0.0, abs=0.01)


def test_spring_returns_to_one_when_target_goes_back_up():
    spring = Spring(1.0, 420.0, 26.0)
    _settle(spring, 0.22)
    assert _settle(spring, 1.0) == pytest.approx(1.0, abs=0.01)


# ---- V3UI fade targets: inside / outside / scoped (real GL, no fx) --------

@pytest.fixture
def ui(gl_context, monkeypatch):
    monkeypatch.setenv("HCPD_NO_FX", "1")
    from src.ui_album.options import AlbumOptions
    from src.ui_album.v3_ui import V3UI
    from src.ui_album import v3_flow as F

    options = AlbumOptions(enabled=True, guest=True, ui="v3")
    instance = V3UI(ROOT, ROOT / "assets" / "shaders", options)
    instance.initialize()
    instance.painter.set_frame((1280, 720))
    instance.state.go(F.PLAY)
    yield instance
    instance.release()


def _settle_fade(ui, seconds: float = 1.0, step: float = 1 / 60) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        ui._tick_hud_fade(step)
        elapsed += step


def _first_fadeable_card_centre(ui):
    from src.ui_album import v3_flow as F
    from src.ui_album.hud_cards import cards_for_screen

    cards = cards_for_screen(ui.manifest, F.PLAY)
    index = next(i for i, c in enumerate(cards) if c.fades)
    x, y, w, h = cards[index].rect
    return index, (x + w * 0.5, y + h * 0.5)


def test_crosshair_inside_a_card_fades_only_that_card(ui):
    index, centre = _first_fadeable_card_centre(ui)
    ui.mouse = centre
    _settle_fade(ui)
    alpha = ui.hud_card_alpha
    assert alpha[index] == pytest.approx(0.22, abs=0.02)
    for other_index, other_alpha in alpha.items():
        if other_index != index:
            assert other_alpha == pytest.approx(1.0, abs=0.02)


def test_crosshair_outside_every_card_leaves_alpha_at_one(ui):
    ui.mouse = (-500.0, -500.0)   # off the design canvas entirely
    _settle_fade(ui)
    assert all(a == pytest.approx(1.0, abs=0.02) for a in ui.hud_card_alpha.values())


def test_scoped_fades_every_card_to_zero_even_near_a_card(ui):
    index, centre = _first_fadeable_card_centre(ui)
    ui.mouse = centre   # deliberately ALSO near a card
    ui._scope.enter(ui.mouse)
    # Push the zoom spring to a meaningfully-scoped value directly, same
    # trick ScopeState's own tests use — avoids a slow real spring ramp.
    ui._scope.zoom.x = 2.0
    assert ui.scope_visually_active is True
    _settle_fade(ui)
    assert all(a == pytest.approx(0.0, abs=0.02) for a in ui.hud_card_alpha.values())


def test_leaving_the_gun_screen_drops_the_springs(ui):
    from src.ui_album import v3_flow as F

    ui.mouse = (0.0, 0.0)
    ui._tick_hud_fade(1 / 60)
    assert ui._hud_springs
    ui.state.go(F.MENU)
    ui._tick_hud_fade(1 / 60)
    assert ui._hud_springs == {}
    assert ui.hud_card_alpha == {}


# ---- Fix 10's own G12/G5-style guarantee: fire() is untouched -------------

def test_hud_cards_module_never_imports_game_bridge():
    import src.ui_album.hud_cards as module

    assert "game_bridge" not in module.__dict__
    assert not hasattr(module, "fire")


def test_tick_hud_fade_never_touches_game_state():
    """``_tick_hud_fade`` never reaches into ``self.game`` (the live
    match ``fire()`` operates on) at all — this is a pure HUD-drawing
    feature, structurally unable to change a shot's result.
    ``V3Manifest``/``cards_for_screen`` are pure PIL/JSON, no GL needed,
    so it can run for real here."""
    from src.ui_album.v3 import V3Manifest
    from src.ui_album.v3_ui import V3UI
    from src.ui_album.scope import ScopeState
    from src.ui_album import v3_flow as F

    sentinel = object()
    instance = V3UI.__new__(V3UI)
    instance.state = F.V3Flow()
    instance.state.go(F.PLAY)
    instance.game = sentinel
    instance.mouse = (0.0, 0.0)
    instance.manifest = V3Manifest(ROOT)
    instance._scope = ScopeState()
    instance._hud_springs = {}
    instance._hud_springs_screen = None

    instance._tick_hud_fade(1 / 60)
    assert instance.game is sentinel
