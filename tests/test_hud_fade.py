"""
TASK_game_feel_fix_1118 Fix 10.2/10.3/10.4: HUD card fade (proximity and
scoped) and the compact-HUD toggle.

- Fix 10.2: a card whose (grown) rect contains the crosshair fades to
  0.22 via Spring(k=420, d=26), unscoped, every gun.
- Fix 10.3: scoped (Precision gun), every card fades to 0 instead.
- Fix 10.4: H toggles ``hud_compact``, persisted in prefs; the compact
  edge pills show the same already-computed display strings as the
  normal cards.

Pure-Python where possible (Spring convergence, compact-pill text,
H-toggle persistence); the proximity/scoped fade tests need a real GL
``V3UI`` (card rects come from the real ``top.png`` assets) but run with
``HCPD_NO_FX=1`` to skip the heavy fx preload entirely — this file does
not need fx at all, and test_v3ui_fx_hooks.py already covers that path.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from src.config import prefs
from src.fx.anim import Spring
from src.ui_album.v3 import V3State, _draw_compact_hud

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


# ---- H toggle: flip + persist ----------------------------------------------

def test_toggle_hud_compact_flips_and_returns_the_new_value():
    from src.ui_album.v3_ui import V3UI
    from src.ui_album import v3_flow as F

    instance = V3UI.__new__(V3UI)
    instance.state = F.V3Flow()
    assert instance.state.hud_compact is False
    assert instance.toggle_hud_compact() is True
    assert instance.state.hud_compact is True
    assert instance.toggle_hud_compact() is False
    assert instance.state.hud_compact is False


class _FakeWindowForPrefs:
    def __init__(self, display: str = "window", rect=None) -> None:
        self.display_mode = display
        self._rect = rect or [10, 20, 800, 600]

    def remembered_rect(self):
        return self._rect


def test_apply_hud_compact_persists_without_touching_display_or_rect(tmp_path):
    from src.engine.app import HcpdApplication

    prefs.save(tmp_path, "window", [10, 20, 800, 600])
    app = HcpdApplication.__new__(HcpdApplication)
    app._root = tmp_path
    app._window = _FakeWindowForPrefs("window", [10, 20, 800, 600])

    app._apply_hud_compact(True)

    loaded = prefs.load(tmp_path)
    assert loaded["hud_compact"] is True
    assert loaded["display"] == "window"
    assert loaded["window_rect"] == [10, 20, 800, 600]


def test_hud_compact_loads_from_prefs_at_construction(gl_context, monkeypatch):
    """``V3UI.__init__`` reads ``hud_compact`` the same way it already
    reads ``screen_shake``/``fx`` — patch the loader so this doesn't need
    a real prefs file on disk."""
    monkeypatch.setenv("HCPD_NO_FX", "1")
    monkeypatch.setattr(prefs, "hud_compact_enabled", lambda root: True)
    from src.ui_album.options import AlbumOptions
    from src.ui_album.v3_ui import V3UI

    options = AlbumOptions(enabled=True, guest=True, ui="v3")
    instance = V3UI(ROOT, ROOT / "assets" / "shaders", options)
    assert instance.state.hud_compact is True
    instance.initialize()
    instance.release()


# ---- compact-pill text for given values ------------------------------------

class _FakeCompactPainter:
    def __init__(self) -> None:
        self.pills: list = []

    def pill(self, record, text, style, alpha: float = 1.0) -> None:
        self.pills.append((record, text, style, alpha))

    def text_width(self, runs, texts) -> float:
        return 100.0


def test_compact_hud_pill_text_for_given_values():
    state = V3State()
    state.texts = {
        "level_tab": "LEVEL 3 · Layer 2/4",
        "layer": "layer 2/4",
        "timer": "02:20",
        "kills": ["1", "/4"],
        "intact_value": ["84", "%"],
        "hp_value": "100",
        "st_value": "50",
        "ammo": ["9", "/12"],
        "weapon_name": "Precision gun",
    }
    painter = _FakeCompactPainter()

    _draw_compact_hud(painter, state)

    texts = [call[1] for call in painter.pills]
    assert "Lv 3 · Layer 2/4" in texts
    assert "02:20 · 1/4 · 84%" in texts
    # TASK_perf_textures_1119 Fix 7: "♥" isn't in the pack's Nunito cmap
    # (FreeType substitutes its .notdef glyph) -- "HP" is plain ASCII.
    assert "HP 100 · 50" in texts
    assert "Precision gun 9/12" in texts
    assert "Pause" in texts


def test_compact_hud_pvp_hunt_uses_kills_you_key():
    state = V3State()
    state.texts = {"layer": "layer 1/4", "timer": "01:00",
                   "kills_you": ["0", "/4"]}
    painter = _FakeCompactPainter()

    _draw_compact_hud(painter, state)

    texts = [call[1] for call in painter.pills]
    assert any("0/4" in t for t in texts)


def test_compact_hud_skips_empty_pills():
    """No ammo/weapon and no hp text -> bottom-left/right pills just
    don't get drawn, instead of drawing an empty/garbage pill. The Pause
    pill (TASK_perf_textures_1119 Fix 7) is unconditional -- it never
    reads ``state.texts`` at all, so it's the one pill that always
    appears alongside whatever else is non-empty."""
    state = V3State()
    state.texts = {"timer": "00:30"}
    painter = _FakeCompactPainter()

    _draw_compact_hud(painter, state)

    texts = [call[1] for call in painter.pills]
    assert texts == ["00:30", "Pause"]


# ---- Fix 10's own G12/G5-style guarantee: fire() is untouched -------------

def test_hud_cards_module_never_imports_game_bridge():
    import src.ui_album.hud_cards as module

    assert "game_bridge" not in module.__dict__
    assert not hasattr(module, "fire")


def test_toggle_and_tick_hud_fade_never_touch_game_state():
    """Neither the H-toggle nor the fade spring stepper reaches into
    ``self.game`` (the live match ``fire()`` operates on) at all — Fix 10
    is a pure HUD-drawing feature, structurally unable to change a shot's
    result. ``V3Manifest``/``cards_for_screen`` are pure PIL/JSON, no GL
    needed, so ``_tick_hud_fade`` can run for real here."""
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

    instance.toggle_hud_compact()
    instance._tick_hud_fade(1 / 60)
    assert instance.game is sentinel
