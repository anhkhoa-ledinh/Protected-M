"""
TASK_game_feel_fix_1118 Fix 10.2/10.3: HUD card fade (proximity and
scoped). TASK_hud_fix_1119 Fix 2: H hides/shows the whole HUD instead —
session-only, never persisted, and every other function (item slots,
Esc, scope, auto-fire) keeps working while hidden.

- Fix 10.2: a card whose (grown) rect contains the crosshair fades to
  0.22 via Spring(k=420, d=26), unscoped, every gun.
- Fix 10.3: scoped (Precision gun), every card fades to 0 instead.
- Fix 2 (hud_fix_1119): H toggles a session-only flag/Spring (k=600,
  d=40, ~0.15s settle) that fades EVERY card and "free" HUD element
  (item slots, head-tracking pill, etc.) to 0 except the crosshair/bite
  whitelist in ``v3.py``'s ``_card_fade``. A fresh match always starts
  visible.

Pure-Python where possible (Spring convergence, H-toggle semantics);
the proximity/scoped/hide fade tests need a real GL ``V3UI`` (card rects
come from the real ``top.png`` assets) but run with ``HCPD_NO_FX=1`` to
skip the heavy fx preload entirely — this file does not need fx at all,
and test_v3ui_fx_hooks.py already covers that path.
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


def test_hide_spring_settles_near_zero_within_about_0_15s():
    """TASK_hud_fix_1119 Fix 2: the task's own "~0.15s" -- k=600/d=40
    reaches within 2% of target by t=0.15s (and stays there, negligible
    overshoot)."""
    from src.ui_album.v3_ui import _HUD_HIDE_SPRING_D, _HUD_HIDE_SPRING_K

    spring = Spring(1.0, _HUD_HIDE_SPRING_K, _HUD_HIDE_SPRING_D)
    spring.to = 0.0
    elapsed = 0.0
    while elapsed < 0.15:
        spring.step(1 / 60)
        elapsed += 1 / 60
    assert abs(spring.x) <= 0.05


# ---- V3UI fade targets: inside / outside / scoped (real GL, no fx) --------

@pytest.fixture
def ui(gl_context, monkeypatch):
    monkeypatch.setenv("HCPD_NO_FX", "1")
    from src.ui_album.options import AlbumOptions
    from src.account.service import LocalAccountService
    from src.services.prefs_service import LocalPrefsService
    from src.services.shared_album_service import LocalSharedAlbumService
    from src.ui_album.v3_ui import V3UI
    from src.ui_album import v3_flow as F

    options = AlbumOptions(enabled=True, guest=True, ui="v3")
    instance = V3UI(ROOT, ROOT / "assets" / "shaders", options,
                    accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")),
                    shared_album=LocalSharedAlbumService(),
                    prefs=LocalPrefsService(ROOT))
    instance.initialize()
    instance.painter.set_frame((1280, 720))
    instance.state.go(F.PLAY)
    # One tick so _hud_hidden_match_screen already tracks "on PLAY" --
    # matches the real app (tick() runs every frame), and keeps a test
    # that then toggles hidden from being immediately undone by the
    # "fresh match" reset on its first-ever _tick_hud_hidden call.
    instance._tick_hud_hidden(0.0)
    yield instance
    instance.release()


def _settle_fade(ui, seconds: float = 1.0, step: float = 1 / 60) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        ui._tick_hud_fade(step)
        elapsed += step


def _settle_hidden(ui, seconds: float = 1.0, step: float = 1 / 60) -> None:
    elapsed = 0.0
    while elapsed < seconds:
        ui._tick_hud_hidden(step)
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


# ---- Fix 2: H hides/shows the whole HUD (session only) --------------------

def test_toggle_hud_hidden_flips_and_returns_the_new_value(ui):
    assert ui.hud_hidden is False
    assert ui.toggle_hud_hidden() is True
    assert ui.hud_hidden is True
    assert ui.toggle_hud_hidden() is False
    assert ui.hud_hidden is False


def test_toggle_hud_hidden_sets_the_spring_target(ui):
    ui.toggle_hud_hidden()
    assert ui._hud_hide_spring.to == 0.0
    ui.toggle_hud_hidden()
    assert ui._hud_hide_spring.to == 1.0


def test_hud_hidden_alpha_fades_to_zero_then_recovers(ui):
    ui.toggle_hud_hidden()
    _settle_hidden(ui)
    assert ui.hud_hidden_alpha == pytest.approx(0.0, abs=0.02)
    ui.toggle_hud_hidden()
    _settle_hidden(ui)
    assert ui.hud_hidden_alpha == pytest.approx(1.0, abs=0.02)


def test_toggle_hud_hidden_shows_a_toast_only_on_hide(ui):
    ui.state.message = ""
    ui.toggle_hud_hidden()
    assert "HUD hidden" in ui.state.message
    ui.state.message = ""
    ui.toggle_hud_hidden()           # showing again -- no toast
    assert ui.state.message == ""


def test_toggle_hud_hidden_never_writes_prefs(ui, monkeypatch):
    import src.config.prefs as prefs_mod

    def boom(*a, **k):
        raise AssertionError("toggle_hud_hidden must never touch prefs")

    monkeypatch.setattr(prefs_mod, "save", boom)
    ui.toggle_hud_hidden()
    ui.toggle_hud_hidden()


def test_a_fresh_match_resets_hidden_to_visible(ui):
    from src.ui_album import v3_flow as F

    ui.toggle_hud_hidden()
    _settle_hidden(ui, seconds=0.3)
    assert ui.hud_hidden_alpha < 0.5
    # Leaving PLAY for a non-match screen, then landing back on PLAY --
    # exactly what a new match does (TILT -> PLAY) -- must snap visible.
    ui.state.go(F.MENU)
    ui._tick_hud_hidden(1 / 60)
    ui.state.go(F.PLAY)
    ui._tick_hud_hidden(1 / 60)
    assert ui.hud_hidden is False
    assert ui.hud_hidden_alpha == pytest.approx(1.0, abs=0.01)


def test_staying_on_the_same_match_screen_does_not_re_trigger_the_reset(ui):
    """Ticking PLAY every frame of one ongoing match must NOT keep
    snapping hidden back to visible -- only a FRESH entry does."""
    ui.toggle_hud_hidden()
    for _ in range(10):
        ui._tick_hud_hidden(1 / 60)
    assert ui.hud_hidden is True


def test_hidden_does_not_block_esc_back(ui):
    """Esc routes PLAY back to the menu exactly the same whether the
    HUD is hidden or not -- back() never reads hud_hidden at all."""
    from src.ui_album import v3_flow as F

    ui.toggle_hud_hidden()
    action = ui.back()
    assert action == ""
    assert ui.state.screen == F.MENU


def test_hidden_does_not_block_item_slot_selection(ui):
    ui.toggle_hud_hidden()
    ui.state.slot_selected = 2
    assert ui.state.slot_selected == 2


def test_hidden_does_not_block_scope(ui):
    ui.toggle_hud_hidden()
    ui.mouse = (640.0, 360.0)
    ui._scope.enter(ui.mouse)
    ui._scope.zoom.x = 2.0
    assert ui.scope_visually_active is True


# ---- Fix 10's own G12/G5-style guarantee: fire() is untouched -------------

def test_hud_cards_module_never_imports_game_bridge():
    import src.ui_album.hud_cards as module

    assert "game_bridge" not in module.__dict__
    assert not hasattr(module, "fire")


def test_toggle_and_tick_hud_fade_never_touch_game_state():
    """Neither the H-toggle nor the fade spring steppers reach into
    ``self.game`` (the live match ``fire()`` operates on) at all — this
    is a pure HUD-drawing feature, structurally unable to change a
    shot's result. ``V3Manifest``/``cards_for_screen`` are pure
    PIL/JSON, no GL needed, so both tick functions can run for real
    here."""
    from src.ui_album.v3 import V3Manifest
    from src.ui_album.v3_ui import V3UI
    from src.fx.anim import Spring as _Spring
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
    instance._hud_hidden = False
    instance._hud_hide_spring = _Spring(1.0, 600.0, 40.0)
    instance._hud_hidden_match_screen = None

    instance.toggle_hud_hidden()
    instance._tick_hud_fade(1 / 60)
    instance._tick_hud_hidden(1 / 60)
    assert instance.game is sentinel


# ---- item slots are a fadeable card like the others -----------------------

def _slot_card(ui):
    from src.ui_album import v3_flow as F
    from src.ui_album.hud_cards import cards_for_screen, item_slot_card_index

    cards = cards_for_screen(ui.manifest, F.PLAY)
    index = item_slot_card_index(cards)
    assert index is not None and cards[index].fades and not cards[index].art
    x, y, w, h = cards[index].rect
    return index, (x + w * 0.5, y + h * 0.5)


def test_crosshair_on_the_item_slots_fades_them_like_any_card(ui):
    slot_index, centre = _slot_card(ui)
    health_index, _ = _first_fadeable_card_centre(ui)
    ui.mouse = centre
    _settle_fade(ui)
    near = ui.hud_card_alpha[slot_index]
    assert near == pytest.approx(0.22, abs=0.02)
    # Same alpha as a normal card under the crosshair.
    ui.mouse = _first_fadeable_card_centre(ui)[1]
    _settle_fade(ui)
    assert ui.hud_card_alpha[health_index] == pytest.approx(near, abs=0.02)
    assert ui.hud_card_alpha[slot_index] == pytest.approx(1.0, abs=0.02)


def test_scope_fades_the_item_slots_to_zero(ui):
    slot_index, _ = _slot_card(ui)
    ui.mouse = (-500.0, -500.0)
    ui._scope.enter(ui.mouse)
    ui._scope.zoom.x = 2.0
    _settle_fade(ui)
    assert ui.hud_card_alpha[slot_index] == pytest.approx(0.0, abs=0.02)


def test_slot_art_and_text_are_tagged_to_fade_with_the_slot_card(ui):
    from src.ui_album import v3_flow as F
    from src.ui_album.v3 import V3State
    from src.ui_album.v3_values import _hud_items

    ui.state.items = list(F.ITEM_WEIGHT)[:2]
    state = V3State()
    _hud_items(state, ui.state, None, ui.manifest, F.PLAY)
    assert state.slot_extra
    on_slots = [lbl for lbl in state.borrowed if lbl.split("#")[0] in
                ("islot", "ikey", "islot_icon", "iused")]
    assert on_slots and set(on_slots) <= state.slot_labels
    # Slot texts no longer ride in ``extra`` (which does not fade).
    assert not state.extra
