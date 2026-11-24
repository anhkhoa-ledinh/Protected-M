"""
End-to-end ``V3UI`` fx hooks — click dispatch (ui_confirm/ui_tap/sticker
star), shot hooks (hit_kill/hit_crit/hit_block), the per-tick
state-transition triggers (loading bar, tilt checklist, toast,
WIN/PVP_RESULT), and the ``HCPD_NO_FX`` off switch. All need a real GL
context (texture upload).

TASK_game_feel_1118 P3 made fx construction much heavier (preload decodes
+ bleeds + bands-uploads EVERY sheet/word at launch, not one lazily per
screen) — building a FRESH ``V3UI`` per test, 19 times over in this file
alone, repeated that whole cost 19x and was observed to exhaust process
memory/threads on this machine. So every test but the NO_FX one (which
needs its OWN instance, built with the env var already set) shares ONE
module-scoped ``V3UI``, built with its OWN module-scoped GL window
(``tests/conftest.py``'s ``gl_context`` is function-scoped — a shared
fixture many other files rely on being function-scoped, not something to
change for this file alone); each test resets the shared instance's
per-test mutable state first via the ``ui`` fixture below.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import glfw
import pytest

from src.ui_album import v3_flow as F
from src.ui_album.options import AlbumOptions
from src.account.service import LocalAccountService
from src.services.prefs_service import LocalPrefsService
from src.services.shared_album_service import LocalSharedAlbumService
from src.ui_album.v3_ui import V3UI

ROOT = Path(".").resolve()


def _wait_resident(ui, name, timeout=90.0, **kwargs):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        ui.draw((1280, 720))
        if ui.spawn_fx(name, 0.0, 0.0, **kwargs):
            return True
        time.sleep(0.01)
    return False


def _reset_ui_state(ui) -> None:
    """Put the SHARED ``ui`` fixture's per-test mutable state back to
    fresh-construction defaults — mirrors the relevant slice of
    ``V3UI.__init__`` — without rebuilding the expensive parts (painter,
    fx_library's already-decoded/resident textures)."""
    ui.state = F.V3Flow()
    ui.game = None
    ui.mouse = (0.0, 0.0)
    ui.mouse_down = False
    ui.pending = ""
    ui._hover_last_t = None
    ui._fx_screen = None
    ui._fx_last_t = None
    ui._fx_trigger_last_t = None
    ui._fx_trigger_screen = None
    ui._fx_load_edge_elapsed = 0.0
    ui._fx_gogo_spawned = False
    ui._fx_load_done_spawned = False
    ui._fx_last_step_done = 0
    ui._fx_last_tilt_checks = None
    ui._fx_last_message_until = 0.0
    ui._fx_win_entered_t = None
    ui._fx_win_star_spawned = False
    ui._fx_rank_up_spawned = False
    ui._fx_pvp_result_spawned = False
    ui.spawn_fx = type(ui).spawn_fx.__get__(ui, type(ui))   # undo any monkeypatch
    if ui._fx_player is not None:
        ui._fx_player.clear(lambda name: True)


@pytest.fixture(scope="module")
def _module_gl_window():
    """Same hidden-window recipe as ``tests/conftest.py``'s ``gl_context``,
    but MODULE-scoped — see the file docstring for why this file needs
    its own instead of the shared (function-scoped) fixture."""
    if os.environ.get("PM_FORCE_NO_GL") == "1":
        pytest.skip("PM_FORCE_NO_GL=1 (forced)")
    if not glfw.init():
        pytest.skip("glfw.init() failed — no display/driver available")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
    glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
    glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    window = glfw.create_window(64, 64, "pytest-fx-hooks", None, None)
    if window is None:
        glfw.terminate()
        pytest.skip("Could not create an OpenGL 3.3 core-profile context")
    glfw.make_context_current(window)
    try:
        from OpenGL import GL
        if not GL.glGetString(GL.GL_VERSION):
            raise RuntimeError("no GL_VERSION")
    except Exception:
        glfw.destroy_window(window)
        glfw.terminate()
        pytest.skip("No real OpenGL driver on this runner")
    try:
        yield window
    finally:
        glfw.destroy_window(window)
        glfw.terminate()


@pytest.fixture(scope="module")
def _shared_ui(_module_gl_window):
    options = AlbumOptions(enabled=True, guest=True, ui="v3")
    shared = V3UI(ROOT, ROOT / "assets" / "shaders", options,
                  accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")),
                  shared_album=LocalSharedAlbumService(),
                  prefs=LocalPrefsService(ROOT))
    shared.initialize()
    shared.painter.set_frame((1280, 720))
    # Pay the "wait for preload" cost ONCE for the whole module.
    deadline = time.perf_counter() + 90.0
    while (time.perf_counter() < deadline
          and shared._fx_library is not None
          and shared._fx_library.preload_remaining > 0):
        shared.draw((1280, 720))
        time.sleep(0.005)
    yield shared
    shared.release()


@pytest.fixture
def ui(_shared_ui):
    _reset_ui_state(_shared_ui)
    return _shared_ui


# ---- click dispatch ----------------------------------------------------
def test_primary_button_spawns_ui_confirm(ui):
    ui.state.go(F.SIGN_IN)
    ui.state.player = "khoa"
    ui.state.password = "x"
    assert _wait_resident(ui, "ui_confirm")
    before = len(ui._fx_player._effects)
    ui.activate("btn_Sign in")
    assert len(ui._fx_player._effects) == before + 1


def test_tab_click_spawns_ui_tap_and_navigates(ui):
    ui.state.go(F.MENU)
    assert _wait_resident(ui, "ui_tap")
    before = len(ui._fx_player._effects)
    ui.activate("btn_Settings")
    assert len(ui._fx_player._effects) == before + 1
    assert ui.state.screen == F.SETTINGS


def test_field_click_spawns_nothing(ui):
    ui.state.go(F.SIGN_IN)
    ui.draw((1280, 720))
    before = len(ui._fx_player._effects)
    ui.activate("field_user")
    assert len(ui._fx_player._effects) == before


def test_sticker_placement_spawns_tap_and_star(ui):
    ui.state.go(F.LOADOUT)
    ui.state.weapon = 0
    assert _wait_resident(ui, "star_pop", words=False)
    assert _wait_resident(ui, "ui_tap")
    before = len(ui._fx_player._effects)
    ui.activate("sticker_Flare")
    after = len(ui._fx_player._effects)
    assert after == before + 2, [e.name for e in ui._fx_player._effects]
    assert "Flare" in ui.state.items


def test_dialog_only_its_own_buttons_spawn_fx(ui):
    ui.state.go(F.PHOTOS)
    ui.state.shared_album_pending = "some/photo.png"
    assert _wait_resident(ui, "ui_tap")
    before = len(ui._fx_player._effects)
    ui.activate("btn_confirm_yes")
    assert len(ui._fx_player._effects) == before + 1


# ---- shot hooks ----------------------------------------------------
def test_spawn_hit_fx_kill_and_star(ui):
    ui.state.go(F.PLAY)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_kill")
    assert _wait_resident(ui, "star_pop", words=False)
    before = len(ui._fx_player._effects)
    ui.spawn_hit_fx(crit=False)
    # hit_kill at crosshair + wordless star_pop over the kills counter.
    assert len(ui._fx_player._effects) == before + 2


def test_spawn_hit_fx_crit(ui):
    ui.state.go(F.PLAY)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_crit")
    before = len(ui._fx_player._effects)
    ui.spawn_hit_fx(crit=True)
    assert any(e.name == "hit_crit" for e in ui._fx_player._effects[before:])


def test_spawn_miss_fx(ui):
    ui.state.go(F.PLAY)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_block")
    before = len(ui._fx_player._effects)
    ui.spawn_miss_fx()
    assert len(ui._fx_player._effects) == before + 1


def test_spawn_hit_fx_on_pvp_hide_has_no_kill_counter(ui):
    # PVP_HIDE has no kills text (the hider doesn't shoot) — only the
    # crosshair effect spawns, no star_pop.
    ui.state.go(F.PVP_HIDE)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_kill")
    before = len(ui._fx_player._effects)
    ui.spawn_hit_fx(crit=False)
    assert len(ui._fx_player._effects) == before + 1


# ---- per-tick state-transition triggers --------------------------------
def test_loading_bar_triggers_edge_and_gogo(ui):
    ui.state.go(F.LOADING)
    ui.state.pipeline_pct = 50.0
    ui.state.layer_count = 4
    ui.draw((1280, 720))
    assert _wait_resident(ui, "load_edge")
    assert _wait_resident(ui, "star_pop", words=False)
    before = len(ui._fx_player._effects)
    ui._fx_load_edge_elapsed = 999.0
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) > before
    assert ui._fx_gogo_spawned is True


def test_loading_done_spawns_load_done_once(ui):
    ui.state.go(F.LOADING)
    ui.state.pipeline_pct = 100.0
    assert _wait_resident(ui, "load_done")
    ui.draw((1280, 720))  # the internal trigger checks residency BEFORE
                          # that frame's own pump(), one frame behind a
                          # manual probe like _wait_resident's — give it
                          # one more frame to catch up.
    assert ui._fx_load_done_spawned is True
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    ui.draw((1280, 720))
    # One-shot: no new load_done after the first successful spawn.
    assert not any(
        e.name == "load_done" for e in ui._fx_player._effects[before:])


def test_tilt_checklist_transition_spawns_star(ui):
    ui.state.go(F.TILT)
    ui.draw((1280, 720))
    assert ui._fx_last_tilt_checks == (False, False, False)
    assert _wait_resident(ui, "star_pop", words=False)
    ui._fx_last_tilt_checks = (False, False, False)
    spawned = []
    ui.spawn_fx = lambda name, x, y, **kw: (spawned.append((name, x, y)) or True)
    ui.state.tilt_checks = lambda: (True, False, False)
    ui._step_tilt_fx()
    assert spawned == [("star_pop", 783.0, 497.0)]


def test_win_entry_spawns_star_five_once(ui):
    ui.state.go(F.LEVELS)
    assert _wait_resident(ui, "star_five")
    ui.state.go(F.WIN)
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before + 1
    assert ui._fx_win_star_spawned is True
    before2 = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before2  # one-shot


def test_pvp_result_win_spawns_star_five(ui):
    ui.state.go(F.PVP_MODE)
    ui.state.last_result = {"winner": "you", "score": 10, "opponent_score": 5}
    assert _wait_resident(ui, "star_five")
    ui.state.go(F.PVP_RESULT)
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before + 1


def test_pvp_result_loss_spawns_nothing(ui):
    ui.state.go(F.PVP_MODE)
    ui.state.last_result = {"winner": "opponent", "score": 2, "opponent_score": 9}
    ui.draw((1280, 720))
    ui.state.go(F.PVP_RESULT)
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before
    assert ui._fx_pvp_result_spawned is True


def test_toast_spawns_once_per_new_message(ui):
    ui.state.go(F.MENU)
    assert _wait_resident(ui, "toast_in")
    before = len(ui._fx_player._effects)
    F.say(ui.state, "Hello there")
    ui.draw((1280, 720))
    after = len(ui._fx_player._effects)
    assert after == before + 1
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == after  # same message: no re-spawn


def test_no_toast_on_match_screens(ui):
    ui.state.go(F.PLAY)
    F.say(ui.state, "Hello there")
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before


# ---- HCPD_NO_FX off switch ----------------------------------------------
def test_hcpd_no_fx_disables_everything(_module_gl_window, monkeypatch):
    # Needs its OWN V3UI instance (fx-disabled is decided in __init__, so
    # it can't share ``_shared_ui``, already built with fx enabled) but
    # reuses the SAME module-scoped window/context as everything else in
    # this file — mixing it with the separate function-scoped
    # ``gl_context`` fixture let the two GLFW lifecycles interleave and
    # corrupt each other (``glfw.terminate()`` from one firing while the
    # other still expected a live context).
    monkeypatch.setenv("HCPD_NO_FX", "1")
    options = AlbumOptions(enabled=True, guest=True, ui="v3")
    solo_ui = V3UI(ROOT, ROOT / "assets" / "shaders", options,
                   accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")),
                   shared_album=LocalSharedAlbumService(),
                   prefs=LocalPrefsService(ROOT))
    solo_ui.initialize()
    solo_ui.painter.set_frame((1280, 720))
    assert solo_ui.fx_enabled is False
    assert solo_ui._fx_library is None
    assert solo_ui._fx_player is None
    assert solo_ui.spawn_fx("ui_tap", 0.0, 0.0) is False
    assert solo_ui.spawn_hit_fx(False) is None
    solo_ui.state.go(F.MENU)
    solo_ui.mouse = (-1000.0, -1000.0)
    solo_ui.draw((1280, 720))  # must not crash with fx off
    solo_ui.release()
