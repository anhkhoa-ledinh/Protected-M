"""
TASK_ui_sakuga_1117 Phase 7: end-to-end ``V3UI`` fx hooks — click dispatch
(ui_confirm/ui_tap/sticker star), shot hooks (hit_kill/hit_crit/hit_block),
the per-tick state-transition triggers (loading bar, tilt checklist,
toast, WIN/PVP_RESULT), and the ``HCPD_NO_FX`` off switch. All need a
real GL context (texture upload) — see ``tests/conftest.py``.
"""

import time
from pathlib import Path

from src.ui_album import v3_flow as F
from src.ui_album.options import AlbumOptions
from src.ui_album.v3_ui import V3UI

ROOT = Path(".").resolve()


def _wait_resident(ui, name, timeout=5.0, **kwargs):
    deadline = time.perf_counter() + timeout
    while time.perf_counter() < deadline:
        ui.draw((1280, 720))
        if ui.spawn_fx(name, 0.0, 0.0, **kwargs):
            return True
        time.sleep(0.01)
    return False


def _make_ui(gl_context, **option_kwargs):
    options = AlbumOptions(enabled=True, guest=True, ui="v3", **option_kwargs)
    ui = V3UI(ROOT, ROOT / "assets" / "shaders", options)
    ui.initialize()
    ui.painter.set_frame((1280, 720))
    return ui


# ---- click dispatch ----------------------------------------------------
def test_primary_button_spawns_ui_confirm(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.SIGN_IN)
    ui.state.player = "khoa"
    ui.state.password = "x"
    assert _wait_resident(ui, "ui_confirm")
    before = len(ui._fx_player._effects)
    ui.activate("btn_Sign in")
    assert len(ui._fx_player._effects) == before + 1


def test_tab_click_spawns_ui_tap_and_navigates(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.MENU)
    assert _wait_resident(ui, "ui_tap")
    before = len(ui._fx_player._effects)
    ui.activate("btn_Settings")
    assert len(ui._fx_player._effects) == before + 1
    assert ui.state.screen == F.SETTINGS


def test_field_click_spawns_nothing(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.SIGN_IN)
    ui.draw((1280, 720))
    before = len(ui._fx_player._effects)
    ui.activate("field_user")
    assert len(ui._fx_player._effects) == before


def test_sticker_placement_spawns_tap_and_star(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.LOADOUT)
    ui.state.weapon = 0
    assert _wait_resident(ui, "star_pop", words=False)
    assert _wait_resident(ui, "ui_tap")
    before = len(ui._fx_player._effects)
    ui.activate("sticker_Flare")
    after = len(ui._fx_player._effects)
    assert after == before + 2, [e.name for e in ui._fx_player._effects]
    assert "Flare" in ui.state.items


def test_dialog_only_its_own_buttons_spawn_fx(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PHOTOS)
    ui.state.shared_album_pending = "some/photo.png"
    assert _wait_resident(ui, "ui_tap")
    before = len(ui._fx_player._effects)
    ui.activate("btn_confirm_yes")
    assert len(ui._fx_player._effects) == before + 1


# ---- shot hooks ----------------------------------------------------
def test_spawn_hit_fx_kill_and_star(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PLAY)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_kill")
    assert _wait_resident(ui, "star_pop", words=False)
    before = len(ui._fx_player._effects)
    ui.spawn_hit_fx(crit=False)
    # hit_kill at crosshair + wordless star_pop over the kills counter.
    assert len(ui._fx_player._effects) == before + 2


def test_spawn_hit_fx_crit(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PLAY)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_crit")
    before = len(ui._fx_player._effects)
    ui.spawn_hit_fx(crit=True)
    assert any(e.name == "hit_crit" for e in ui._fx_player._effects[before:])


def test_spawn_miss_fx(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PLAY)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_block")
    before = len(ui._fx_player._effects)
    ui.spawn_miss_fx()
    assert len(ui._fx_player._effects) == before + 1


def test_spawn_hit_fx_on_pvp_hide_has_no_kill_counter(gl_context):
    # PVP_HIDE has no kills text (the hider doesn't shoot) — only the
    # crosshair effect spawns, no star_pop.
    ui = _make_ui(gl_context)
    ui.state.go(F.PVP_HIDE)
    ui.mouse = (640.0, 360.0)
    assert _wait_resident(ui, "hit_kill")
    before = len(ui._fx_player._effects)
    ui.spawn_hit_fx(crit=False)
    assert len(ui._fx_player._effects) == before + 1


# ---- per-tick state-transition triggers --------------------------------
def test_loading_bar_triggers_edge_and_gogo(gl_context):
    ui = _make_ui(gl_context)
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


def test_loading_done_spawns_load_done_once(gl_context):
    ui = _make_ui(gl_context)
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


def test_tilt_checklist_transition_spawns_star(gl_context):
    ui = _make_ui(gl_context)
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


def test_win_entry_spawns_star_five_once(gl_context):
    ui = _make_ui(gl_context)
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


def test_pvp_result_win_spawns_star_five(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PVP_MODE)
    ui.state.last_result = {"winner": "you", "score": 10, "opponent_score": 5}
    assert _wait_resident(ui, "star_five")
    ui.state.go(F.PVP_RESULT)
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before + 1


def test_pvp_result_loss_spawns_nothing(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PVP_MODE)
    ui.state.last_result = {"winner": "opponent", "score": 2, "opponent_score": 9}
    ui.draw((1280, 720))
    ui.state.go(F.PVP_RESULT)
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before
    assert ui._fx_pvp_result_spawned is True


def test_toast_spawns_once_per_new_message(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.MENU)
    assert _wait_resident(ui, "toast_in")
    before = len(ui._fx_player._effects)
    F.say(ui.state, "Hello there")
    ui.draw((1280, 720))
    after = len(ui._fx_player._effects)
    assert after == before + 1
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == after  # same message: no re-spawn


def test_no_toast_on_match_screens(gl_context):
    ui = _make_ui(gl_context)
    ui.state.go(F.PLAY)
    F.say(ui.state, "Hello there")
    before = len(ui._fx_player._effects)
    ui.draw((1280, 720))
    assert len(ui._fx_player._effects) == before


# ---- HCPD_NO_FX off switch ----------------------------------------------
def test_hcpd_no_fx_disables_everything(gl_context, monkeypatch):
    monkeypatch.setenv("HCPD_NO_FX", "1")
    ui = _make_ui(gl_context)
    assert ui.fx_enabled is False
    assert ui._fx_library is None
    assert ui._fx_player is None
    assert ui.spawn_fx("ui_tap", 0.0, 0.0) is False
    assert ui.spawn_hit_fx(False) is None
    ui.state.go(F.MENU)
    ui.mouse = (-1000.0, -1000.0)
    ui.draw((1280, 720))  # must not crash with fx off
