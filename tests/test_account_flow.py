"""
TASK_accounts_1119 Phase 2: sign-up/sign-in wired through ``v3_routes``/
``v3_values``/``v3_ui.V3UI`` — live validation feedback, the cooldown,
the worker-thread hashing, and the actual click/submit flow end to end.

``on_click``/``_signup_problem`` are pure (``V3Flow`` + a plain
``accounts`` dict — no GL). The end-to-end async flow needs a real
``V3Painter`` (its constructor compiles a shader — ``gl_context``), but
never touches ``_fx_library`` (``HCPD_NO_FX`` forced here) and never
calls ``initialize()`` — nothing exercised below reads a texture.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.shared import account_rules
from src.account import store as account_store
from src.ui_album import v3_flow as F
from src.ui_album import v3_routes
from src.ui_album import v3_values
from src.ui_album.v3 import V3Manifest

ROOT = Path(".").resolve()


# ---- v3_routes.on_click: pure, no GL ---------------------------------------

def _flow(screen: str) -> F.V3Flow:
    flow = F.V3Flow()
    flow.go(screen)
    return flow


def test_sign_in_click_with_empty_fields_shows_local_error_no_action():
    flow = _flow(F.SIGN_IN)
    action = v3_routes.on_click(flow, "btn_Sign in", 0.0)
    assert action == ""
    assert flow.screen == F.SIGN_IN_ERR
    assert "username and password" in flow.sign_in_error


def test_sign_in_click_with_fields_filled_returns_the_async_action():
    flow = _flow(F.SIGN_IN)
    flow.player, flow.password = "anhkhoa", "password123"
    action = v3_routes.on_click(flow, "btn_Sign in", 0.0)
    assert action == v3_routes.ACT_SIGN_IN
    # Still ON the sign-in screen — the actual verdict comes back async.
    assert flow.screen == F.SIGN_IN


def test_sign_up_click_with_bad_name_touches_all_fields_no_action():
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password, flow.password2 = "ab", "password123", "password123"
    action = v3_routes.on_click(flow, "btn_Create album", 0.0)
    assert action == ""
    assert flow.touched == {"field_user", "field_pass", "field_pass2"}
    assert flow.screen == F.SIGN_UP


def test_sign_up_click_with_short_password_no_action():
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password, flow.password2 = "gooduser", "short1", "short1"
    action = v3_routes.on_click(flow, "btn_Create album", 0.0)
    assert action == ""


def test_sign_up_click_with_mismatch_no_action():
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password, flow.password2 = "gooduser", "password123", "password124"
    action = v3_routes.on_click(flow, "btn_Create album", 0.0)
    assert action == ""


def test_sign_up_click_with_everything_valid_returns_the_async_action():
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password, flow.password2 = "gooduser", "password123", "password123"
    action = v3_routes.on_click(flow, "btn_Create album", 0.0)
    assert action == v3_routes.ACT_SIGN_UP


def test_guest_links_set_the_guest_account_key():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_guest", 0.0)
    assert flow.account_key == account_rules.GUEST_KEY
    assert flow.screen == F.MENU

    flow2 = _flow(F.SIGN_IN_ERR)
    v3_routes.on_click(flow2, "footer_link", 0.0)
    assert flow2.account_key == account_rules.GUEST_KEY


def test_field_focus_still_works():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "field_user", 0.0)
    assert flow.focus == "field_user"


# ---- field caps and the "touched" set (pure V3Flow, no GL) -----------------

def test_password_field_cap_is_64_not_24():
    flow = F.V3Flow()
    flow.focus = "field_pass"
    for _ in range(70):
        F.type_char(flow, "x")
    assert len(flow.password) == 64


def test_username_field_marks_touched_on_first_char():
    flow = F.V3Flow()
    flow.focus = "field_user"
    assert "field_user" not in flow.touched
    F.type_char(flow, "a")
    assert "field_user" in flow.touched


def test_backspace_also_marks_touched():
    flow = F.V3Flow()
    flow.focus = "field_pass"
    F.backspace(flow)
    assert "field_pass" in flow.touched


def test_touched_resets_on_a_fresh_entry_to_sign_up():
    flow = F.V3Flow()
    flow.go(F.SIGN_UP)
    flow.focus = "field_user"
    F.type_char(flow, "a")
    assert flow.touched == {"field_user"}
    flow.go(F.MENU)
    flow.go(F.SIGN_UP)
    assert flow.touched == set()


def test_touched_survives_bouncing_within_sign_up():
    """Only a FRESH arrival (from a different screen) resets touched —
    ``go(SIGN_UP)`` while already on SIGN_UP (a no-op screen-wise, but a
    caller might still invoke it) must not wipe progress."""
    flow = F.V3Flow()
    flow.go(F.SIGN_UP)
    flow.focus = "field_user"
    F.type_char(flow, "a")
    flow.go(F.SIGN_UP)
    assert flow.touched == {"field_user"}


# ---- live sign-up feedback: valid_user / hint / hint_bad (pure) ------------

@pytest.fixture
def manifest():
    return V3Manifest(ROOT)


@pytest.fixture
def painter_stub():
    """``state_for`` needs a ``painter`` argument but ``_sign_in`` never
    reads it (only ``_loadout`` and friends do) — a stub is enough and
    avoids a GL context for these pure-logic tests."""
    return object()


def _signup_state(manifest, painter_stub, flow, accounts=None):
    return v3_values.state_for(manifest, painter_stub, flow, accounts=accounts)


def test_valid_user_only_when_valid_and_free(manifest, painter_stub):
    accounts = {"accounts": {"taken": {}}}
    flow = _flow(F.SIGN_UP)
    flow.player = "Taken"
    state = _signup_state(manifest, painter_stub, flow, accounts)
    assert state.sprites.get("valid_user") is None   # case-insensitively taken

    flow.player = "FreeName"
    state2 = _signup_state(manifest, painter_stub, flow, accounts)
    assert state2.sprites.get("valid_user") is True


def test_hint_shows_taken_name_problem(manifest, painter_stub):
    accounts = {"accounts": {"taken": {}}}
    flow = _flow(F.SIGN_UP)
    flow.player = "Taken"
    flow.touched = {"field_user"}
    state = _signup_state(manifest, painter_stub, flow, accounts)
    assert state.sprites.get("hint_bad") is True
    assert state.texts.get("hint") == "Username already taken"


def test_hint_shows_short_password_problem(manifest, painter_stub):
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password = "gooduser", "short1"
    flow.touched = {"field_user", "field_pass"}
    state = _signup_state(manifest, painter_stub, flow, {"accounts": {}})
    assert state.texts.get("hint") == "8-64 characters"


def test_hint_shows_mismatch_problem(manifest, painter_stub):
    flow = _flow(F.SIGN_UP)
    flow.player = "gooduser"
    flow.password, flow.password2 = "password123", "password124"
    flow.touched = {"field_user", "field_pass", "field_pass2"}
    state = _signup_state(manifest, painter_stub, flow, {"accounts": {}})
    assert state.texts.get("hint") == "Passwords don't match"


def test_hint_ok_when_nothing_touched_has_a_problem(manifest, painter_stub):
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password, flow.password2 = "gooduser", "password123", "password123"
    flow.touched = {"field_user", "field_pass", "field_pass2"}
    state = _signup_state(manifest, painter_stub, flow, {"accounts": {}})
    assert state.sprites.get("hint_ok") is True
    assert state.sprites.get("hint_bad") is None


def test_hint_text_key_is_explicitly_hidden_when_ok(manifest, painter_stub):
    """Regression: ``_draw_texts`` falls back to the manifest's own
    SAMPLE text (``v3_01b_sign_up``'s "hint" run is literally "Username
    already taken", so every screen has placeholder content for
    --ui-shots) whenever a key is ABSENT from ``state.texts`` — found by
    rendering v3_01b with a real GL context and seeing the green
    hint_ok sprite overlap with red leftover sample text. The "ok"
    branch must explicitly hide "hint", not just skip setting it."""
    flow = _flow(F.SIGN_UP)
    flow.player, flow.password, flow.password2 = "gooduser", "password123", "password123"
    flow.touched = {"field_user", "field_pass", "field_pass2"}
    state = _signup_state(manifest, painter_stub, flow, {"accounts": {}})
    assert "hint" in state.hidden
    assert state.texts.get("hint") is None


def test_hint_hidden_until_the_field_is_touched(manifest, painter_stub):
    """A field with a problem that has NOT been typed into yet must not
    show a hint — the task's own "shown only once the user has typed
    into that field"."""
    flow = _flow(F.SIGN_UP)
    flow.player = "ab"   # too short — but never touched
    state = _signup_state(manifest, painter_stub, flow, {"accounts": {}})
    assert state.sprites.get("hint_bad") is None
    assert state.sprites.get("hint_ok") is True


def test_sign_in_error_screen_shows_the_dynamic_error_text(manifest, painter_stub):
    flow = _flow(F.SIGN_IN_ERR)
    flow.sign_in_error = "Wrong password. Try again."
    state = _signup_state(manifest, painter_stub, flow)
    assert state.texts.get("error") == "Wrong password. Try again."


# ---- name fitting: every username slot, measured with the real font -------

@pytest.fixture
def real_painter(gl_context, manifest):
    from src.ui_album.v3_painter import V3Painter

    painter = V3Painter(manifest, ROOT / "assets" / "shaders")
    painter.initialize()
    yield painter
    painter.release()


#: (screen, key) for every slot TASK_accounts_1119 names: player_name
#: (menu), the PvP you_name/host_name/opp_name slots. Initials are a
#: single char — never need fitting, not included here.
NAME_SLOTS = [
    ("v3_02a_menu", "player_name"),
    ("v3_11a_pvp_searching", "you_name"),
    ("v3_11b_pvp_invite", "host_name"),
    ("v3_11c_pvp_found", "you_name"),
    ("v3_11c_pvp_found", "opp_name"),
    ("v3_13_pvp_result", "you_name"),
    ("v3_13_pvp_result", "opp_name"),
]


@pytest.mark.parametrize("screen,key", NAME_SLOTS)
def test_sixteen_char_name_fits_every_slot(manifest, real_painter, screen, key):
    name16 = "A" * 16
    fitted = v3_values._fit_name(manifest, real_painter, screen, key, name16)
    record = manifest.text(screen, key)
    width = real_painter.text_width([record["runs"][0]], [fitted])
    assert width <= float(record["rect"][2])


def test_fit_name_truncates_with_ellipsis_when_too_long(manifest, real_painter):
    fitted = v3_values._fit_name(manifest, real_painter, "v3_02a_menu",
                                 "player_name", "A" * 16)
    assert fitted.endswith("…")
    assert fitted != "A" * 16


def test_fit_name_leaves_a_short_name_unchanged(manifest, real_painter):
    fitted = v3_values._fit_name(manifest, real_painter, "v3_02a_menu", "player_name", "ab")
    assert fitted == "ab"


def test_fit_name_handles_a_missing_painter_gracefully(manifest):
    fitted = v3_values._fit_name(manifest, None, "v3_02a_menu", "player_name", "A" * 16)
    assert fitted == "A" * 16   # unfitted, not a crash — the --ui-shots path


def test_network_supplied_opponent_name_is_fitted_in_the_real_flow(manifest, real_painter):
    """The PvP opponent's name arrives over the wire with no length
    promise from this side — the live _pvp_lobby/_pvp_result code paths
    (not just the pure _fit_name helper) must fit it too."""
    flow = _flow(F.PVP_FOUND)
    flow.mode = F.PVP_MODES[0]
    flow.pvp_opponent_name = "B" * 40
    state = v3_values.state_for(manifest, real_painter, flow, accounts={"accounts": {}})
    record = manifest.text(F.PVP_FOUND, "opp_name")
    width = real_painter.text_width([record["runs"][0]], [state.texts["opp_name"]])
    assert width <= float(record["rect"][2])
    assert state.texts["opp_name"] != "B" * 40


# ---- end-to-end async flow through V3UI (real GL, no fx) -------------------

@pytest.fixture
def ui(gl_context, monkeypatch, tmp_path):
    monkeypatch.setenv("HCPD_NO_FX", "1")
    from src.ui_album.options import AlbumOptions
    from src.account.service import LocalAccountService
    from src.services.prefs_service import LocalPrefsService
    from src.services.shared_album_service import LocalSharedAlbumService
    from src.ui_album.v3_ui import V3UI

    options = AlbumOptions(enabled=True, ui="v3")
    instance = V3UI(ROOT, ROOT / "assets" / "shaders", options,
                    accounts=LocalAccountService(tmp_path),
                    shared_album=LocalSharedAlbumService(),
                    prefs=LocalPrefsService(ROOT))
    # The accounts live in an ISOLATED temp dir; manifest/shaders still come
    # from the real repo, but accounts.json must never touch the real one.
    yield instance
    instance.release()


def _wait_until_not_busy(ui, timeout=5.0):
    deadline = time.time() + timeout
    while ui.account_busy and time.time() < deadline:
        ui.tick()
        time.sleep(0.005)
    assert not ui.account_busy, "account job never finished"


def test_sign_up_success_goes_to_menu_creates_the_account(ui):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "NewPlayer1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    assert ui.account_busy
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.MENU
    assert ui.state.account_key == "newplayer1"
    assert "newplayer1" in ui._accounts._data["accounts"]


def test_sign_in_unknown_name(ui):
    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "nosuchuser", "password123"
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.SIGN_IN_ERR
    assert "No album named nosuchuser" in ui.state.sign_in_error


def test_sign_in_wrong_password(ui):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "RealUser1"
    ui.state.password = ui.state.password2 = "correctpass1"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "RealUser1", "wrongpassword"
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.SIGN_IN_ERR
    assert ui.state.sign_in_error == "Wrong password. Try again."


def test_sign_in_success(ui):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "RealUser2"
    ui.state.password = ui.state.password2 = "correctpass1"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "realuser2", "correctpass1"  # case-insensitive
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.MENU
    assert ui.state.account_key == "realuser2"
    assert ui.state.player == "RealUser2"   # exact typed-at-signup form


def test_five_failures_trigger_the_cooldown_then_it_ends(ui):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "ThrottleMe1"
    ui.state.password = ui.state.password2 = "correctpass1"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    for _ in range(account_store.MAX_FAILURES):
        ui.state.go(F.SIGN_IN)
        ui.state.player, ui.state.password = "ThrottleMe1", "wrongpassword"
        ui.sign_in_submit()
        _wait_until_not_busy(ui)
    assert "Too many tries" in ui.state.sign_in_error

    key = account_rules.username_key("ThrottleMe1")
    assert ui._accounts._throttle.cooldown_remaining(key, time.perf_counter()) > 0.0


def test_successful_sign_in_resets_the_failure_counter(ui):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "ResetMe1"
    ui.state.password = ui.state.password2 = "correctpass1"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    for _ in range(account_store.MAX_FAILURES - 1):
        ui.state.go(F.SIGN_IN)
        ui.state.player, ui.state.password = "ResetMe1", "wrongpassword"
        ui.sign_in_submit()
        _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "ResetMe1", "correctpass1"
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.MENU

    key = account_rules.username_key("ResetMe1")
    assert ui._accounts._throttle.cooldown_remaining(key, time.perf_counter()) == 0.0


def test_repeat_clicks_while_busy_start_only_one_job(ui, monkeypatch):
    """``sign_up_submit`` always calls into ``_start_account_job``, but
    the busy guard INSIDE it must refuse to spin up a second worker
    thread while one is already in flight — checked here by counting
    actual ``threading.Thread`` construction, not call count on the
    wrapper (which fires every time regardless of the guard)."""
    import threading

    import src.ui_album.v3_ui as v3_ui_module

    threads_started = []
    real_thread = threading.Thread

    def tracking_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads_started.append(thread)
        return thread

    monkeypatch.setattr(v3_ui_module, "threading",
                        type("_T", (), {"Thread": staticmethod(tracking_thread)}))
    ui.state.go(F.SIGN_UP)
    ui.state.player = "BusyUser1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    ui.sign_up_submit()   # repeat click while the first is in flight
    ui.sign_up_submit()
    assert len(threads_started) == 1
    _wait_until_not_busy(ui)


def test_sign_up_taken_name_race_falls_back_to_hint(ui):
    """Format-level checks in v3_routes pass, but the account was taken
    by the time the async job actually ran (simulated here by seeding
    the store directly, bypassing the UI)."""
    account_store.create_account(ui._accounts._data, "RaceUser1", "password123")
    ui.state.go(F.SIGN_UP)
    ui.state.player = "RaceUser1"
    ui.state.password = ui.state.password2 = "differentpass1"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.SIGN_UP
    assert ui.state.hint_bad is True
    assert ui.state.hint_text == "Username already taken"
