"""
TASK_accounts_1119 Phase 6: the one-time recovery-code dialog shown
after sign-up, and the offline forgot-password dialog (verify code ->
reset password -> a NEW recovery code, since the old one is used up).

Same split as Phase 2's test_account_flow.py: ``v3_routes.on_click`` and
``v3_flow``'s field helpers are pure (``V3Flow`` only, no GL); the async
verify/reset round trip needs a real ``V3UI`` (``gl_context`` for the
``V3Painter`` shader compile) with its account files redirected to an
isolated ``tmp_path``.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.shared import account_rules
from src.account import store as account_store
from src.ui_album import v3_flow as F
from src.ui_album import v3_routes

ROOT = Path(".").resolve()


def _flow(screen: str) -> F.V3Flow:
    flow = F.V3Flow()
    flow.go(screen)
    return flow


# ---- v3_routes.on_click: the recovery-code dialog, pure, no GL ------------

def test_recovery_copy_click_returns_the_copy_action():
    flow = _flow(F.MENU)
    flow.recovery_code_shown = "ABCD-EFGH-JKMN"
    action = v3_routes.on_click(flow, "btn_recovery_copy", 0.0)
    assert action == v3_routes.ACT_COPY_RECOVERY_CODE
    assert flow.recovery_code_shown == "ABCD-EFGH-JKMN"   # copy does not close it


def test_recovery_saved_click_closes_the_dialog_and_returns_the_action():
    flow = _flow(F.MENU)
    flow.recovery_code_shown = "ABCD-EFGH-JKMN"
    action = v3_routes.on_click(flow, "btn_recovery_saved", 0.0)
    assert action == v3_routes.ACT_RECOVERY_SAVED
    assert flow.recovery_code_shown == ""


def test_recovery_dialog_is_modal_other_clicks_do_nothing():
    """The dialog covers the whole screen — a click meant for whatever
    is underneath (MENU, say) must not reach it while the dialog is
    open, same rule as the pre-existing share-album confirm dialog."""
    flow = _flow(F.MENU)
    flow.recovery_code_shown = "ABCD-EFGH-JKMN"
    action = v3_routes.on_click(flow, "btn_Solo", 0.0)
    assert action == ""
    assert flow.screen == F.MENU            # did NOT navigate
    assert flow.recovery_code_shown == "ABCD-EFGH-JKMN"   # still open


# ---- v3_routes.on_click: opening/closing the forgot-password dialog -------

def test_link_forgot_opens_the_dialog_prefilled_and_focused():
    flow = _flow(F.SIGN_IN)
    flow.player = "SomeUser1"
    action = v3_routes.on_click(flow, "link_forgot", 0.0)
    assert action == ""
    assert flow.forgot_open is True
    assert flow.forgot_user == "SomeUser1"
    assert flow.forgot_code == flow.forgot_new_pass == flow.forgot_new_pass2 == ""
    assert flow.forgot_focus == "field_forgot_user"
    assert flow.forgot_touched == set()


def test_forgot_cancel_closes_the_dialog():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    action = v3_routes.on_click(flow, "btn_forgot_cancel", 0.0)
    assert action == ""
    assert flow.forgot_open is False
    assert flow.forgot_focus is None


def test_forgot_dialog_is_modal_sign_in_clicks_do_not_reach_it():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    action = v3_routes.on_click(flow, "btn_Sign in", 0.0)
    assert action == ""
    assert flow.forgot_open is True           # the sign-in button never fired


def test_field_click_moves_forgot_focus():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    v3_routes.on_click(flow, "field_forgot_pass", 0.0)
    assert flow.forgot_focus == "field_forgot_pass"


# ---- v3_routes.on_click: btn_forgot_submit's FORMAT-only checks -----------

def test_forgot_submit_bad_username_shows_reason_no_action():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    flow.forgot_user = "ab"   # too short
    flow.forgot_code = "ABCDEFGHJKMN"
    flow.forgot_new_pass = flow.forgot_new_pass2 = "password123"
    action = v3_routes.on_click(flow, "btn_forgot_submit", 0.0)
    assert action == ""
    assert flow.forgot_error
    assert flow.forgot_touched == {"field_forgot_user", "field_forgot_code",
                                   "field_forgot_pass", "field_forgot_pass2"}


def test_forgot_submit_short_code_shows_reason_no_action():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    flow.forgot_user = "gooduser"
    flow.forgot_code = "ABCD"
    flow.forgot_new_pass = flow.forgot_new_pass2 = "password123"
    action = v3_routes.on_click(flow, "btn_forgot_submit", 0.0)
    assert action == ""
    assert "12-character" in flow.forgot_error


def test_forgot_submit_short_password_shows_reason_no_action():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    flow.forgot_user = "gooduser"
    flow.forgot_code = "ABCDEFGHJKMN"
    flow.forgot_new_pass = flow.forgot_new_pass2 = "short1"
    action = v3_routes.on_click(flow, "btn_forgot_submit", 0.0)
    assert action == ""
    assert flow.forgot_error


def test_forgot_submit_mismatched_passwords_shows_reason_no_action():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    flow.forgot_user = "gooduser"
    flow.forgot_code = "ABCDEFGHJKMN"
    flow.forgot_new_pass, flow.forgot_new_pass2 = "password123", "password124"
    action = v3_routes.on_click(flow, "btn_forgot_submit", 0.0)
    assert action == ""
    assert "match" in flow.forgot_error.lower()


def test_forgot_submit_format_valid_returns_the_async_action():
    flow = _flow(F.SIGN_IN)
    v3_routes.on_click(flow, "link_forgot", 0.0)
    flow.forgot_user = "gooduser"
    flow.forgot_code = "ABCD-EFGH-JKMN"   # dashes allowed, stripped before checking
    flow.forgot_new_pass = flow.forgot_new_pass2 = "password123"
    action = v3_routes.on_click(flow, "btn_forgot_submit", 0.0)
    assert action == v3_routes.ACT_FORGOT_SUBMIT
    assert flow.forgot_error == ""
    assert flow.forgot_open is True   # still open — the real verdict is async


# ---- v3_flow: the forgot dialog's own field editing, pure, no GL ----------

def test_forgot_type_char_routes_to_the_focused_field():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = "field_forgot_code"
    F.forgot_type_char(flow, "A")
    F.forgot_type_char(flow, "B")
    assert flow.forgot_code == "AB"
    assert flow.forgot_user == ""
    assert "field_forgot_code" in flow.forgot_touched


def test_forgot_type_char_does_nothing_with_no_focus():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = None
    F.forgot_type_char(flow, "A")
    assert flow.forgot_user == flow.forgot_code == flow.forgot_new_pass == ""


def test_forgot_backspace_removes_the_last_character():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = "field_forgot_user"
    flow.forgot_user = "abc"
    flow.caret_index = 3   # TASK_small_fixes_1120: backspace deletes LEFT of the caret
    F.forgot_backspace(flow)
    assert flow.forgot_user == "ab"


def test_forgot_backspace_deletes_at_the_caret_not_always_the_end():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = "field_forgot_user"
    flow.forgot_user = "abc"
    flow.caret_index = 1   # caret between 'a' and 'b'
    F.forgot_backspace(flow)
    assert flow.forgot_user == "bc"
    assert flow.caret_index == 0


def test_forgot_field_respects_its_own_maxlen():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = "field_forgot_pass"
    for _ in range(80):
        F.forgot_type_char(flow, "x")
    assert len(flow.forgot_new_pass) == F.FORGOT_FIELD_MAXLEN["field_forgot_pass"]


def test_forgot_focus_next_cycles_through_the_fixed_order():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = "field_forgot_user"
    seen = [flow.forgot_focus]
    for _ in range(len(F.FORGOT_FIELD_ORDER)):
        F.forgot_focus_next(flow)
        seen.append(flow.forgot_focus)
    assert seen == list(F.FORGOT_FIELD_ORDER) + [F.FORGOT_FIELD_ORDER[0]]


def test_forgot_focus_next_with_no_prior_focus_lands_on_first_field():
    flow = _flow(F.SIGN_IN)
    flow.forgot_focus = None
    F.forgot_focus_next(flow)
    assert flow.forgot_focus == F.FORGOT_FIELD_ORDER[0]


# ---- V3UI keyboard routing: Tab/Enter/Esc respect whichever dialog is open --

@pytest.fixture
def ui(gl_context, monkeypatch, tmp_path):
    """TASK_perf_textures_1119 Fix 8: redirect account_store's path
    functions BEFORE constructing V3UI, not override ``_account_root``/
    ``_account_data`` after -- by then ``__init__`` has already read the
    REAL ``accounts.json``/``session.json`` (last_account pre-fills
    ``state.player``, a "remember me" session auto-signs in) and a
    developer's real signed-in machine made
    ``test_key_routes_into_the_forgot_dialog_when_open`` fail
    (``state.player`` wasn't "" to begin with). Same pattern as
    ``tests/test_account_session.py``'s ``isolated_accounts`` fixture."""
    monkeypatch.setenv("HCPD_NO_FX", "1")
    from src.ui_album.options import AlbumOptions
    from src.account.service import LocalAccountService
    from src.ui_album.v3_ui import V3UI

    monkeypatch.setattr(account_store, "accounts_path",
                        lambda root: tmp_path / "assets" / "config" / "accounts.json")
    monkeypatch.setattr(account_store, "session_path",
                        lambda root: tmp_path / "assets" / "config" / "session.json")
    options = AlbumOptions(enabled=True, ui="v3")
    instance = V3UI(ROOT, ROOT / "assets" / "shaders", options,
                    accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")))
    yield instance
    instance.release()


def _wait_until_not_busy(ui, timeout=5.0):
    deadline = time.time() + timeout
    while ui.account_busy and time.time() < deadline:
        ui.tick()
        time.sleep(0.005)
    assert not ui.account_busy, "account job never finished"


def _sign_up(ui, name, password="password123"):
    ui.state.go(F.SIGN_UP)
    ui.state.player = name
    ui.state.password = ui.state.password2 = password
    ui.sign_up_submit()
    _wait_until_not_busy(ui)
    code = ui.state.recovery_code_shown
    ui.activate("btn_recovery_saved")   # dismiss the one-time dialog
    return code


def _fill_forgot(ui, username, code, new_pass, new_pass2=None):
    new_pass2 = new_pass if new_pass2 is None else new_pass2
    ui.activate("field_forgot_user")
    ui.state.forgot_user = ""
    for ch in username:
        ui.key(ch)
    ui.activate("field_forgot_code")
    ui.state.forgot_code = ""
    for ch in code:
        ui.key(ch)
    ui.activate("field_forgot_pass")
    ui.state.forgot_new_pass = ""
    for ch in new_pass:
        ui.key(ch)
    ui.activate("field_forgot_pass2")
    ui.state.forgot_new_pass2 = ""
    for ch in new_pass2:
        ui.key(ch)


def test_key_routes_into_the_forgot_dialog_when_open(ui):
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    ui.key("x")
    assert ui.state.forgot_user == "x"
    assert ui.state.player == ""   # the screen underneath is untouched


def test_tab_moves_forgot_focus(ui):
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    ui.focus_next()
    assert ui.state.forgot_focus == "field_forgot_code"


def test_esc_closes_the_forgot_dialog_with_no_action(ui):
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    action = ui.back()
    assert action == ""
    assert ui.state.forgot_open is False


def test_esc_closes_the_recovery_dialog(ui):
    _sign_up_code = None
    ui.state.go(F.SIGN_UP)
    ui.state.player = "EscUser1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)
    assert ui.state.recovery_code_shown
    action = ui.back()
    assert action == ""
    assert ui.state.recovery_code_shown == ""


def test_enter_submits_the_forgot_dialog(ui):
    code = _sign_up(ui, "EnterUser1")
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "EnterUser1", code, "newpassword1")
    action = ui.submit()
    assert action == v3_routes.ACT_FORGOT_SUBMIT


def test_enter_dismisses_the_recovery_dialog(ui):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "EnterUser2"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)
    action = ui.submit()
    assert action == v3_routes.ACT_RECOVERY_SAVED
    assert ui.state.recovery_code_shown == ""


# ---- the full async verify-recovery-code -> reset -> new-code round trip --

def test_forgot_submit_unknown_username_sets_error(ui):
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "nosuchuser", "ABCDEFGHJKMN", "newpassword1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    assert "No album named nosuchuser" in ui.state.forgot_error
    assert ui.state.forgot_open is True


def test_forgot_submit_wrong_code_sets_error(ui):
    code = _sign_up(ui, "WrongCodeUser1")
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    wrong = "A" * 12 if code[:12].replace("-", "") != "A" * 12 else "B" * 12
    _fill_forgot(ui, "WrongCodeUser1", wrong, "newpassword1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    assert ui.state.forgot_error == "Wrong code. Try again."
    assert ui.state.forgot_open is True


def test_forgot_submit_correct_code_resets_password_and_shows_a_new_code(ui):
    old_code = _sign_up(ui, "ResetUser1")
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "ResetUser1", old_code, "brandnewpass1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)

    assert ui.state.forgot_open is False
    assert ui.state.recovery_code_shown
    new_code = ui.state.recovery_code_shown
    assert new_code != old_code

    # the password was actually changed
    ui.activate("btn_recovery_saved")
    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "ResetUser1", "brandnewpass1"
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert ui.state.screen == F.MENU


def test_old_recovery_code_is_dead_after_a_reset(ui):
    old_code = _sign_up(ui, "DeadCodeUser1")
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "DeadCodeUser1", old_code, "firstreset1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    ui.activate("btn_recovery_saved")

    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "DeadCodeUser1", old_code, "secondreset1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    assert ui.state.forgot_open is True   # the old code did NOT work a second time
    assert ui.state.forgot_error == "Wrong code. Try again."


def test_new_code_resets_again_after_the_first_reset(ui):
    old_code = _sign_up(ui, "ChainUser1")
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "ChainUser1", old_code, "firstreset1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    new_code = ui.state.recovery_code_shown
    ui.activate("btn_recovery_saved")

    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "ChainUser1", new_code, "secondreset1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    assert ui.state.forgot_open is False
    assert ui.state.recovery_code_shown
    assert ui.state.recovery_code_shown != new_code


def test_five_wrong_codes_trigger_the_cooldown(ui):
    code = _sign_up(ui, "CooldownUser1")
    wrong = "Z" * 12 if code.replace("-", "") != "Z" * 12 else "Y" * 12
    for _ in range(account_store.MAX_FAILURES):
        ui.state.go(F.SIGN_IN)
        ui.activate("link_forgot")
        _fill_forgot(ui, "CooldownUser1", wrong, "somepassword1")
        ui.forgot_submit()
        _wait_until_not_busy(ui)
    assert "Too many tries" in ui.state.forgot_error

    key = account_rules.username_key("CooldownUser1")
    assert ui._accounts._throttle.cooldown_remaining(key, time.perf_counter()) > 0.0


def test_cooldown_is_shared_between_sign_in_and_forgot_attempts(ui):
    """Task: "wrong codes use the SAME 5-try/30s cooldown" — a mix of
    wrong sign-in passwords and wrong recovery codes for the same name
    must count toward the same throttle."""
    code = _sign_up(ui, "SharedThrotUser1")
    wrong_code = "Z" * 12 if code.replace("-", "") != "Z" * 12 else "Y" * 12

    for _ in range(account_store.MAX_FAILURES - 1):
        ui.state.go(F.SIGN_IN)
        ui.state.player, ui.state.password = "SharedThrotUser1", "wrongpassword"
        ui.sign_in_submit()
        _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "SharedThrotUser1", wrong_code, "somepassword1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    assert "Too many tries" in ui.state.forgot_error

    key = account_rules.username_key("SharedThrotUser1")
    assert ui._accounts._throttle.cooldown_remaining(key, time.perf_counter()) > 0.0


def test_a_successful_reset_resets_the_shared_throttle(ui):
    old_code = _sign_up(ui, "ThrotResetUser1")
    wrong_code = "Z" * 12 if old_code.replace("-", "") != "Z" * 12 else "Y" * 12
    for _ in range(account_store.MAX_FAILURES - 1):
        ui.state.go(F.SIGN_IN)
        ui.activate("link_forgot")
        _fill_forgot(ui, "ThrotResetUser1", wrong_code, "somepassword1")
        ui.forgot_submit()
        _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "ThrotResetUser1", old_code, "finalpassword1")
    ui.forgot_submit()
    _wait_until_not_busy(ui)
    assert ui.state.forgot_open is False   # the correct code still worked

    key = account_rules.username_key("ThrotResetUser1")
    assert ui._accounts._throttle.cooldown_remaining(key, time.perf_counter()) == 0.0


def test_cooldown_check_happens_before_any_hashing_no_thread_started(ui, monkeypatch):
    """Mirrors Phase 2's busy-guard test: once a name is in cooldown,
    ``forgot_submit`` must not even spin up a worker thread — a cheap,
    synchronous check in memory, same as ``sign_in_submit``."""
    import threading

    import src.ui_album.v3_ui as v3_ui_module

    code = _sign_up(ui, "NoThreadUser1")
    wrong_code = "Z" * 12 if code.replace("-", "") != "Z" * 12 else "Y" * 12
    for _ in range(account_store.MAX_FAILURES):
        ui.state.go(F.SIGN_IN)
        ui.activate("link_forgot")
        _fill_forgot(ui, "NoThreadUser1", wrong_code, "somepassword1")
        ui.forgot_submit()
        _wait_until_not_busy(ui)
    assert "Too many tries" in ui.state.forgot_error

    threads_started = []
    real_thread = threading.Thread

    def tracking_thread(*args, **kwargs):
        thread = real_thread(*args, **kwargs)
        threads_started.append(thread)
        return thread

    monkeypatch.setattr(v3_ui_module, "threading",
                        type("_T", (), {"Thread": staticmethod(tracking_thread)}))
    ui.state.go(F.SIGN_IN)
    ui.activate("link_forgot")
    _fill_forgot(ui, "NoThreadUser1", wrong_code, "anotherpassword1")
    ui.forgot_submit()
    assert threads_started == []   # still cooling down — no hash, no thread


# ---- clipboard copy ---------------------------------------------------------

def test_copy_recovery_code_writes_to_the_window_clipboard(ui):
    class FakeWindow:
        def __init__(self):
            self.clip = None

        def set_clipboard(self, text):
            self.clip = text

    fake = FakeWindow()
    ui._window = fake
    ui.state.recovery_code_shown = "ABCD-EFGH-JKMN"
    ui.copy_recovery_code()
    assert fake.clip == "ABCD-EFGH-JKMN"


def test_copy_recovery_code_with_no_window_does_not_crash(ui):
    ui._window = None
    ui.state.recovery_code_shown = "ABCD-EFGH-JKMN"
    ui.copy_recovery_code()   # must not raise
