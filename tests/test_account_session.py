"""
TASK_accounts_1119 Phase 4: "remember me" sessions, auto sign-in at
launch, sign-out, and pre-filling the last username.

The auto-sign-in check happens inside ``V3UI.__init__`` itself, keyed
off the SAME ``root`` the manifest/shaders load from — so these tests
monkeypatch ``account_store.accounts_path``/``session_path`` (the two
functions every store call goes through) to redirect to an isolated
``tmp_path``, while ``V3UI`` still gets the REAL project root for the
manifest/font files it also needs. A real GL context is required (the
``V3Painter`` constructor compiles a shader), but ``HCPD_NO_FX=1`` keeps
construction cheap — no fx preload.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.shared import account_rules
from src.account import store as account_store
from src.ui_album import v3_flow as F

ROOT = Path(".").resolve()


@pytest.fixture
def isolated_accounts(monkeypatch, tmp_path):
    """Redirects every ``account_store`` file operation to ``tmp_path``,
    regardless of what ``root`` a caller passes — lets ``V3UI(ROOT, ...)``
    use the real manifest/shaders while its account files stay isolated."""
    def fake_accounts_path(root):
        return tmp_path / "assets" / "config" / "accounts.json"

    def fake_session_path(root):
        return tmp_path / "assets" / "config" / "session.json"

    monkeypatch.setattr(account_store, "accounts_path", fake_accounts_path)
    monkeypatch.setattr(account_store, "session_path", fake_session_path)
    return tmp_path


def _make_ui(monkeypatch):
    monkeypatch.setenv("HCPD_NO_FX", "1")
    from src.ui_album.options import AlbumOptions
    from src.account.service import LocalAccountService
    from src.ui_album.v3_ui import V3UI

    options = AlbumOptions(enabled=True, ui="v3")
    return V3UI(ROOT, ROOT / "assets" / "shaders", options,
                accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")))


def _wait_until_not_busy(ui, timeout=5.0):
    deadline = time.time() + timeout
    while ui.account_busy and time.time() < deadline:
        ui.tick()
        time.sleep(0.005)
    assert not ui.account_busy


@pytest.fixture
def ui(gl_context, monkeypatch, isolated_accounts):
    instance = _make_ui(monkeypatch)
    yield instance
    instance.release()


# ---- a session is created when the switch is on ----------------------------

def test_session_created_on_sign_in_with_remember_on(ui, isolated_accounts):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "SessUser1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "SessUser1", "password123"
    ui.state.remember = True
    ui.sign_in_submit()
    _wait_until_not_busy(ui)

    assert ui.state.screen == F.MENU
    assert account_store.session_path(isolated_accounts).exists()
    key = account_rules.username_key("SessUser1")
    assert ui._accounts._data["accounts"][key]["session_token_hash"] is not None


def test_no_session_created_with_remember_off(ui, isolated_accounts):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "NoSessUser1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "NoSessUser1", "password123"
    ui.state.remember = False
    ui.sign_in_submit()
    _wait_until_not_busy(ui)

    assert ui.state.screen == F.MENU
    assert not account_store.session_path(isolated_accounts).exists()


def test_switching_remember_off_clears_a_previous_session(ui, isolated_accounts):
    """A session created on an EARLIER sign-in (switch was on) must not
    outlive signing in again with the switch now off."""
    ui.state.go(F.SIGN_UP)
    ui.state.player = "FlipUser1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "FlipUser1", "password123"
    ui.state.remember = True
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert account_store.session_path(isolated_accounts).exists()

    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "FlipUser1", "password123"
    ui.state.remember = False
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert not account_store.session_path(isolated_accounts).exists()


# ---- auto sign-in at launch, via V3UI.__init__ ------------------------------

def test_auto_sign_in_on_a_fresh_launch_with_a_valid_session(monkeypatch, isolated_accounts, gl_context):
    setup_ui = _make_ui(monkeypatch)
    setup_ui.state.go(F.SIGN_UP)
    setup_ui.state.player = "AutoUser1"
    setup_ui.state.password = setup_ui.state.password2 = "password123"
    setup_ui.sign_up_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "AutoUser1", "password123"
    setup_ui.state.remember = True
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.release()

    # A SECOND, fresh V3UI (simulates a new launch) must land straight
    # on MENU, never showing the sign-in screen.
    relaunched = _make_ui(monkeypatch)
    assert relaunched.state.screen == F.MENU
    assert relaunched.state.account_key == account_rules.username_key("AutoUser1")
    assert relaunched.state.player == "AutoUser1"
    relaunched.release()


def test_auto_sign_in_refreshes_the_expiry(monkeypatch, isolated_accounts, gl_context):
    setup_ui = _make_ui(monkeypatch)
    setup_ui.state.go(F.SIGN_UP)
    setup_ui.state.player = "RefreshUser1"
    setup_ui.state.password = setup_ui.state.password2 = "password123"
    setup_ui.sign_up_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "RefreshUser1", "password123"
    setup_ui.state.remember = True
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    key = account_rules.username_key("RefreshUser1")
    first_expiry = setup_ui._accounts._data["accounts"][key]["session_expires_at"]
    setup_ui.release()

    time.sleep(0.05)
    relaunched = _make_ui(monkeypatch)
    second_expiry = relaunched._accounts._data["accounts"][key]["session_expires_at"]
    assert second_expiry > first_expiry
    relaunched.release()


def test_expired_token_does_not_auto_sign_in(monkeypatch, isolated_accounts, gl_context):
    setup_ui = _make_ui(monkeypatch)
    setup_ui.state.go(F.SIGN_UP)
    setup_ui.state.player = "ExpiredUser1"
    setup_ui.state.password = setup_ui.state.password2 = "password123"
    setup_ui.sign_up_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "ExpiredUser1", "password123"
    setup_ui.state.remember = True
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    account_key = account_rules.username_key("ExpiredUser1")
    setup_ui._accounts._data["accounts"][account_key]["session_expires_at"] = 1.0  # way in the past
    account_store.save(isolated_accounts, setup_ui._accounts._data)
    setup_ui.release()

    relaunched = _make_ui(monkeypatch)
    assert relaunched.state.screen != F.MENU
    relaunched.release()


def test_tampered_token_does_not_auto_sign_in(monkeypatch, isolated_accounts, gl_context):
    import json

    setup_ui = _make_ui(monkeypatch)
    setup_ui.state.go(F.SIGN_UP)
    setup_ui.state.player = "TamperUser1"
    setup_ui.state.password = setup_ui.state.password2 = "password123"
    setup_ui.sign_up_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "TamperUser1", "password123"
    setup_ui.state.remember = True
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.release()

    session_file = account_store.session_path(isolated_accounts)
    payload = json.loads(session_file.read_text(encoding="utf-8"))
    payload["token"] = "not-the-real-token"
    session_file.write_text(json.dumps(payload), encoding="utf-8")

    relaunched = _make_ui(monkeypatch)
    assert relaunched.state.screen != F.MENU
    relaunched.release()


def test_sign_out_removes_the_session(ui, isolated_accounts):
    ui.state.go(F.SIGN_UP)
    ui.state.player = "OutUser1"
    ui.state.password = ui.state.password2 = "password123"
    ui.sign_up_submit()
    _wait_until_not_busy(ui)
    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "OutUser1", "password123"
    ui.state.remember = True
    ui.sign_in_submit()
    _wait_until_not_busy(ui)
    assert account_store.session_path(isolated_accounts).exists()

    ui.sign_out()
    assert not account_store.session_path(isolated_accounts).exists()


def test_sign_out_then_relaunch_shows_sign_in(monkeypatch, isolated_accounts, gl_context):
    setup_ui = _make_ui(monkeypatch)
    setup_ui.state.go(F.SIGN_UP)
    setup_ui.state.player = "OutUser2"
    setup_ui.state.password = setup_ui.state.password2 = "password123"
    setup_ui.sign_up_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "OutUser2", "password123"
    setup_ui.state.remember = True
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.sign_out()
    setup_ui.release()

    relaunched = _make_ui(monkeypatch)
    assert relaunched.state.screen != F.MENU
    relaunched.release()


# ---- pre-fill the last username with no session ----------------------------

def test_prefill_last_username_with_no_session(monkeypatch, isolated_accounts, gl_context):
    setup_ui = _make_ui(monkeypatch)
    setup_ui.state.go(F.SIGN_UP)
    setup_ui.state.player = "PrefillUser1"
    setup_ui.state.password = setup_ui.state.password2 = "password123"
    setup_ui.sign_up_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "PrefillUser1", "password123"
    setup_ui.state.remember = False     # no session, just the "last account" pointer
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    setup_ui.release()

    relaunched = _make_ui(monkeypatch)
    assert relaunched.state.screen != F.MENU   # no session -> sign-in screen
    assert relaunched.state.player == "PrefillUser1"
    relaunched.release()


def test_no_prefill_when_no_account_has_ever_signed_in(monkeypatch, isolated_accounts, gl_context):
    fresh = _make_ui(monkeypatch)
    assert fresh.state.player == ""
    fresh.release()
