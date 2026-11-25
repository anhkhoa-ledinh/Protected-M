"""
TASK_accounts_1119 Phase 5: saved progress per account (history, recent,
best score), the one-time guest toast, and the real leaderboard "you"
row. Same ``isolated_accounts`` pattern as test_account_session.py — a
monkeypatch on ``account_store``'s path functions redirects file I/O to
``tmp_path`` while ``V3UI`` still gets the real manifest/shaders.
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from src.shared import account_rules
from src.account import store as account_store
from src.ui_album import v3_flow as F
from src.ui_album import v3_values

ROOT = Path(".").resolve()


@pytest.fixture
def isolated_accounts(monkeypatch, tmp_path):
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
    from src.services.prefs_service import LocalPrefsService
    from src.services.shared_album_service import LocalSharedAlbumService
    from src.ui_album.v3_ui import V3UI

    options = AlbumOptions(enabled=True, ui="v3")
    return V3UI(ROOT, ROOT / "assets" / "shaders", options,
                accounts=LocalAccountService(__import__("tempfile").mkdtemp(prefix="hcpd_acc_")),
                shared_album=LocalSharedAlbumService(),
                prefs=LocalPrefsService(ROOT))


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


def _sign_up(ui, name, password="password123"):
    ui.state.go(F.SIGN_UP)
    ui.state.player = name
    ui.state.password = ui.state.password2 = password
    ui.sign_up_submit()
    _wait_until_not_busy(ui)


# ---- save_progress / _load_progress ----------------------------------------

def test_save_progress_writes_history_recent_and_best_score(ui, isolated_accounts):
    _sign_up(ui, "ProgUser1")
    ui.state.history = {"photo1.jpg": {"date": "01/01/2026", "eaten": 42.0, "score": 100}}
    ui.state.recent = ["photo1.jpg"]
    ui.save_progress()

    key = account_rules.username_key("ProgUser1")
    progress = account_store.load_progress(ui._accounts._data, key)
    assert progress["history"] == ui.state.history
    assert progress["recent"] == ["photo1.jpg"]
    assert progress["best_score"] == 100.0


def test_best_score_is_the_max_across_history(ui, isolated_accounts):
    _sign_up(ui, "ProgUser2")
    ui.state.history = {
        "a.jpg": {"score": 50},
        "b.jpg": {"score": 120},
        "c.jpg": {"score": 80},
    }
    ui.save_progress()
    key = account_rules.username_key("ProgUser2")
    progress = account_store.load_progress(ui._accounts._data, key)
    assert progress["best_score"] == 120.0


def test_save_progress_is_a_noop_for_guest(ui, isolated_accounts):
    ui.state.account_key = account_rules.GUEST_KEY
    ui.state.player = "guest"
    ui.state.history = {"photo1.jpg": {"score": 999}}
    ui.save_progress()
    assert account_rules.GUEST_KEY not in ui._accounts._data["accounts"]


def test_save_progress_is_a_noop_with_no_account_signed_in(ui, isolated_accounts):
    ui.state.account_key = ""
    ui.state.history = {"photo1.jpg": {"score": 999}}
    ui.save_progress()   # must not raise
    assert ui._accounts._data["accounts"] == {}


def test_history_does_not_leak_between_accounts(monkeypatch, isolated_accounts, gl_context):
    ui_a = _make_ui(monkeypatch)
    _sign_up(ui_a, "AccA1")
    ui_a.state.history = {"a.jpg": {"score": 10}}
    ui_a.save_progress()
    ui_a.release()

    ui_b = _make_ui(monkeypatch)
    _sign_up(ui_b, "AccB1")
    ui_b.state.history = {"b.jpg": {"score": 20}}
    ui_b.save_progress()

    key_a = account_rules.username_key("AccA1")
    progress_a = account_store.load_progress(ui_b._accounts._data, key_a)
    assert progress_a["history"] == {"a.jpg": {"score": 10}}
    key_b = account_rules.username_key("AccB1")
    progress_b = account_store.load_progress(ui_b._accounts._data, key_b)
    assert progress_b["history"] == {"b.jpg": {"score": 20}}
    ui_b.release()


def test_progress_survives_a_sign_out_and_sign_back_in(ui, isolated_accounts):
    _sign_up(ui, "ReloadUser1")
    ui.state.history = {"p.jpg": {"date": "02/02/2026", "eaten": 10.0, "score": 77}}
    ui.state.recent = ["p.jpg"]
    ui.sign_out()   # Phase 4's sign_out() also calls save_progress()

    ui.state.account_key = ""
    ui.state.history, ui.state.recent = {}, []
    ui.state.go(F.SIGN_IN)
    ui.state.player, ui.state.password = "ReloadUser1", "password123"
    ui.sign_in_submit()
    _wait_until_not_busy(ui)

    assert ui.state.history == {"p.jpg": {"date": "02/02/2026", "eaten": 10.0, "score": 77}}
    assert ui.state.recent == [Path("p.jpg")]


def test_progress_survives_a_full_relaunch(monkeypatch, isolated_accounts, gl_context):
    setup_ui = _make_ui(monkeypatch)
    _sign_up(setup_ui, "RelaunchUser1")
    setup_ui.state.history = {"q.jpg": {"date": "03/03/2026", "eaten": 5.0, "score": 55}}
    setup_ui.state.recent = ["q.jpg"]
    setup_ui.state.go(F.SIGN_IN)
    setup_ui.state.player, setup_ui.state.password = "RelaunchUser1", "password123"
    setup_ui.state.remember = True
    setup_ui.sign_in_submit()
    _wait_until_not_busy(setup_ui)
    # The sign-in above overwrote history with (empty) loaded progress -
    # simulate a real match finishing AFTER sign-in, then persist it the
    # way app.py's _tick_album_match does (note_match_finished).
    setup_ui.state.history = {"q.jpg": {"date": "03/03/2026", "eaten": 5.0, "score": 55}}
    setup_ui.state.recent = ["q.jpg"]
    setup_ui.note_match_finished()
    setup_ui.release()

    relaunched = _make_ui(monkeypatch)
    assert relaunched.state.screen == F.MENU   # remembered session
    assert relaunched.state.history == {"q.jpg": {"date": "03/03/2026", "eaten": 5.0, "score": 55}}
    assert relaunched.state.recent == [Path("q.jpg")]
    relaunched.release()


# ---- guest toast -------------------------------------------------------------

def test_guest_gets_a_one_time_toast_on_first_finished_match(ui, isolated_accounts):
    ui.state.account_key = account_rules.GUEST_KEY
    ui.state.player = "guest"
    assert ui._guest_toast_shown is False
    ui.note_match_finished()
    assert ui._guest_toast_shown is True
    assert "create an album to keep your scores" in ui.state.message


def test_guest_toast_shows_only_once(ui, isolated_accounts):
    ui.state.account_key = account_rules.GUEST_KEY
    ui.state.player = "guest"
    ui.note_match_finished()
    ui.state.message = ""   # simulate the toast expiring
    ui.note_match_finished()
    assert ui.state.message == ""   # did NOT fire a second time


def test_signed_in_account_gets_no_guest_toast(ui, isolated_accounts):
    _sign_up(ui, "NoToastUser1")
    ui.state.history = {"x.jpg": {"score": 1}}
    ui.note_match_finished()
    assert "guest" not in ui.state.message.lower()
    assert ui._guest_toast_shown is False


def test_a_failed_save_does_not_crash_the_match_result(ui, isolated_accounts, monkeypatch, caplog):
    """TASK_arch_refactor_1120 Part A: a save that raises must never take
    the result screen down with it (``note_match_finished`` is called
    right after the score is written into history)."""
    _sign_up(ui, "SaveFailUser1")
    ui.state.history = {"x.jpg": {"score": 1}}

    def boom(root, data):
        raise OSError("disk full")

    monkeypatch.setattr(account_store, "save", boom)
    with caplog.at_level("ERROR"):
        ui.note_match_finished()   # must not raise

    assert any(record.levelname == "ERROR" for record in caplog.records)
    assert ui.state.message == "Couldn't save your progress."


def test_guest_history_is_never_persisted_via_note_match_finished(ui, isolated_accounts):
    ui.state.account_key = account_rules.GUEST_KEY
    ui.state.player = "guest"
    ui.state.history = {"g.jpg": {"score": 500}}
    ui.note_match_finished()
    assert ui._accounts._data["accounts"] == {}


# ---- guest / sign-out do not leak a previous account's data ----------------

def test_switching_to_guest_after_sign_out_clears_history(monkeypatch, isolated_accounts, gl_context):
    from src.ui_album import v3_routes

    ui_instance = _make_ui(monkeypatch)
    _sign_up(ui_instance, "LeakUser1")
    # TASK_accounts_1119 Phase 6: sign-up now opens the one-time recovery
    # -code dialog, which is modal — dismiss it before anything else can
    # be clicked, same as a real player would.
    v3_routes.on_click(ui_instance.state, "btn_recovery_saved", 0.0)
    ui_instance.state.history = {"leak.jpg": {"score": 999}}
    ui_instance.save_progress()
    ui_instance.sign_out()
    ui_instance.state.account_key = ""
    ui_instance.state.go(F.SIGN_IN)

    v3_routes.on_click(ui_instance.state, "link_guest", 0.0)
    assert ui_instance.state.history == {}
    assert ui_instance.state.recent == []
    assert ui_instance.state.account_key == account_rules.GUEST_KEY
    ui_instance.release()


# ---- leaderboard "you" row ---------------------------------------------------

def test_board_you_row_uses_real_history(manifest_and_painter):
    manifest, painter = manifest_and_painter
    flow = F.V3Flow()
    flow.go(F.BOARD)
    flow.player = "Bo"   # short enough that "You · Bo" fits unfitted
    flow.account_key = "bo"
    accounts = {"accounts": {"bo": {"display": "Bo", "earned_total": 300,
                                    "levels_cleared": [1]}}}
    state = v3_values.state_for(manifest, painter, flow, accounts=accounts)
    assert state.texts["you_score"] == "300"
    assert state.texts["you_name"] == "You · Bo"


def test_board_you_row_name_is_fitted(manifest_and_painter):
    manifest, painter = manifest_and_painter
    flow = F.V3Flow()
    flow.go(F.BOARD)
    flow.player = "A" * 16
    flow.history = {}
    state = v3_values.state_for(manifest, painter, flow)
    record = manifest.text(F.BOARD, "you_name")
    width = painter.text_width([record["runs"][0]], [state.texts["you_name"]])
    assert width <= float(state.text_rect["you_name"][2])   # the row's real width


@pytest.fixture
def manifest_and_painter(gl_context):
    from src.ui_album.v3 import V3Manifest
    from src.ui_album.v3_painter import V3Painter

    manifest = V3Manifest(ROOT)
    painter = V3Painter(manifest, ROOT / "assets" / "shaders")
    painter.initialize()
    yield manifest, painter
    painter.release()
