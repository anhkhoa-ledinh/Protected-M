"""The account round trip across relaunches, in a temp root (logic-only: a
hidden GL context, no real window): create an account, quit; relaunch and sign
in with Remember me; relaunch lands in the menu; finish a match; relaunch and
the match and the wallet are still there."""

from __future__ import annotations

import time
from pathlib import Path

from src.account.service import LocalAccountService

from src.services.prefs_service import LocalPrefsService

from src.services.shared_album_service import LocalSharedAlbumService
from src.ui_album import v3_flow as F

ROOT = Path(".").resolve()


def _launch(tmp_path, monkeypatch):
    monkeypatch.setenv("HCPD_NO_FX", "1")
    from src.ui_album.options import AlbumOptions
    from src.ui_album.v3_ui import V3UI

    return V3UI(ROOT, ROOT / "assets" / "shaders", AlbumOptions(enabled=True, ui="v3"),
                accounts=LocalAccountService(tmp_path),
                shared_album=LocalSharedAlbumService(),
                prefs=LocalPrefsService(ROOT))


def _wait(ui):
    deadline = time.time() + 8
    while ui.account_busy and time.time() < deadline:
        ui.tick()
        time.sleep(0.005)
    assert not ui.account_busy


def test_create_remember_relaunch_and_a_finished_match_persist(gl_context, monkeypatch, tmp_path):
    ui = _launch(tmp_path, monkeypatch)
    try:
        assert ui.state.screen == F.SIGN_IN
        ui.state.go(F.SIGN_UP)
        ui.state.player, ui.state.password, ui.state.password2 = "RoundTrip1", "password123", "password123"
        ui.sign_up_submit()
        _wait(ui)
        assert ui.state.screen == F.MENU and ui.state.account_key == "roundtrip1"
        wallet_new = ui.state.wallet
    finally:
        ui.release()                                    # quit

    ui = _launch(tmp_path, monkeypatch)                 # relaunch: no session yet
    try:
        assert ui.state.screen == F.SIGN_IN and ui.state.player == "RoundTrip1"
        ui.state.password, ui.state.remember = "password123", True
        ui.sign_in_submit()
        _wait(ui)
        assert ui.state.screen == F.MENU and ui.state.account_key == "roundtrip1"
        assert ui.state.wallet == wallet_new
    finally:
        ui.release()                                    # quit with Remember me on

    ui = _launch(tmp_path, monkeypatch)                 # relaunch lands in the menu
    try:
        assert ui.state.screen == F.MENU and ui.state.account_key == "roundtrip1"
        ui.state.history = {"photo1.png": {"date": "24/11/2026", "eaten": 6.0, "score": 150}}
        ui.state.recent = ["photo1.png"]
        ui.note_match_finished(won=True, level=1, score=150)
        wallet_after = ui.state.wallet
        assert wallet_after == wallet_new + 150
    finally:
        ui.release()

    ui = _launch(tmp_path, monkeypatch)                 # relaunch: match and wallet are there
    try:
        assert ui.state.screen == F.MENU
        assert ui.state.wallet == wallet_after
        assert ui.state.levels_cleared == (1,)
        assert ui.state.history["photo1.png"]["score"] == 150
    finally:
        ui.release()
