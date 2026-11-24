"""Points economy through the real V3UI (logic-only: GL context, no window)."""

from __future__ import annotations

import json

from src.account import store as account_store
from src.ui_album import v3_flow as F

from tests.test_account_progress import (_make_ui, _sign_up,  # noqa: F401
                                         isolated_accounts, ui)


def _accounts(tmp_path):
    return json.loads((tmp_path / "assets" / "config" / "accounts.json")
                      .read_text("utf-8"))["accounts"]


def test_new_account_shows_100(ui):
    _sign_up(ui, "Wally")
    assert ui.state.wallet == 100


def test_equip_pay_win_and_persist(ui, monkeypatch, isolated_accounts):
    _sign_up(ui, "Wally")
    F.toggle_item(ui.state, "Flare")
    assert ui.state.items == ["Flare"]
    assert ui.state.wallet == 100                 # nothing paid before the start
    assert ui.charge_items(ui.state.items)
    assert ui.state.wallet == 70                  # paid at match start
    ui.note_match_finished(won=True, level=2, score=120)
    assert (ui.state.wallet, ui.state.earned_total,
            ui.state.levels_cleared) == (190, 120, (2,))
    record = _accounts(isolated_accounts)["wally"]
    assert (record["wallet"], record["earned_total"],
            record["levels_cleared"]) == (190, 120, [2])
    # a restart on the same config dir reads the same numbers
    again = account_store.load(None).data["accounts"]["wally"]
    assert (again["wallet"], again["earned_total"]) == (190, 120)


def test_lost_match_adds_nothing(ui):
    _sign_up(ui, "Wally")
    ui.note_match_finished(won=False, level=1, score=0)
    assert ui.state.wallet == 100 and ui.state.earned_total == 0


def test_cannot_equip_without_points(ui):
    _sign_up(ui, "Wally")
    ui._account_data["accounts"]["wally"]["wallet"] = 20
    ui.sync_economy()
    F.toggle_item(ui.state, "Flare")              # costs 30
    assert ui.state.items == []
    assert "Not enough points" in ui.state.message


def test_two_items_must_fit_together(ui):
    _sign_up(ui, "Wally")
    ui.state.level = 3
    F.toggle_item(ui.state, "Thermal scan")       # 80
    F.toggle_item(ui.state, "Flare")              # 30 -> 110 > 100
    assert ui.state.items == ["Thermal scan"]


def test_guest_has_no_wallet_and_items_are_free(ui):
    ui.state.account_key = "guest"
    ui.sync_economy()
    assert ui.state.wallet is None
    F.toggle_item(ui.state, "Thermal scan")
    assert ui.state.items == ["Thermal scan"]
    assert ui.charge_items(ui.state.items)


def test_item_price_is_weight_times_ten():
    assert F.item_price("Flare") == 30 and F.item_price("Thermal scan") == 80
