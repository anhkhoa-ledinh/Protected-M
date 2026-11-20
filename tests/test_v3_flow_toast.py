"""
TASK_game_feel_1118 Phase 3: a match-only toast (e.g. "Reloading") must
not survive past the match screens — 1117's own review screenshot
(``fx_win.png``) caught one still showing on WIN.
"""

from __future__ import annotations

from src.ui_album import v3_flow as F


def _flow_with_toast(screen: str) -> F.V3Flow:
    flow = F.V3Flow()
    flow.screen = screen
    F.say(flow, "Reloading")
    assert flow.message == "Reloading"
    return flow


def test_leaving_play_clears_the_toast():
    flow = _flow_with_toast(F.PLAY)
    flow.go(F.WIN)
    assert flow.message == ""
    assert flow.message_until_s == 0.0


def test_leaving_pvp_hunt_clears_the_toast():
    flow = _flow_with_toast(F.PVP_HUNT)
    flow.go(F.PVP_RESULT)
    assert flow.message == ""


def test_leaving_pvp_hide_clears_the_toast():
    flow = _flow_with_toast(F.PVP_HIDE)
    flow.go(F.PVP_RESULT)
    assert flow.message == ""


def test_toast_set_on_a_non_match_screen_is_left_alone():
    flow = _flow_with_toast(F.LOADOUT)
    flow.go(F.TILT)
    assert flow.message == "Reloading"


def test_going_to_the_same_screen_is_a_noop_and_keeps_the_toast():
    flow = _flow_with_toast(F.PLAY)
    flow.go(F.PLAY)
    assert flow.message == "Reloading"
