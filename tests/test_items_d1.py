"""D1: Flare and Thermal scan work; the other six are "Coming soon"
(logic-only: no window, no GL)."""

from __future__ import annotations

import pytest

from src.game.match import AlbumMatch
from src.game.pvp import ITEM_EFFECT_S, ITEM_WEIGHT, item_is_live
from src.game.scoring import match_score
from src.ui_album import v3_flow as F

LIVE = ("Flare", "Thermal scan")
SOON = tuple(n for n in ITEM_WEIGHT if n not in LIVE)


def _match(items, level=1):
    state = type("S", (), {"props": [{}] * 6, "found": [False] * 6})()
    return AlbumMatch(level=level, state=state, started_s=0.0, items=list(items),
                      weights=dict(ITEM_WEIGHT))


def test_exactly_two_items_are_live_and_six_are_soon():
    assert tuple(ITEM_EFFECT_S) == LIVE
    assert len(SOON) == 6 and not any(item_is_live(n) for n in SOON)
    assert all(ITEM_EFFECT_S[n] > 0 for n in LIVE)


@pytest.mark.parametrize("item", SOON)
def test_coming_soon_is_not_equippable(item):
    flow = F.V3Flow()
    flow.level = 1
    flow.wallet = 500
    F.toggle_item(flow, item)
    assert flow.items == [] and "coming soon" in flow.message.lower()
    assert flow.price_total() == 0                      # never charged


@pytest.mark.parametrize("item", LIVE)
def test_live_item_is_equippable(item):
    flow = F.V3Flow()
    flow.level = 1
    flow.wallet = 500
    F.toggle_item(flow, item)
    assert flow.items == [item]


def test_use_item_counts_once_and_expires():
    match = _match(["Flare"])
    assert not match.item_active("Flare", 10.0)
    assert match.use_item("Flare", 10.0)
    assert match.used == {"Flare": 1}
    assert match.item_active("Flare", 10.0 + ITEM_EFFECT_S["Flare"] - 0.1)
    assert not match.item_active("Flare", 10.0 + ITEM_EFFECT_S["Flare"] + 0.1)
    assert not match.use_item("Flare", 30.0)            # one use per match
    assert match.used == {"Flare": 1}


def test_use_item_refuses_what_is_not_equipped_or_not_working():
    match = _match(["Compass"])
    assert not match.use_item("Compass", 1.0)           # coming soon
    assert not match.use_item("Flare", 1.0)             # not equipped
    assert match.used == {}


def test_score_with_one_flare_equals_the_formula():
    match = _match(["Flare"])
    match.use_item("Flare", 1.0)
    match.ended = "win"
    left = match.time_left_s
    assert match.score() == match_score(1, False, left, {"Flare": 1}, ITEM_WEIGHT)
    assert match.score() == max(0, int(round(left * 1 - ITEM_WEIGHT["Flare"])))
