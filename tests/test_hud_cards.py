"""
TASK_game_feel_fix_1118 Fix 10.1: the HUD panel map — card rects derived
from each screen's ``top.png`` alpha channel, and the "every dynamic
element belongs to exactly one card, or is free" invariant the fix
spec itself asks to be tested.

Real assets, no GL (``detect_cards``/``card_index_for_rect`` are pure
numpy/PIL/scipy) — same posture as ``tests/test_fx_layout.py``.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.ui_album.hud_cards import (CardRect, card_index_for_point,
                                    card_index_for_rect, cards_for_top_png,
                                    detect_cards)
from src.ui_album.v3 import V3Manifest

MANIFEST = V3Manifest(".")

#: The three screens Fix 10 covers — every one with a baked top.png HUD.
HUD_SCREENS = ("v3_08_gameplay_hud", "v3_12a_pvp_hunter_hud", "v3_12b_pvp_bug_hud")

#: Element keys that must NEVER be assigned to a card — floating sprites
#: with their own art, not card content (the task's own examples: "the
#: crosshair, bite, effects" stay on top, plus the item-slot tiles, which
#: have no card art behind them at all on v3_08). ``st_block``/``head:*``
#: are deliberately NOT here — their rects sit squarely inside the hp/
#: stamina card and the ammo card respectively, so riding along with
#: that card's fade is correct, not a bug.
EXPECTED_FREE_SPRITE_PREFIXES = ("crosshair", "bite", "effect:", "islot:",
                                 "ikey:", "iused")

#: Floating feedback text that shows up wherever the player was aiming
#: (e.g. "Miss!"), not on any card — legitimately free on every screen.
#: The ``islot_*`` texts are the item-slot tiles' own key/name/count
#: labels — v3_08/v3_12a's item-slot area (bottom-centre) has NO card art
#: behind it at all, same reason the ``islot:*``/``ikey:*`` SPRITES above
#: are free: their background comes from the slot sprite itself.
def _is_free_text(key: str) -> bool:
    return (key in ("hit", "hit_sub")
            or key.startswith("islot") or key.startswith("effect_"))


def _cards(screen: str):
    record = MANIFEST.screens[screen]
    return cards_for_top_png(MANIFEST.root, record["top"], MANIFEST.image_scale)


# ---- detect_cards: pure geometry -------------------------------------------

def test_detect_cards_splits_two_separate_blobs():
    alpha = np.zeros((100, 200), dtype=np.uint8)
    alpha[10:30, 10:50] = 255     # a real card, 20*40=800... too small, bump up
    alpha[10:60, 10:90] = 255     # 50*80 = 4000 px^2 -> fadeable
    alpha[10:60, 120:190] = 255   # 50*70 = 3500 px^2 -> fadeable
    cards = detect_cards(alpha, image_scale=1.0, min_fadeable_area_px2=3000.0)
    assert len(cards) == 2
    assert all(c.fades for c in cards)


def test_detect_cards_marks_small_blob_as_non_fading():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[10:60, 10:90] = 255     # 50*80=4000 -> fadeable
    alpha[90:95, 90:95] = 255     # 5*5=25 -> trim, never fades
    cards = detect_cards(alpha, image_scale=1.0, min_fadeable_area_px2=3000.0)
    assert len(cards) == 2
    fades = sorted(c.fades for c in cards)
    assert fades == [False, True]


def test_detect_cards_converts_to_design_space_by_image_scale():
    alpha = np.zeros((100, 100), dtype=np.uint8)
    alpha[20:80, 10:70] = 255   # pixel rect (10,20,60,60)
    cards = detect_cards(alpha, image_scale=2.0, min_fadeable_area_px2=1.0)
    assert len(cards) == 1
    x, y, w, h = cards[0].rect
    assert (x, y, w, h) == pytest.approx((5.0, 10.0, 30.0, 30.0))


def test_detect_cards_empty_alpha_is_no_cards():
    alpha = np.zeros((50, 50), dtype=np.uint8)
    assert detect_cards(alpha, image_scale=1.0) == ()


# ---- CardRect.contains_point / contains_rect -------------------------------

def test_contains_point_respects_grow():
    card = CardRect((100.0, 100.0, 50.0, 50.0), fades=True)
    assert card.contains_point(90.0, 125.0) is False
    assert card.contains_point(90.0, 125.0, grow=20.0) is True


def test_contains_rect_uses_the_rects_centre():
    card = CardRect((100.0, 100.0, 50.0, 50.0), fades=True)
    # Centre at (125, 125) -> inside; even though the rect itself pokes
    # out the left edge.
    assert card.contains_rect((90.0, 110.0, 70.0, 30.0)) is True
    # Centre at (80, 80) -> outside.
    assert card.contains_rect((60.0, 60.0, 40.0, 40.0)) is False


# ---- card_index_for_rect / card_index_for_point ----------------------------

def test_card_index_for_rect_only_matches_fadeable_cards():
    cards = (CardRect((0.0, 0.0, 10.0, 10.0), fades=False),
             CardRect((0.0, 0.0, 10.0, 10.0), fades=True))
    assert card_index_for_rect(cards, (2.0, 2.0, 2.0, 2.0)) == 1


def test_card_index_for_point_none_when_outside_every_card():
    cards = (CardRect((0.0, 0.0, 10.0, 10.0), fades=True),)
    assert card_index_for_point(cards, (500.0, 500.0)) is None


# ---- real assets: the panel-map invariant ----------------------------------

@pytest.mark.parametrize("screen", HUD_SCREENS)
def test_real_hud_has_at_least_one_fadeable_card(screen):
    cards = _cards(screen)
    assert any(c.fades for c in cards)


@pytest.mark.parametrize("screen", HUD_SCREENS)
def test_real_hud_text_elements_are_assigned_or_free(screen):
    """Every text key either lands on exactly one card, or is one of the
    known free-floating feedback texts (e.g. "Miss!", shown near the
    crosshair rather than on any card)."""
    cards = _cards(screen)
    record = MANIFEST.screens[screen]
    unassigned = []
    for spec in record.get("texts", ()):
        key = spec["key"]
        if _is_free_text(key):
            assert card_index_for_rect(cards, tuple(spec["rect"])) is None
            continue
        if card_index_for_rect(cards, tuple(spec["rect"])) is None:
            unassigned.append(key)
    assert unassigned == []


@pytest.mark.parametrize("screen", HUD_SCREENS)
def test_real_hud_free_sprites_are_not_assigned_to_a_card(screen):
    """The crosshair/bite/effect/item-slot sprites must come back "free"
    (None) — they float over the scene with their own art, not on any
    baked card background."""
    cards = _cards(screen)
    record = MANIFEST.screens[screen]
    for sprite in record.get("sprites", ()):
        key = sprite["key"]
        if any(key.startswith(p) for p in EXPECTED_FREE_SPRITE_PREFIXES):
            index = card_index_for_rect(cards, tuple(sprite["rect"]))
            assert index is None, f"{screen}:{key} unexpectedly landed on card {index}"


def test_play_screen_card_count_matches_the_scanned_asset():
    """Pins down the actual v3_08_gameplay_hud layout: 5 real cards (top-
    left level/layer, top-centre timer, top-right, bottom-left hp/
    stamina, bottom-right weapon/ammo) plus 4 tiny corner-trim regions
    that never fade."""
    cards = _cards("v3_08_gameplay_hud")
    fading = [c for c in cards if c.fades]
    trim = [c for c in cards if not c.fades]
    assert len(fading) == 5
    assert len(trim) == 4


def test_cards_for_top_png_is_cached():
    record = MANIFEST.screens["v3_08_gameplay_hud"]
    a = cards_for_top_png(MANIFEST.root, record["top"], MANIFEST.image_scale)
    b = cards_for_top_png(MANIFEST.root, record["top"], MANIFEST.image_scale)
    assert a is b
