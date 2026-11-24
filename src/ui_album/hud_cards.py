"""
TASK_game_feel_fix_1118 Fix 10.1: the HUD "panel map".

The user's playtest note: the HUD gets in the way with every gun, not
just the scope — cards drawn from ``top.png`` cover whatever the player
is aiming at. Fixing that needs to know, for a given screen, WHERE the
baked card art actually sits (so each card can fade independently) and
WHICH dynamic element (a text/bar/pip/pill/sprite key) belongs to which
card (so its alpha fades along with the card it sits on).

Card rects are derived from ``top.png``'s own alpha channel — each
connected component of non-transparent pixels is one region. A real
card (level/timer/hp/ammo panels, several thousand px² of baked art) is
kept as a *fading* card; a handful of tiny corner-rivet decorations (a
few hundred px²) are kept as regions too (so they still get drawn) but
never fade — they are not panels anyone reads, just trim.

Pure PIL/numpy/scipy — no GL — so this is unit-testable without a
window, and the per-(file) result is cached (computed once, not every
frame): "keep as constants" from the task's own wording.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

#: A card whose baked-art area (in SOURCE/asset pixels, before dividing by
#: ``image_scale``) is at least this big counts as a real, fadeable panel.
#: Below it: corner trim / rivets — always drawn, never faded. Measured
#: against the actual assets (TASK_game_feel_fix_1118 Fix 10 scan): real
#: cards are tens of thousands of px²; corner trim is ~200 px².
MIN_FADEABLE_AREA_PX2 = 3000.0


@dataclass(frozen=True)
class CardRect:
    """One region of a HUD's ``top.png``, in DESIGN-SPACE pixels (same
    space as every other manifest rect)."""

    rect: tuple[float, float, float, float]   # (x, y, w, h)
    fades: bool                               # False = corner trim, never fades
    art: bool = True                          # False = no baked art (item-slot group)

    def contains_point(self, x: float, y: float, grow: float = 0.0) -> bool:
        rx, ry, rw, rh = self.rect
        rx, ry, rw, rh = rx - grow, ry - grow, rw + 2.0 * grow, rh + 2.0 * grow
        return rx <= x <= rx + rw and ry <= y <= ry + rh

    def contains_rect(self, rect: tuple[float, float, float, float]) -> bool:
        """True if ``rect``'s CENTRE falls inside this card — used to
        assign a dynamic element to the card its rect sits on."""
        x, y, w, h = rect
        return self.contains_point(x + w * 0.5, y + h * 0.5)


def detect_cards(alpha: np.ndarray, image_scale: float,
                  min_fadeable_area_px2: float = MIN_FADEABLE_AREA_PX2
                  ) -> tuple[CardRect, ...]:
    """``alpha``: the ``top.png``'s alpha channel, full asset resolution
    (``image_scale`` applied, e.g. 1920x1080 for a 1280x720 design at
    1.5x). Returns every connected component of ``alpha > 0``, as
    DESIGN-SPACE rects, sorted top-to-bottom/left-to-right (deterministic
    iteration order for callers and tests)."""
    mask = alpha > 0
    labels, count = ndimage.label(mask)
    cards = []
    for label in range(1, count + 1):
        ys, xs = np.nonzero(labels == label)
        if ys.size == 0:
            continue
        area_px2 = float(ys.size)
        x0, x1 = float(xs.min()), float(xs.max()) + 1.0
        y0, y1 = float(ys.min()), float(ys.max()) + 1.0
        rect = (x0 / image_scale, y0 / image_scale,
                (x1 - x0) / image_scale, (y1 - y0) / image_scale)
        cards.append(CardRect(rect, fades=area_px2 >= min_fadeable_area_px2))
    cards.sort(key=lambda c: (round(c.rect[1]), round(c.rect[0])))
    return tuple(cards)


_cache: dict[str, tuple[CardRect, ...]] = {}


def cards_for_top_png(root: Path, relative: str, image_scale: float,
                      min_fadeable_area_px2: float = MIN_FADEABLE_AREA_PX2
                      ) -> tuple[CardRect, ...]:
    """``root``: the manifest's own asset root (``V3Manifest.root``);
    ``relative``: a screen record's ``"top"`` value. Cached by resolved
    path — computed once per file per process, not once per frame."""
    path = Path(root) / relative
    key = str(path)
    cached = _cache.get(key)
    if cached is not None:
        return cached
    with Image.open(path) as img:
        alpha = np.array(img.convert("RGBA"))[:, :, 3]
    cards = detect_cards(alpha, image_scale, min_fadeable_area_px2)
    _cache[key] = cards
    return cards


def card_index_for_rect(cards: tuple[CardRect, ...],
                        rect: tuple[float, float, float, float]) -> int | None:
    """Which card (by index into ``cards``) a dynamic element's rect
    belongs to — its CENTRE must fall inside a *fadeable* card. ``None``
    = "free" (crosshair, bite, effects, corner trim — nothing that should
    ever fade with a card). The item-slot group is a card too, but its
    contents are tagged by ``v3_values`` (``slot_labels``), not found here."""
    for index, card in enumerate(cards):
        if card.fades and card.art and card.contains_rect(rect):
            return index
    return None


def card_index_for_point(cards: tuple[CardRect, ...], point: tuple[float, float],
                         grow: float = 0.0) -> int | None:
    """Which fadeable card's (grown) rect contains ``point`` — used for
    the crosshair-proximity fade (Fix 10.2)."""
    x, y = point
    for index, card in enumerate(cards):
        if card.fades and card.contains_point(x, y, grow):
            return index
    return None


def cards_for_screen(manifest, screen: str) -> tuple[CardRect, ...]:
    """Convenience over :func:`cards_for_top_png` — takes the manifest +
    screen name directly, like the rest of ``v3.py``'s drawing code."""
    record = manifest.screens[screen]
    top = record.get("top")
    if not top:
        return ()
    cards = cards_for_top_png(manifest.root, top, manifest.image_scale)
    slots = record.get("item_slots")
    if not slots:
        return cards
    # The item slots are drawn live (no art in top.png), so they get one
    # extra fadeable card around the whole row: same crosshair-proximity
    # and scope fade as every other card. Appended last, so the indexes of
    # the top.png cards do not move.
    x0 = min(s[0] for s in slots)
    y0 = min(s[1] for s in slots)
    x1 = max(s[0] + s[2] for s in slots)
    y1 = max(s[1] + s[3] for s in slots)
    return cards + (CardRect((x0, y0, x1 - x0, y1 - y0), fades=True, art=False),)


def item_slot_card_index(cards: tuple[CardRect, ...]) -> int | None:
    """Index of the item-slot group card in ``cards`` (``None`` = none)."""
    for index, card in enumerate(cards):
        if not card.art:
            return index
    return None


_element_cache: dict[str, dict[str, int | None]] = {}


def element_card_map(manifest, screen: str) -> dict[str, int | None]:
    """``key`` (of every text/bar/pip/pill/sprite on ``screen``) -> the
    card index it rides along with, or ``None`` if free. Cached per
    screen — the manifest/assets don't change mid-process."""
    cached = _element_cache.get(screen)
    if cached is not None:
        return cached
    record = manifest.screens[screen]
    top = record.get("top")
    cards = (cards_for_top_png(manifest.root, top, manifest.image_scale)
            if top else ())
    mapping: dict[str, int | None] = {}
    for spec in record.get("texts", ()):
        mapping[spec["key"]] = card_index_for_rect(cards, tuple(spec["rect"]))
    for spec in record.get("bars", ()):
        mapping[spec["key"]] = card_index_for_rect(cards, tuple(spec["rect"]))
    for spec in record.get("pips", ()):
        # One pip row can span a wide area; its FIRST pip's position is
        # representative (a row never straddles two cards in practice).
        rects = spec.get("rects") or ()
        mapping[spec["key"]] = (card_index_for_rect(cards, tuple(rects[0]))
                                if rects else None)
    for spec in record.get("pills", ()):
        mapping[spec["key"]] = card_index_for_rect(cards, tuple(spec["rect"]))
    for spec in record.get("sprites", ()):
        mapping[spec["key"]] = card_index_for_rect(cards, tuple(spec["rect"]))
    _element_cache[screen] = mapping
    return mapping
