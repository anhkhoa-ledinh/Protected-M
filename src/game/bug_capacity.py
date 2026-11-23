"""How many bugs a photo can really hold for a level, without opening a match.

Runs the SAME steps ``HcpdApplication._start_album_match`` runs (``place_props``
at ``count × 3``, the viewport filter, the 5×5 exposure check), so with the
same seed and rest eye the number equals what the match then places. Pure
logic, no GL; the loading screen uses it to say the real count up front.

``choose_layout`` tries ``LAYOUT_TRIES`` layouts (seeds derived from the match
seed) and keeps the fullest, so a lucky or unlucky single draw cannot decide
how many bugs a match gets.
"""

from __future__ import annotations

import random
from typing import NamedTuple

from src.content.hidden_props_search import exposure_count, place_props
from src.content.prop_rest_eye import z_rest_mm
from src.game.levels import LEGACY_PREFER
from src.game.viewport_filter import filter_props

#: Layouts tried per match; the one with the most bugs is kept.
LAYOUT_TRIES = 8


class Layout(NamedTuple):
    kept: list        # winnable props, at most ``count``
    outside: int      # placed but projected outside the viewport
    dropped: int      # inside the viewport but never exposed
    placed: list      # everything ``place_props`` returned for this seed


class Choice(NamedTuple):
    layout: Layout
    counts: list      # bugs kept by each of the tries, in order
    index: int        # which try was kept
    seed: object      # that try's seed


def _winnable(planes, props, rest_eye_mm):
    return [p for p in props
            if exposure_count(planes, p["pos_mm"], p["radius_mm"],
                              rest_eye_mm[:2], rest_eye_mm[2]) > 0]


def layout_once(planes, count, radius_mm, rest_eye_mm, glass_mm, rect,
                centre_mm, seed, prefer=LEGACY_PREFER) -> Layout:
    placed = place_props(planes, int(count) * 3, float(radius_mm),
                         tuple(rest_eye_mm[:2]),
                         z_rest=float(z_rest_mm(rest_eye_mm[2])), seed=seed,
                         prefer=prefer)
    if not placed:
        return Layout([], 0, 0, [])
    kept, outside = filter_props(placed, rest_eye_mm, int(count), glass_mm,
                                 rect, centre_mm)
    winnable = _winnable(planes, kept, rest_eye_mm)
    return Layout(winnable, outside, len(kept) - len(winnable), list(placed))


def placeable_bugs(planes, count, radius_mm, rest_eye_mm, glass_mm, rect,
                   centre_mm, seed, prefer=LEGACY_PREFER):
    """``(winnable props, outside the viewport, dropped as unwinnable)``."""
    layout = layout_once(planes, count, radius_mm, rest_eye_mm, glass_mm, rect,
                         centre_mm, seed, prefer)
    return layout.kept, layout.outside, layout.dropped


def spare_spots(planes, layout, rest_eye_mm, rect, glass_mm, centre_mm) -> list:
    """Spots of the SAME placement run that were not kept as bugs: inside the
    viewport and exposed at some eye position (so a bug moved there can be
    found). The spacing rule against live bugs is checked at move time."""
    from src.game.viewport_filter import inside_viewport, project_to_design

    kept = {id(p) for p in layout.kept}
    rest = tuple(rest_eye_mm)
    inside = [p for p in layout.placed if id(p) not in kept
              and inside_viewport(project_to_design(tuple(p["pos_mm"]), rest,
                                                    glass_mm, centre_mm), rect)]
    return _winnable(planes, inside, rest)


def layout_seeds(seed, tries: int = LAYOUT_TRIES) -> list:
    """The per-try seeds, derived from the match seed."""
    return [f"{seed}:{i}" for i in range(int(tries))]


def choose_layout(planes, count, radius_mm, rest_eye_mm, glass_mm, rect,
                  centre_mm, seed, prefer=LEGACY_PREFER,
                  tries: int = LAYOUT_TRIES) -> Choice:
    """Try ``tries`` layouts, keep the one with the most bugs; ties are
    broken at random from the match seed (same seed, same answer)."""
    seeds = layout_seeds(seed, tries)
    layouts = [layout_once(planes, count, radius_mm, rest_eye_mm, glass_mm,
                           rect, centre_mm, s, prefer) for s in seeds]
    counts = [len(lay.kept) for lay in layouts]
    best = [i for i, c in enumerate(counts) if c == max(counts)]
    index = random.Random(f"layout:{seed}").choice(best)
    return Choice(layouts[index], counts, index, seeds[index])
