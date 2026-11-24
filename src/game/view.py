"""What the UI may read from a match, and the only ``src.game`` module the UI
imports.

``MatchView`` is a frozen snapshot, built once per frame by
``AlbumMatch.view(now)``: the UI reads these plain values and never touches
the live match. The static tables below are re-exported so the UI has one
import path into the game rules (``tests/test_architecture.py`` enforces it).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.game.hit import SpriteMask  # noqa: F401  (UI-facing surface)
from src.game.levels import LEGACY_CAMO  # noqa: F401
from src.game.match import WEAPON_LABEL  # noqa: F401
from src.game.pvp import (ITEM_WEIGHT, ITEMS_HAVE_EFFECT, LEVEL_TABLE,  # noqa: F401
                          PVP_MODES, PVP_UNLOCK_LEVELS, item_price)
from src.game.viewport_filter import DESIGN_SIZE, project_to_design  # noqa: F401


@dataclass(frozen=True)
class MatchView:
    layer_total: int
    layer_index: int
    time_left_s: float
    eaten_pct: float
    hp: float
    stamina: float
    #: ``((x, y, fired_at_s, hit), ...)``: bullet marks.
    shots: tuple
    #: ``((x, y, hp_left, alpha), ...)``: bug health bars at this frame.
    hp_bars: tuple
    reloading: bool
    fire_interval_s: float
    next_shot_s: float
    missed_until_s: float
    weapon_name: str
    #: ``(rounds_left, magazine)``.
    rounds: tuple
    bugs_total: int
    bugs_found: int
    my_kills: int
    bot_found: int
    bot_items: int
    exposure_pct: float
    #: ``{item: times used}`` (a copy).
    used: Any
    #: ``((effect_name, seconds_left), ...)``.
    effects: tuple


def build_match_view(match, now: float) -> MatchView:
    return MatchView(
        layer_total=int(getattr(match, "layer_total", 0) or 0),
        layer_index=int(match.layer_index),
        time_left_s=float(match.time_left_s),
        eaten_pct=float(match.eaten_pct),
        hp=float(match.hp),
        stamina=float(match.stamina),
        shots=tuple(tuple(s) for s in getattr(match, "shots", ())),
        hp_bars=tuple(tuple(b) for b in match.hp_bars(now)),
        reloading=bool(match.reloading(now)),
        fire_interval_s=float(getattr(match, "fire_interval_s", 0.0) or 0.0),
        next_shot_s=float(getattr(match, "next_shot_s", 0.0)),
        missed_until_s=float(getattr(match, "missed_until_s", 0.0)),
        weapon_name=match.weapon_name,
        rounds=tuple(match.rounds),
        bugs_total=int(match.bugs_total),
        bugs_found=int(match.bugs_found),
        my_kills=int(getattr(match, "my_kills", 0)),
        bot_found=int(getattr(match, "bot_found", 0)),
        bot_items=int(getattr(match, "bot_items", 0)),
        exposure_pct=float(getattr(match, "exposure_pct", 0.0)),
        used=dict(getattr(match, "used", {}) or {}),
        effects=tuple(tuple(e) for e in (getattr(match, "effects", ()) or ())),
    )
