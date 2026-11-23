"""The one table of level rules: bug count, time, look, health, damage.

Everything that differs by level lives here, as numbers. To rebalance the
game edit this file only; no other module keeps its own copy.

Solo uses all of it. PvP keeps its own fixed behaviour (one hit kills, the
old look), so it reads only the count / seconds / slots / counter columns and
the ``LEGACY_*`` constants below.
"""

from __future__ import annotations

from typing import NamedTuple


class Level(NamedTuple):
    name_vi: str            # name in the old v2 album UI
    name_en: str            # name in the v3 UI and in PvP
    bugs: int               # bugs asked for
    seconds: int
    slots: int              # item slots
    counter: bool           # the bugs hit back (the old "phản công" flag)
    prefer_lo: float        # preferred lean to reveal, mm (a preference only)
    prefer_hi: float
    camo: float             # camouflage strength, 0..1
    size: float             # bug size, x the base size
    move_after_s: float | None   # a bug seen this long moves; None = never
    hp: float               # bug health
    damage_div: float       # damage per hit = base * pierce * head / damage_div

    @property
    def prefer_mid(self) -> float:
        return (self.prefer_lo + self.prefer_hi) / 2.0


#: Today's look, still used by PvP and as the fallback for a prop that carries
#: no per-level value (props that came over the wire).
LEGACY_PREFER = (50.0, 120.0)
LEGACY_CAMO = 0.75

LEVELS = {
    1: Level("Dễ", "Easy", 6, 180, 2, False,
             50.0, 120.0, 0.75, 1.0, None, 100.0, 1.0),
    2: Level("Cơ bản", "Normal", 8, 150, 3, False,
             50.0, 120.0, 0.75, 1.0, None, 100.0, 1.0),
    3: Level("Khó", "Hard", 10, 150, 3, True,
             50.0, 120.0, 0.75, 1.0, None, 150.0, 1.0),
    4: Level("Xâm thực", "Infestation", 12, 120, 4, True,
             50.0, 120.0, 0.75, 1.0, None, 200.0, 1.0),
    5: Level("Châu Á", "Asian Mode", 12, 120, 4, True,
             50.0, 120.0, 0.75, 1.0, None, 200.0, 1.0),
}


def rules(level: int) -> Level:
    """The row for ``level``; an unknown level falls back to level 1."""
    return LEVELS.get(level, LEVELS[1])


class WeaponSpec(NamedTuple):
    """Magazine, fire interval and reload time of one gun."""

    mag: int
    fire_s: float        # minimum seconds between two shots
    reload_s: float


class DamageSpec(NamedTuple):
    """Damage of one round (one pellet for the shotgun), headshot factor,
    and the armour pierce factor against bugs."""

    damage: float
    headshot_mult: float
    pierce: float = 1.0


#: Keyed by the internal gun id, in the order the loadout screen shows them.
WEAPONS_SPEC = {
    "Precision gun": WeaponSpec(12, 0.50, 1.1),
    "Spread gun": WeaponSpec(8, 0.38, 0.9),
    "Heavy-mag gun": WeaponSpec(50, 0.18, 2.5),
}
WEAPON_DAMAGE = {
    "Precision gun": DamageSpec(100.0, 2.0),
    "Spread gun": DamageSpec(25.0, 1.0),
    "Heavy-mag gun": DamageSpec(34.0, 1.5),
}
