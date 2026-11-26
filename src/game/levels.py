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
    hp: float               # bug health (solo)
    flee_chance: float = 0.0     # a hit that does not kill sends the bug to a new spot
    #: a bug that fled cannot flee again for this long. Levels 1-4 never flee
    #: (``flee_chance`` 0), so 0.0 there is only "no gate", never a rule.
    flee_cooldown_s: float = 0.0

    @property
    def prefer_mid(self) -> float:
        return (self.prefer_lo + self.prefer_hi) / 2.0


#: Today's look, still used by PvP and as the fallback for a prop that carries
#: no per-level value (props that came over the wire).
LEGACY_PREFER = (50.0, 120.0)
LEGACY_CAMO = 0.75

LEVELS = {
    1: Level("Dễ", "Easy", 6, 180, 2, False,
             40.0, 70.0, 0.40, 1.0, None, 100.0, 0.0, 0.0),
    2: Level("Cơ bản", "Normal", 8, 150, 3, False,
             50.0, 85.0, 0.55, 1.0, None, 200.0, 0.0, 0.0),
    3: Level("Khó", "Hard", 10, 150, 3, True,
             60.0, 100.0, 0.70, 1.0, None, 300.0, 0.0, 0.0),
    4: Level("Xâm thực", "Infestation", 12, 120, 4, True,
             75.0, 110.0, 0.85, 0.875, 4.0, 400.0, 0.0, 0.0),
    5: Level("Châu Á", "Asian Mode", 12, 120, 4, True,
             90.0, 120.0, 1.00, 0.75, 2.5, 500.0, 0.5, 1.0),
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
    "Heavy-mag gun": WeaponSpec(100, 0.18, 5.0),
}
#: PvP keeps the old heavy magazine: only solo changed (PvP is one hit, one kill).
PVP_WEAPONS_SPEC = {**WEAPONS_SPEC, "Heavy-mag gun": WeaponSpec(50, 0.18, 2.5)}

#: Base damage, headshot factor, armour pierce (1.00 / 0.85 / 0.70, in the
#: order the loadout screen lists the guns: Sniper, Shotgun, Heavy-mag).
WEAPON_DAMAGE = {
    "Precision gun": DamageSpec(150.0, 2.0, 1.00),
    "Spread gun": DamageSpec(37.5, 1.0, 0.85),
    "Heavy-mag gun": DamageSpec(51.0, 1.5, 0.70),
}


def hit_damage(weapon_name: str, crit: bool, level: int) -> float | None:
    """Damage of one round on a bug at ``level``; ``None`` for a gun not in
    the table. ``base * pierce * (head factor if crit)``; it does not depend
    on the level (bug hp does)."""
    spec = WEAPON_DAMAGE.get(weapon_name)
    if spec is None:
        return None
    head = spec.headshot_mult if crit else 1.0
    return spec.damage * spec.pierce * head


def hits_to_kill(weapon_name: str, crit: bool, level: int) -> int:
    """Rounds (shotgun: pellets) that land on one bug to kill it at ``level``."""
    import math

    damage = hit_damage(weapon_name, crit, level)
    return int(math.ceil(round(rules(level).hp / damage, 9)))
