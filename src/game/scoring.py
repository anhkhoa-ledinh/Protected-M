"""The match score formula, with no imports beyond the standard library, so
the game and the central server compute the very same number.

``max(0, seconds left x level)``; using an item costs no score; a lost match is 0;
level 5 also loses ``OVERTIME_PENALTY`` points per second past the time.
"""

from __future__ import annotations

#: Past the time at level 5: -5 points a second, a loss after this many seconds.
OVERTIME_MAX_S = 30.0
OVERTIME_PENALTY = 5.0


def match_score(level: int, lost: bool, time_left_s: float) -> int:
    if lost:
        return 0
    base = max(0.0, time_left_s) * level
    if level == 5:
        base -= OVERTIME_PENALTY * max(0.0, -time_left_s)
    return max(0, int(round(base)))
