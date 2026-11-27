"""Levels 4-5 keep one spare hiding spot on a tight photo (logic-only)."""

from __future__ import annotations

import time
import unittest

from src.content.prop_game import GameState
from src.game import levels as L
from src.game.bug_capacity import bugs_to_place, hold_back_spares
from src.game.bug_move import BugMover
from src.game.match import AlbumMatch

R = 4.8
EYE = (-20.0, 100.0, 450.0)


def _prop(x, y=0.0, z=-174.0):
    return {"pos_mm": (x, y, z), "radius_mm": R, "reveal_axis": "x",
            "reveal_450_mm": 80.0, "reveal_330_mm": 60.0, "size_k": 1.0}


def _album(n):
    """A fake album that can hide exactly ``n`` bugs."""
    return [_prop(100.0 * i) for i in range(n)]


class FakeRng:
    def __init__(self, *rolls):
        self.rolls = list(rolls)

    def random(self):
        return self.rolls.pop(0)

    def shuffle(self, seq):
        pass


def _start(level, capacity):
    rule = L.rules(level)
    kept, held = hold_back_spares(_album(capacity), rule.bugs,
                                  rule.min_spare_spots)
    game = GameState(image="t", props=kept, tracking_on=True, started_at_s=0.0)
    match = AlbumMatch(level=level, state=game, started_s=0.0)
    mover = BugMover(game, held, rule.move_after_s or 0.0, R, seed=1,
                     flee_chance=rule.flee_chance,
                     flee_cooldown_s=rule.flee_cooldown_s)
    return game, match, mover


class Table(unittest.TestCase):
    def test_min_spare_spots_column(self):
        self.assertEqual([L.LEVELS[lv].min_spare_spots for lv in range(1, 6)],
                         [0, 0, 0, 1, 1])


class Formula(unittest.TestCase):
    def test_formula(self):
        self.assertEqual(bugs_to_place(8, 12, 1), 7)
        self.assertEqual(bugs_to_place(12, 12, 1), 11)
        self.assertEqual(bugs_to_place(30, 12, 1), 12)     # room: unchanged
        self.assertEqual(bugs_to_place(1, 12, 1), 1)       # never 0 bugs
        self.assertEqual(bugs_to_place(8, 10, 0), 8)       # level 3: unchanged

    def test_levels_1_to_3_unchanged_on_a_tight_photo(self):
        game, _m, mover = _start(3, 8)
        self.assertEqual(len(game.props), 8)
        self.assertEqual(mover.spares, [])

    def test_capacity_one_on_level_5_places_one_bug(self):
        game, _m, mover = _start(5, 1)
        self.assertEqual(len(game.props), 1)
        self.assertEqual(mover.spares, [])

    def test_the_last_in_placement_order_is_held_back(self):
        kept = _album(8)
        bugs, held = hold_back_spares(kept, 12, 1)
        self.assertEqual(bugs, kept[:7])
        self.assertEqual(held, kept[7:])


class Level5Tight(unittest.TestCase):
    def test_eight_spots_gives_seven_bugs_and_one_spare(self):
        game, match, mover = _start(5, 8)
        self.assertEqual(len(game.props), 7)
        self.assertEqual(len(mover.spares), 1)
        mover.flee_rng = FakeRng(0.1)               # "flee"
        match.apply_damage(0, 120.0, (1.0, 1.0), 1.0)
        old = game.props[0]["pos_mm"]
        moved = mover.flee([(0, False, False)], [], EYE, 1.0,
                           hp_of=lambda i: match.bug_hp[i])
        self.assertEqual(len(moved), 1)
        self.assertNotEqual(game.props[0]["pos_mm"], old)
        self.assertEqual(match.bug_hp[0], 500.0 - 120.0)

    def test_winning_needs_all_placed_bugs(self):
        game, match, _mv = _start(5, 8)
        match.started_s = time.perf_counter()
        for i in range(6):
            game.found[i] = True
        self.assertEqual(match.bugs_total, 7)
        self.assertFalse(match.tick(True))
        game.found[6] = True
        self.assertEqual(match.tick(True), "win")


class Level4Tight(unittest.TestCase):
    def test_twelve_spots_gives_eleven_bugs_and_a_timed_move(self):
        game, _m, mover = _start(4, 12)
        self.assertEqual(len(game.props), 11)
        self.assertEqual(len(mover.spares), 1)
        game.visible_since_s[0] = 0.0
        moved = mover.tick([], EYE, 4.0)            # seen for 4 s
        self.assertEqual([m[0] for m in moved], [0])


if __name__ == "__main__":
    unittest.main()
