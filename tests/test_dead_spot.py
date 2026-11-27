"""A dead bug gives its hiding spot back to the spare pool (logic-only)."""

from __future__ import annotations

import unittest

from src.content.prop_game import GameState
from src.game import levels as L
from src.game.bug_move import DEATH_EFFECT_S, SPOT_KEYS, BugMover
from src.game.match import AlbumMatch

R = 4.8
EYE = (-20.0, 100.0, 450.0)


def _prop(x, y=0.0, z=-174.0):
    return {"pos_mm": (x, y, z), "radius_mm": R, "reveal_axis": "x",
            "reveal_450_mm": 80.0, "reveal_330_mm": 60.0, "size_k": 1.0}


class FakeRng:
    def __init__(self, *rolls):
        self.rolls = list(rolls)

    def random(self):
        return self.rolls.pop(0)

    def shuffle(self, seq):
        pass


def _spot(prop):
    return {k: prop[k] for k in SPOT_KEYS if k in prop}


def _setup(xs=(0.0, 100.0), spares=()):
    game = GameState(image="t", props=[_prop(x) for x in xs],
                     tracking_on=True, started_at_s=0.0)
    match = AlbumMatch(level=5, state=game, started_s=0.0)
    mover = BugMover(game, list(spares), 2.5, R, seed=1,
                     flee_chance=0.5, flee_cooldown_s=1.0)
    mover.flee_rng = FakeRng(0.1)
    return game, match, mover


def _kill(game, match, index, now):
    assert match.apply_damage(index, 1e6, (0.0, 0.0), now)
    game.found[index] = True


class DeadSpot(unittest.TestCase):
    def test_dead_bug_spot_lets_a_wounded_bug_flee(self):
        game, match, mover = _setup()           # empty pool from the start
        old_a = _spot(game.props[0])
        _kill(game, match, 0, 1.0)
        mover.tick([], EYE, 1.0)                # the death is noticed
        later = 1.0 + DEATH_EFFECT_S + 0.1
        match.apply_damage(1, 120.0, (1.0, 1.0), later)
        moved = mover.flee([(1, False, False)], [], EYE, later,
                           hp_of=lambda i: match.bug_hp[i])
        self.assertEqual(len(moved), 1)
        self.assertEqual(game.props[1]["pos_mm"], old_a["pos_mm"])
        self.assertEqual(match.bug_hp[1], 500.0 - 120.0)

    def test_spot_is_held_back_while_the_death_effect_draws(self):
        game, match, mover = _setup()
        _kill(game, match, 0, 1.0)
        mover.tick([], EYE, 1.0)
        mover.tick([], EYE, 1.0 + DEATH_EFFECT_S - 0.1)
        self.assertEqual(mover.spares, [])
        mover.tick([], EYE, 1.0 + DEATH_EFFECT_S)
        self.assertEqual(len(mover.spares), 1)

    def test_spot_already_in_pool_is_not_added_twice(self):
        game, match, mover = _setup(spares=[_prop(0.0)])
        _kill(game, match, 0, 1.0)
        mover.tick([], EYE, 1.0)
        mover.tick([], EYE, 10.0)
        mover.tick([], EYE, 20.0)
        self.assertEqual(len(mover.spares), 1)

    def test_a_death_is_freed_once(self):
        game, match, mover = _setup(xs=(0.0, 100.0, 200.0))
        _kill(game, match, 0, 1.0)
        for t in (1.0, 5.0, 6.0, 7.0):
            mover.tick([], EYE, t)
        self.assertEqual(len(mover.spares), 1)

    def test_effect_length_matches_the_painter(self):
        from src.ui_album import bugs
        self.assertAlmostEqual(DEATH_EFFECT_S, bugs.FADE_S + bugs.DUST_S)


if __name__ == "__main__":
    unittest.main()
