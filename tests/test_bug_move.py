"""Bugs that move after being seen (logic-only)."""

from __future__ import annotations

import unittest

from src.content.prop_game import GameState
from src.game import levels as L
from src.game.bug_move import SPOT_KEYS, BugMover, too_close
from src.game.match import AlbumMatch

R = 4.8
EYE = (-20.0, 100.0, 450.0)


def _prop(x, y=0.0, z=-174.0):
    return {"pos_mm": (x, y, z), "radius_mm": R, "reveal_axis": "x",
            "reveal_450_mm": 80.0, "reveal_330_mm": 60.0, "size_k": 1.0}


def _game(xs):
    return GameState(image="t", props=[_prop(x) for x in xs], tracking_on=True,
                     started_at_s=0.0)


def _mover(game, spare_xs, after=2.5):
    return BugMover(game, [_prop(x) for x in spare_xs], after, R, seed=1)


class OldSpotIsFreed(unittest.TestCase):
    def test_timed_move_returns_the_old_spot_to_the_pool(self):
        g = _game([0.0, 100.0])
        g.visible_since_s[0] = 0.0
        m = _mover(g, [200.0])
        old = dict(g.props[0])
        m.tick((), EYE, 3.0)
        self.assertEqual(len(m.spares), 1)
        for key in SPOT_KEYS:
            self.assertEqual(m.spares[0][key], old[key])

    def test_the_bug_never_lands_on_the_spot_it_is_leaving(self):
        g = _game([0.0])
        g.visible_since_s[0] = 0.0
        m = _mover(g, [200.0])
        m.tick((), EYE, 3.0)
        self.assertEqual(g.props[0]["pos_mm"][0], 200.0)
        m.spares = []                                   # nothing but its own spot
        g.visible_since_s[0] = 3.0
        g.props[0]["pos_mm"] = (0.0, 0.0, -174.0)
        m.tick((), EYE, 9.0)                            # no spare: stays
        self.assertEqual(g.props[0]["pos_mm"][0], 0.0)
        self.assertEqual(m.spares, [])                  # and nothing was added

    def test_another_bug_can_take_the_freed_spot(self):
        g = _game([0.0, 100.0])
        g.visible_since_s[0] = 0.0
        m = _mover(g, [200.0])
        m.tick((), EYE, 3.0)                            # bug 0: 0 -> 200
        g.visible_since_s[1] = 5.0
        out = m.tick((), EYE, 8.0)                      # bug 1 takes x = 0
        self.assertEqual([(i, n[0]) for i, _o, n, _s in out], [(1, 0.0)])
        self.assertEqual([s["pos_mm"][0] for s in m.spares], [100.0])


class MoveRules(unittest.TestCase):
    def test_table_rows(self):
        self.assertEqual([L.LEVELS[lv].move_after_s for lv in range(1, 6)],
                         [None, None, None, 4.0, 2.5])

    def test_moves_only_after_the_time(self):
        g = _game([0.0, 100.0])
        g.visible_since_s[0] = 10.0
        m = _mover(g, [200.0])
        self.assertEqual(m.tick((), EYE, 12.4), [])
        out = m.tick((), EYE, 12.5)
        self.assertEqual([(i, o[0], n[0]) for i, o, n, _s in out], [(0, 0.0, 200.0)])
        self.assertAlmostEqual(out[0][3], 2.5)

    def test_move_resets_timer_and_uses_up_the_spare(self):
        g = _game([0.0])
        g.visible_since_s[0] = 0.0
        m = _mover(g, [200.0])
        m.tick((), EYE, 3.0)
        self.assertIsNone(g.visible_since_s[0])
        # the spare is used up, and the old spot took its place in the pool
        self.assertEqual([s["pos_mm"][0] for s in m.spares], [0.0])
        self.assertEqual(g.props[0]["pos_mm"][0], 200.0)

    def test_health_size_and_index_survive(self):
        g = _game([0.0])
        g.visible_since_s[0] = 0.0
        match = AlbumMatch(level=5, state=g, started_s=0.0)
        match.apply_damage(0, 120.0, (1.0, 1.0), 1.0)
        m = _mover(g, [200.0])
        m.tick((), EYE, 3.0)
        self.assertEqual(match.bug_hp[0], 380.0)
        self.assertEqual(g.props[0]["size_k"], 1.0)

    def test_spacing_against_every_other_live_bug(self):
        g = _game([0.0, 100.0])
        g.visible_since_s[0] = 0.0
        # 105 is within 3 x radius of the live bug at 100; 150 is clear.
        m = _mover(g, [105.0, 150.0])
        out = m.tick((), EYE, 3.0)
        self.assertEqual(out[0][2][0], 150.0)
        self.assertTrue(too_close((105.0, 0.0, 0.0), (100.0, 0.0, 0.0), R))

    def test_a_dead_bug_does_not_block_a_spot_and_does_not_move(self):
        g = _game([0.0, 100.0])
        g.found[1] = True
        g.visible_since_s[0] = 0.0
        g.visible_since_s[1] = 0.0
        m = _mover(g, [103.0])
        out = m.tick((), EYE, 3.0)
        self.assertEqual([i for i, _o, _n, _s in out], [0])

    def test_unseen_bug_never_moves(self):
        g = _game([0.0])
        m = _mover(g, [200.0])
        self.assertEqual(m.tick((), EYE, 1000.0), [])


if __name__ == "__main__":
    unittest.main()
