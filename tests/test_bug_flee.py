"""A hit bug may flee to a new hiding spot on level 5 (logic-only)."""

from __future__ import annotations

import math
import unittest

from src.content.prop_game import GameState
from src.game import levels as L
from src.game.bug_move import SPOT_KEYS, BugMover
from src.game.match import AlbumMatch, fire

R = 4.8
EYE = (-20.0, 100.0, 450.0)


def _prop(x, y=0.0, z=-174.0):
    return {"pos_mm": (x, y, z), "radius_mm": R, "reveal_axis": "x",
            "reveal_450_mm": 80.0, "reveal_330_mm": 60.0, "size_k": 1.0}


def _game(xs):
    return GameState(image="t", props=[_prop(x) for x in xs], tracking_on=True,
                     started_at_s=0.0)


class FakeRng:
    """Hands out the queued rolls and counts how many were asked for."""

    def __init__(self, *rolls):
        self.rolls = list(rolls)
        self.calls = 0

    def random(self):
        self.calls += 1
        return self.rolls.pop(0)

    def shuffle(self, seq):
        pass


def _setup(level, spare_xs=(200.0,), xs=(0.0,), roll=0.1, hp_hit=120.0):
    """A match on ``level`` where bug 0 has just been hit for ``hp_hit``."""
    game = _game(list(xs))
    match = AlbumMatch(level=level, state=game, started_s=0.0)
    killed = match.apply_damage(0, hp_hit, (1.0, 1.0), 1.0)
    mover = BugMover(game, [_prop(x) for x in spare_xs], 2.5, R, seed=1,
                     flee_chance=L.rules(level).flee_chance,
                     flee_cooldown_s=L.rules(level).flee_cooldown_s)
    mover.flee_rng = FakeRng(roll)
    return game, match, mover, [(0, False, killed)]


class TableField(unittest.TestCase):
    def test_flee_chance_column(self):
        self.assertEqual([L.LEVELS[lv].flee_chance for lv in range(1, 6)],
                         [0.0, 0.0, 0.0, 0.0, 0.5])


class Cooldown(unittest.TestCase):
    def _setup(self):
        game, match, mover, hits = _setup(5, roll=0.1)
        mover.flee_rng = FakeRng(0.1, 0.1, 0.1)
        return game, match, mover, hits

    def test_table_column(self):
        self.assertEqual(L.LEVELS[5].flee_cooldown_s, 1.0)
        self.assertTrue(all(L.LEVELS[lv].flee_cooldown_s == 0.0 for lv in range(1, 5)))

    def test_inside_the_cooldown_no_move_and_no_rng_call(self):
        game, match, mover, hits = self._setup()
        self.assertEqual(len(mover.flee(hits, (), EYE, 10.0)), 1)
        self.assertEqual(mover.flee_rng.calls, 1)
        spot = game.props[0]["pos_mm"]
        self.assertEqual(mover.flee(hits, (), EYE, 10.5), [])
        self.assertEqual(mover.flee_rng.calls, 1)              # not asked
        self.assertEqual(game.props[0]["pos_mm"], spot)

    def test_after_the_cooldown_it_rolls_again(self):
        game, match, mover, hits = self._setup()
        mover.flee(hits, (), EYE, 10.0)
        self.assertEqual(len(mover.flee(hits, (), EYE, 11.1)), 1)
        self.assertEqual(mover.flee_rng.calls, 2)

    def test_exactly_the_cooldown_counts_as_ready(self):
        game, match, mover, hits = self._setup()
        mover.flee(hits, (), EYE, 10.0)
        self.assertEqual(len(mover.flee(hits, (), EYE, 11.0)), 1)

    def test_a_failed_roll_does_not_start_the_cooldown(self):
        game, match, mover, hits = self._setup()
        mover.flee_rng = FakeRng(0.9, 0.1)
        self.assertEqual(mover.flee(hits, (), EYE, 10.0), [])
        self.assertEqual(len(mover.flee(hits, (), EYE, 10.1)), 1)

    def test_no_spot_stays_and_does_not_start_the_cooldown(self):
        game, match, mover, hits = _setup(5, spare_xs=(), roll=0.1)
        mover.flee_rng = FakeRng(0.1, 0.1)
        self.assertEqual(mover.flee(hits, (), EYE, 10.0), [])
        mover.spares = [_prop(200.0)]
        self.assertEqual(len(mover.flee(hits, (), EYE, 10.1)), 1)

    def test_only_the_eligible_bug_of_a_shot_moves(self):
        game = _game([0.0, 100.0])
        match = AlbumMatch(level=5, state=game, started_s=0.0)
        mover = BugMover(game, [_prop(200.0), _prop(300.0)], 2.5, R, seed=1,
                         flee_chance=0.5, flee_cooldown_s=1.0)
        mover.flee_rng = FakeRng(0.1, 0.1)
        mover.flee([(0, False, False)], (), EYE, 10.0)          # bug 0 flees
        out = mover.flee([(0, False, False), (1, False, False)], (), EYE, 10.5)
        self.assertEqual([i for i, _o, _n in out], [1])
        self.assertEqual(mover.flee_rng.calls, 2)

    def test_heavy_mag_burst_flees_at_most_once_a_second(self):
        game, match, mover, hits = _setup(5, roll=0.0)
        mover.flee_rng = FakeRng(*([0.0] * 15))               # always "flee"
        fled = 0
        for i in range(15):
            fled += len(mover.flee(hits, (), EYE, 10.0 + i * 0.18))
        self.assertEqual(fled, 3)
        self.assertLessEqual(fled, math.ceil(15 * 0.18 / 1.0))
        self.assertEqual(mover.flee_rng.calls, fled)


class FleeOnLevel5(unittest.TestCase):
    def test_roll_below_chance_moves_the_bug_and_keeps_hp(self):
        game, match, mover, hits = _setup(5, roll=0.49)
        hp_after_hit = match.bug_hp[0]
        out = mover.flee(hits, (), EYE, 5.0)
        self.assertEqual([(i, o[0], n[0]) for i, o, n in out], [(0, 0.0, 200.0)])
        self.assertEqual(game.props[0]["pos_mm"][0], 200.0)
        self.assertEqual(match.bug_hp[0], hp_after_hit)
        self.assertEqual(match.bug_hp[0], 380.0)          # 500 - 120, no heal

    def test_roll_at_or_above_chance_does_not_move(self):
        for roll in (0.5, 0.99):
            game, match, mover, hits = _setup(5, roll=roll)
            self.assertEqual(mover.flee(hits, (), EYE, 5.0), [])
            self.assertEqual(game.props[0]["pos_mm"][0], 0.0)
            self.assertEqual(len(mover.spares), 1)

    def test_killing_hit_never_rolls_and_the_bug_stays(self):
        game, match, mover, hits = _setup(5, roll=0.0, hp_hit=500.0)
        self.assertTrue(hits[0][2])
        self.assertEqual(mover.flee(hits, (), EYE, 5.0), [])
        self.assertEqual(mover.flee_rng.calls, 0)
        self.assertEqual(game.props[0]["pos_mm"][0], 0.0)

    def test_a_miss_does_not_roll(self):
        game, match, mover, _hits = _setup(5, roll=0.0)
        self.assertEqual(mover.flee([], (), EYE, 5.0), [])
        self.assertEqual(mover.flee_rng.calls, 0)

    def test_flee_resets_the_seen_timer(self):
        game, match, mover, hits = _setup(5, roll=0.1)
        game.visible_since_s[0] = 1.0
        mover.flee(hits, (), EYE, 5.0)
        self.assertIsNone(game.visible_since_s[0])

    def test_no_free_spot_stays_and_logs_it(self):
        game, match, mover, hits = _setup(5, spare_xs=(), roll=0.1)
        with self.assertLogs("src.game.bug_move", level="INFO") as cm:
            out = mover.flee(hits, (), EYE, 5.0)
        self.assertEqual(out, [])
        self.assertEqual(game.props[0]["pos_mm"][0], 0.0)
        self.assertEqual(match.bug_hp[0], 380.0)
        self.assertTrue(any("stays" in line and "#0" in line for line in cm.output))

    def test_spare_too_close_to_a_live_bug_counts_as_no_spot(self):
        game, match, mover, hits = _setup(5, spare_xs=(103.0,), xs=(0.0, 100.0))
        self.assertEqual(mover.flee(hits, (), EYE, 5.0), [])
        self.assertEqual(game.props[0]["pos_mm"][0], 0.0)


class OldSpotIsFreed(unittest.TestCase):
    def test_flee_returns_the_old_spot_to_the_pool(self):
        game, match, mover, hits = _setup(5, roll=0.1)
        old = dict(game.props[0])
        mover.flee(hits, (), EYE, 5.0)
        self.assertEqual(len(mover.spares), 1)
        for key in SPOT_KEYS:
            self.assertEqual(mover.spares[0][key], old[key])

    def test_a_later_flee_by_another_bug_can_pick_it(self):
        game = _game([0.0, 100.0])
        match = AlbumMatch(level=5, state=game, started_s=0.0)
        mover = BugMover(game, [_prop(200.0)], 2.5, R, seed=1, flee_chance=0.5)
        mover.flee_rng = FakeRng(0.1, 0.1)
        match.apply_damage(0, 120.0, (1.0, 1.0), 1.0)
        mover.flee([(0, False, False)], (), EYE, 5.0)          # 0 -> 200
        match.apply_damage(1, 120.0, (1.0, 1.0), 6.0)
        out = mover.flee([(1, False, False)], (), EYE, 6.0)    # 1 -> 0
        self.assertEqual([(i, n[0]) for i, _o, n in out], [(1, 0.0)])

    def test_the_fleeing_bug_never_lands_on_its_own_old_spot(self):
        game, match, mover, hits = _setup(5, spare_xs=(), roll=0.1)
        self.assertEqual(mover.flee(hits, (), EYE, 5.0), [])   # no spare: stays
        self.assertEqual(mover.spares, [])                     # nothing added


class NoFleeBelowLevel5(unittest.TestCase):
    def test_levels_1_to_4_never_move_on_a_hit(self):
        for level in (1, 2, 3, 4):
            game, match, mover, hits = _setup(level, roll=0.0, hp_hit=50.0)
            self.assertEqual(mover.flee(hits, (), EYE, 5.0), [], level)
            self.assertEqual(game.props[0]["pos_mm"][0], 0.0, level)


class OneRollPerShot(unittest.TestCase):
    def test_spread_blast_with_several_pellets_rolls_once(self):
        game = _game([0.0])
        match = AlbumMatch(level=5, state=game, started_s=0.0,
                           weapon_name="Spread gun")
        hits = [(0, False, match.apply_damage(0, 31.875, (1.0, 1.0), 1.0))
                for _ in range(5)]
        self.assertEqual(len(hits), 5)
        mover = BugMover(game, [_prop(200.0)], 2.5, R, seed=1, flee_chance=0.5)
        mover.flee_rng = FakeRng(0.1)
        self.assertEqual(len(mover.flee(hits, (), EYE, 5.0)), 1)
        self.assertEqual(mover.flee_rng.calls, 1)

    def test_real_spread_fire_gives_one_roll(self):
        class Aim:
            bugs = ()
            eye_mm = EYE
            planes = ()
            glass_mm = (0.0, 0.0)
            centre_mm = (0.0, 0.0)
            alpha_threshold = 0.25
            is_exposed = None

            def resolve(self, point):
                return 0, False

        import src.game.hit as hit
        game = _game([0.0])
        match = AlbumMatch(level=5, state=game, started_s=0.0,
                           weapon_name="Spread gun")
        real = hit.shot_diagnostic
        hit.shot_diagnostic = lambda *a, **k: ""
        try:
            fire(match, (640.0, 360.0), Aim())
        finally:
            hit.shot_diagnostic = real
        self.assertGreater(len(match.last_shot_hits), 1)
        mover = BugMover(game, [_prop(200.0)], 2.5, R, seed=1, flee_chance=0.5)
        mover.flee_rng = FakeRng(0.9)
        mover.flee(match.last_shot_hits, (), EYE, 5.0)
        self.assertEqual(mover.flee_rng.calls, 1)


class Log(unittest.TestCase):
    def test_one_info_line_per_flee(self):
        game, match, mover, hits = _setup(5, roll=0.1)
        with self.assertLogs("src.game.bug_move", level="INFO") as cm:
            mover.flee(hits, (), EYE, 5.0, hp_of=lambda i: match.bug_hp[i])
        lines = [m for m in cm.output if "flee" in m.lower()]
        self.assertEqual(len(lines), 1)
        self.assertIn("#0", lines[0])
        self.assertIn("380", lines[0])
        self.assertIn("200", lines[0])


class Statistics(unittest.TestCase):
    def test_flee_rate_with_the_real_seeded_rng(self):
        game = _game([0.0])
        mover = BugMover(game, [], 2.5, R, seed="stat", flee_chance=L.rules(5).flee_chance,
                         flee_cooldown_s=L.rules(5).flee_cooldown_s)
        n, fled = 10_000, 0
        for i in range(n):
            mover.spares = [_prop(200.0)]
            game.props[0]["pos_mm"] = (0.0, 0.0, -174.0)
            # one shot every 2 s: past the cooldown, so every shot is an eligible roll
            fled += bool(mover.flee([(0, False, False)], (), EYE, 5.0 + 2.0 * i))
        self.assertTrue(0.48 <= fled / n <= 0.52, fled / n)

    def test_same_seed_replays_the_same_rolls(self):
        def run():
            game = _game([0.0])
            m = BugMover(game, [], 2.5, R, seed=7, flee_chance=0.5)
            out = []
            for _ in range(50):
                m.spares = [_prop(200.0)]
                game.props[0]["pos_mm"] = (0.0, 0.0, -174.0)
                out.append(bool(m.flee([(0, False, False)], (), EYE, 5.0)))
            return out
        self.assertEqual(run(), run())


if __name__ == "__main__":
    unittest.main()
