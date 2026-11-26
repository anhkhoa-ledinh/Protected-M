"""The HP bar follows a bug that moved (logic-only: the bar's data, not GL)."""

from __future__ import annotations

import unittest

from src.content.prop_game import GameState
from src.game.hit import sprite_quad
from src.game.match import HP_BAR_FADE_S, AlbumMatch

R = 4.8
REST = (0.0, 0.0, 450.0)
GLASS = (344.0, 194.0)
CENTRE = (0.0, 0.0)
OLD = (0.0, 0.0, -174.0)
NEW = (60.0, 20.0, -150.0)


def _match():
    prop = {"pos_mm": OLD, "radius_mm": R, "size_k": 1.0}
    game = GameState(image="t", props=[prop], tracking_on=True, started_at_s=0.0)
    return AlbumMatch(level=5, state=game, started_s=0.0)


def _centre(pos):
    q = sprite_quad(pos, R, 1.0, REST, GLASS, CENTRE)
    return (q.centre_x, q.centre_y)


class BarFollows(unittest.TestCase):
    def test_flee_puts_the_bar_on_the_new_spot_at_once_with_the_kept_hp(self):
        m = _match()
        ox, oy = _centre(OLD)
        m.apply_damage(0, 120.0, (ox + 3.0, oy - 2.0), 1.0)       # hit near the bug
        m.follow_bug(0, OLD, NEW, REST, GLASS, CENTRE, restamp_s=5.0)
        (x, y, frac, alpha), = m.hp_bars(5.0)
        nx, ny = _centre(NEW)
        self.assertAlmostEqual(x - nx, 3.0, places=6)
        self.assertAlmostEqual(y - ny, -2.0, places=6)
        self.assertAlmostEqual(frac, 380.0 / 500.0)
        self.assertAlmostEqual(alpha, 1.0)                        # fresh hit

    def test_flee_restarts_the_visible_time(self):
        m = _match()
        m.apply_damage(0, 120.0, _centre(OLD), 1.0)
        m.follow_bug(0, OLD, NEW, REST, GLASS, CENTRE, restamp_s=1.2)
        self.assertEqual(len(m.hp_bars(1.2 + HP_BAR_FADE_S - 0.01)), 1)
        self.assertEqual(m.hp_bars(1.2 + HP_BAR_FADE_S + 0.01), [])

    def test_timed_move_visible_bar_follows_and_keeps_its_remaining_time(self):
        m = _match()
        m.apply_damage(0, 120.0, _centre(OLD), 1.0)
        m.follow_bug(0, OLD, NEW, REST, GLASS, CENTRE)            # no restamp
        (x, y, _f, alpha), = m.hp_bars(1.5)
        nx, ny = _centre(NEW)
        self.assertAlmostEqual(x, nx, places=6)
        self.assertAlmostEqual(y, ny, places=6)
        self.assertAlmostEqual(alpha, 1.0 - 0.5 / HP_BAR_FADE_S)  # time not reset
        self.assertEqual(m.hp_bars(1.0 + HP_BAR_FADE_S + 0.01), [])

    def test_timed_move_of_a_bug_with_no_bar_makes_no_bar(self):
        m = _match()
        m.follow_bug(0, OLD, NEW, REST, GLASS, CENTRE)
        self.assertEqual(m.hp_bars(1.0), [])
        self.assertEqual(m.bug_last_hit, {})

    def test_other_bugs_bars_do_not_move(self):
        m = _match()
        m.state.props.append({"pos_mm": (100.0, 0.0, -174.0), "radius_mm": R})
        m.apply_damage(1, 50.0, (700.0, 300.0), 1.0)
        m.follow_bug(0, OLD, NEW, REST, GLASS, CENTRE, restamp_s=2.0)
        self.assertEqual(m.bug_last_hit[1], (700.0, 300.0, 1.0))


if __name__ == "__main__":
    unittest.main()
