"""The level table holds the user's numbers, and readers follow it (logic-only)."""

from __future__ import annotations

import random
import unittest

from src.game import levels as L
from src.ui_album.bugs import (BUG_SIZE_MAX, BUG_SIZE_MIN, SIZE_FACTOR,
                               assign_bug_camo, assign_bug_sizes)

import numpy as np


class TableValues(unittest.TestCase):
    def col(self, name):
        return [getattr(L.LEVELS[lv], name) for lv in range(1, 6)]

    def test_count_and_seconds_unchanged(self):
        self.assertEqual(self.col("bugs"), [6, 8, 10, 12, 12])
        self.assertEqual(self.col("seconds"), [180, 150, 150, 120, 120])
        self.assertEqual(self.col("slots"), [2, 3, 3, 4, 4])
        self.assertEqual(self.col("counter"), [False, False, True, True, True])

    def test_lean_camo_size(self):
        self.assertEqual([(r.prefer_lo, r.prefer_hi) for r in L.LEVELS.values()],
                         [(40, 70), (50, 85), (60, 100), (75, 110), (90, 120)])
        self.assertEqual(self.col("camo"), [0.40, 0.55, 0.70, 0.85, 1.00])
        self.assertEqual(self.col("size"), [1.0, 1.0, 1.0, 0.875, 0.75])

    def test_prefer_mid_is_the_middle(self):
        self.assertEqual(L.LEVELS[1].prefer_mid, 55.0)
        self.assertEqual(L.LEVELS[5].prefer_mid, 105.0)


class HealthAndDamage(unittest.TestCase):
    def test_health_is_100_times_level_and_damage_has_no_divisor(self):
        self.assertEqual([L.LEVELS[lv].hp for lv in range(1, 6)],
                         [100, 200, 300, 400, 500])
        self.assertNotIn("damage_div", L.Level._fields)

    def test_pierce_follows_the_loadout_order(self):
        from src.ui_album.v3_flow import WEAPONS

        self.assertEqual([L.WEAPON_DAMAGE[w].pierce for w in WEAPONS],
                         [1.00, 0.85, 0.70])
        self.assertEqual([L.WEAPON_DAMAGE[w].damage for w in WEAPONS],
                         [150.0, 37.5, 51.0])
        self.assertEqual([L.WEAPON_DAMAGE[w].headshot_mult for w in WEAPONS],
                         [2.0, 1.0, 1.5])

    def test_heavy_mag_solo_and_pvp(self):
        self.assertEqual(L.WEAPONS_SPEC["Heavy-mag gun"], (100, 0.18, 5.0))
        self.assertEqual(L.PVP_WEAPONS_SPEC["Heavy-mag gun"], (50, 0.18, 2.5))

    def test_shots_to_kill_three_cells_by_hand(self):
        # Sniper, body, level 5: 150 * 1.00 = 150 a hit; 500 / 150 = 3.3 -> 4.
        self.assertEqual(L.hits_to_kill("Precision gun", False, 5), 4)
        # Heavy-mag, head, level 4: 51 * 1.5 * 0.70 = 53.55; 400 / 53.55 = 7.5 -> 8.
        self.assertEqual(L.hits_to_kill("Heavy-mag gun", True, 4), 8)
        # Shotgun pellet, level 2: 37.5 * 0.85 = 31.875; 200 / 31.875 = 6.3 -> 7.
        self.assertEqual(L.hits_to_kill("Spread gun", False, 2), 7)

    def test_hits_to_kill_table(self):
        """Damage does not shrink with the level; bug hp is 100 x level."""
        guns = ("Precision gun", "Spread gun", "Heavy-mag gun")
        body = {1: (1, 4, 3), 2: (2, 7, 6), 3: (2, 10, 9),
                4: (3, 13, 12), 5: (4, 16, 15)}
        head = {1: (1, 4, 2), 2: (1, 7, 4), 3: (1, 10, 6),
                4: (2, 13, 8), 5: (2, 16, 10)}
        for level in range(1, 6):
            self.assertEqual(
                tuple(L.hits_to_kill(g, False, level) for g in guns), body[level],
                f"body, level {level}")
            self.assertEqual(
                tuple(L.hits_to_kill(g, True, level) for g in guns), head[level],
                f"head, level {level}")


class ReadersFollowTheTable(unittest.TestCase):
    def test_size_scales_by_the_level_row(self):
        for lv in (1, 4, 5):
            props = [{} for _ in range(6)]
            assign_bug_sizes(props, "s", L.rules(lv).size)
            base = [{} for _ in range(6)]
            assign_bug_sizes(base, "s")
            for p, b in zip(props, base):
                self.assertAlmostEqual(p["size_k"], b["size_k"] * L.rules(lv).size)
                self.assertGreaterEqual(b["size_k"], BUG_SIZE_MIN * SIZE_FACTOR)
                self.assertLessEqual(b["size_k"], BUG_SIZE_MAX * SIZE_FACTOR)

    def test_camo_strength_goes_on_each_prop(self):
        img = np.full((72, 128, 3), 120, np.uint8)
        props = [{"pos_mm": (0.0, 0.0, -100.0)}, {"pos_mm": (10.0, 5.0, -150.0)}]
        assign_bug_camo(props, img, (0.0, 100.0, 450.0), strength=L.rules(5).camo)
        self.assertEqual([p["camo_strength"] for p in props], [1.0, 1.0])

    def test_prefer_orders_but_never_filters(self):
        from src.content.hidden_props_search import _rank

        rng = random.Random(3)
        cands = [{"pos_mm": (i * 100.0, 0.0, -100.0),
                  "reveal_450_mm": rng.choice((-1, 1)) * rng.uniform(40, 130)}
                 for i in range(30)]
        for lv in (1, 5):
            r = L.rules(lv)
            out = _rank(cands, 30, 4.8, None, (r.prefer_lo, r.prefer_hi))
            self.assertEqual(len(out), 30)      # nothing dropped by the preference
            first = abs(out[0]["reveal_450_mm"])
            inside = [c for c in cands
                      if r.prefer_lo <= abs(c["reveal_450_mm"]) <= r.prefer_hi]
            best = min(inside, key=lambda c: abs(abs(c["reveal_450_mm"]) - r.prefer_mid))
            self.assertEqual(first, abs(best["reveal_450_mm"]))

    def test_pvp_keeps_the_old_look(self):
        self.assertEqual(L.LEGACY_PREFER, (50.0, 120.0))
        self.assertEqual(L.LEGACY_CAMO, 0.75)


if __name__ == "__main__":
    unittest.main()
