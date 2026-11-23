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
