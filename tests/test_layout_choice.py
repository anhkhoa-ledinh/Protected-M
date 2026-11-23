"""Keep the fullest of eight bug layouts (logic-only)."""

from __future__ import annotations

import unittest
from unittest import mock

from src.game import bug_capacity as BC

ARGS = ((), 12, 4.8, (-20.0, 100.0, 450.0), (344.0, 194.0),
        (28.0, 28.0, 1224.0, 664.0), (0.0, 0.0))


def _fake(counts):
    """``layout_once`` stand-in: the try seeded ``<seed>:i`` keeps counts[i]."""
    def once(planes, count, radius, rest, glass, rect, centre, seed, prefer):
        n = counts[int(str(seed).rsplit(":", 1)[1])]
        return BC.Layout([{"pos_mm": (i, 0, 0)} for i in range(n)], 0, 0, [])
    return once


class ChooseLayoutTest(unittest.TestCase):
    def test_keeps_the_fullest(self):
        counts = [3, 7, 5, 12, 2, 9, 12, 4]
        with mock.patch.object(BC, "layout_once", _fake(counts)):
            c = BC.choose_layout(*ARGS, 1)
        self.assertEqual(c.counts, counts)
        self.assertIn(c.index, (3, 6))
        self.assertEqual(len(c.layout.kept), 12)

    def test_single_best_is_picked(self):
        counts = [3, 7, 5, 8, 2, 9, 1, 4]
        with mock.patch.object(BC, "layout_once", _fake(counts)):
            self.assertEqual(BC.choose_layout(*ARGS, 1).index, 5)

    def test_same_seed_same_answer_and_ties_vary_with_seed(self):
        counts = [6] * 8
        picks = set()
        with mock.patch.object(BC, "layout_once", _fake(counts)):
            for seed in range(40):
                a = BC.choose_layout(*ARGS, seed)
                self.assertEqual(a.index, BC.choose_layout(*ARGS, seed).index)
                picks.add(a.index)
        self.assertGreater(len(picks), 3)

    def test_eight_distinct_seeds_derived_from_the_match_seed(self):
        seeds = BC.layout_seeds(77)
        self.assertEqual(len(seeds), 8)
        self.assertEqual(len(set(seeds)), 8)
        self.assertEqual(seeds, BC.layout_seeds(77))
        self.assertNotEqual(seeds, BC.layout_seeds(78))

    def test_real_planes_never_worse_than_a_single_try(self):
        from tests.test_bug_placement_seed import _planes

        planes = _planes()
        args = (planes, 4, 4.8, (-20.0, 100.0, 450.0), (344.0, 194.0),
                (28.0, 28.0, 1224.0, 664.0), (0.0, 0.0))
        for seed in (1, 2, 3):
            c = BC.choose_layout(*args, seed)
            self.assertEqual(len(c.layout.kept), max(c.counts))
            self.assertEqual(len(c.counts), 8)
            again = BC.choose_layout(*args, seed)
            self.assertEqual([p["pos_mm"] for p in c.layout.kept],
                             [p["pos_mm"] for p in again.layout.kept])


if __name__ == "__main__":
    unittest.main()
