"""Random bug placement per match (seeded); logic-only, no GL."""

from __future__ import annotations

import unittest

import numpy as np

from src.content import hidden_props_search as S
from src.content.hidden_props import LayerPlane, _nearer, is_visible
from src.ui_album.bugs import assign_bug_sizes

REST = (-20.0, 100.0)


def _plane(z, alpha, w=0.344, h=0.258):
    return LayerPlane(alpha, z, w, h, 0.0, 0.0, z, z)


def _planes():
    far = np.full((64, 64), 255, np.uint8)
    mid = np.zeros((64, 64), np.uint8)
    mid[8:56, 8:56] = 255
    near = np.zeros((64, 64), np.uint8)
    near[:, 20:44] = 255
    return (_plane(-0.350, far), _plane(-0.200, mid), _plane(-0.080, near))


def _keys(props):
    return tuple(sorted((round(p["pos_mm"][0], 3), round(p["pos_mm"][1], 3),
                         round(p["pos_mm"][2], 3)) for p in props))


class SeededPlacementTest(unittest.TestCase):
    def test_same_seed_same_set(self):
        planes = _planes()
        a = S.place_props(planes, 12, 4.8, REST, seed=7)
        b = S.place_props(planes, 12, 4.8, REST, seed=7)
        self.assertTrue(a)
        self.assertEqual(_keys(a), _keys(b))

    def test_different_seeds_give_different_sets(self):
        planes = _planes()
        sets = {_keys(S.place_props(planes, 6, 4.8, REST, seed=s))
                for s in range(10)}
        self.assertGreaterEqual(len(sets), 8)

    def test_no_seed_is_the_old_fixed_result(self):
        planes = _planes()
        self.assertEqual(_keys(S.place_props(planes, 6, 4.8, REST)),
                         _keys(S.place_props(planes, 6, 4.8, REST, seed=None)))

    def test_jittered_bugs_still_hidden_at_rest_and_revealed_on_lean(self):
        planes = _planes()
        for seed in range(5):
            for p in S.place_props(planes, 12, 4.8, REST, seed=seed):
                near = _nearer(planes, p["pos_mm"][2] / 1000.0)
                for z in (450.0, 330.0):
                    for dx in (-S.REST_BOX_X_MM, 0.0, S.REST_BOX_X_MM):
                        for dy in (-S.REST_BOX_Y_MM, 0.0, S.REST_BOX_Y_MM):
                            eye = (REST[0] + dx, REST[1] + dy, z)
                            self.assertFalse(is_visible(
                                eye, p["pos_mm"], 4.8, near))
                self.assertGreaterEqual(abs(p["reveal_450_mm"]), S.MIN_REVEAL_MM)

    def test_shuffle_never_breaks_preference_order(self):
        def cand(x, r):
            return {"pos_mm": (x, 0.0, -100.0), "reveal_450_mm": r}
        # best key = reveal 80/90 (|r-85| = 5); 60 and 200 rank behind it
        cands = [cand(0.0, 200.0), cand(100.0, 60.0), cand(200.0, 80.0),
                 cand(300.0, -80.0), cand(400.0, 90.0)]
        import random
        for s in range(30):
            top = S._rank(cands, 1, 4.8, random.Random(s))[0]
            self.assertIn(abs(top["reveal_450_mm"]), (80.0, 90.0))
        picks = {S._rank(cands, 1, 4.8, random.Random(s))[0]["pos_mm"][0]
                 for s in range(30)}
        self.assertGreater(len(picks), 1)       # ties DO get shuffled

    def test_sizes_follow_the_match_seed(self):
        def sizes(seed):
            props = [{} for _ in range(6)]
            assign_bug_sizes(props, seed)
            return [p["size_k"] for p in props]
        self.assertEqual(sizes(5), sizes(5))
        self.assertNotEqual(sizes(5), sizes(6))


class MatchSeedTest(unittest.TestCase):
    @staticmethod
    def _seed(env):
        from types import SimpleNamespace

        from src.engine.app import HcpdApplication

        app = HcpdApplication.__new__(HcpdApplication)
        app._devtools = SimpleNamespace(env=lambda name, default="": env.get(name, default))
        return app._new_bug_seed()

    def test_env_seed_wins(self):
        self.assertEqual(self._seed({"HCPD_BUG_SEED": "42",
                                     "HCPD_ALBUM_AUTOMATCH": "1"}), 42)

    def test_automatch_default_is_fixed(self):
        from src.engine.app import AUTOMATCH_BUG_SEED

        self.assertEqual(self._seed({"HCPD_ALBUM_AUTOMATCH": "1"}),
                         AUTOMATCH_BUG_SEED)

    def test_real_play_draws_a_fresh_seed(self):
        seeds = {self._seed({}) for _ in range(20)}
        self.assertGreater(len(seeds), 1)

    def test_bad_env_seed_falls_back(self):
        self.assertIsInstance(self._seed({"HCPD_BUG_SEED": "x"}), int)


if __name__ == "__main__":
    unittest.main()
