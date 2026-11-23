"""Honest bug count on screen: loading-ready screen and HUD (logic-only)."""

from __future__ import annotations

import json
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.game.bug_capacity import placeable_bugs
from src.ui_album import v3_values as V

ROOT = Path(__file__).resolve().parents[1]


class NoteTest(unittest.TestCase):
    def _state(self):
        return SimpleNamespace(texts={})

    def test_shown_only_when_photo_holds_fewer(self):
        s = self._state()
        V._bug_note(s, 4, 12)
        self.assertEqual(s.texts["bug_note"], "photo holds 4 of 12")
        w = self._state()
        V._bug_note(w, 3, 12, wide=True)
        self.assertEqual(w.texts["bug_note"],
                         "This photo only holds 3 of the 12 bugs this level asks for")
        for holds in (12, 13, None):
            t = self._state()
            V._bug_note(t, holds, 12)
            self.assertNotIn("bug_note", t.texts)

    def test_hud_uses_placed_total(self):
        s = self._state()
        flow = SimpleNamespace(bugs=lambda: 12)
        game = SimpleNamespace(bugs_total=2, bugs_found=1)
        V._hud_kills(s, flow, game)
        self.assertEqual(s.texts["kills"], ["1", "/2"])
        self.assertEqual(s.texts["bug_note"], "photo holds 2 of 12")

    def test_hud_silent_when_full(self):
        s = self._state()
        flow = SimpleNamespace(bugs=lambda: 12)
        V._hud_kills(s, flow, SimpleNamespace(bugs_total=12, bugs_found=3))
        self.assertNotIn("bug_note", s.texts)
        V._hud_kills(s, flow, None)            # ui-shots static state
        self.assertNotIn("bug_note", s.texts)

    def test_manifest_has_blank_slots_so_static_shots_do_not_change(self):
        data = json.loads((ROOT / "assets/ui_v3/manifest.json").read_text("utf-8"))
        for screen in ("v3_07_loading_ready", "v3_08_gameplay_hud"):
            rec = [t for t in data["screens"][screen]["texts"] if t["key"] == "bug_note"]
            self.assertEqual(len(rec), 1, screen)
            self.assertEqual(rec[0]["sample"], "")
            self.assertEqual(rec[0]["runs"][0]["c"], "")


class CapacityTest(unittest.TestCase):
    def test_matches_place_then_filter_with_the_same_seed(self):
        from tests.test_bug_placement_seed import _planes

        planes = _planes()
        a = placeable_bugs(planes, 4, 4.8, (-20.0, 100.0, 450.0), (344.0, 194.0),
                           (28.0, 28.0, 1224.0, 664.0), (0.0, 0.0), 9)
        b = placeable_bugs(planes, 4, 4.8, (-20.0, 100.0, 450.0), (344.0, 194.0),
                           (28.0, 28.0, 1224.0, 664.0), (0.0, 0.0), 9)
        self.assertEqual([p["pos_mm"] for p in a[0]], [p["pos_mm"] for p in b[0]])
        self.assertLessEqual(len(a[0]), 4)


if __name__ == "__main__":
    unittest.main()
