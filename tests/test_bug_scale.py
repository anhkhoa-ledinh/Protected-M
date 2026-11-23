"""Bugs are 40 % of the old size, from ONE constant (BUG_SCALE).

The drawn size, the placement/exposure radius and the hit area all derive from
the same ``radius_mm``, so a bug drawn inside a gap also fits the gap and the
shot hits what is drawn.
"""

from __future__ import annotations

import unittest

from src.config.schema import ContentSettings
from src.content.prop_game import BUG_SCALE, bug_radius_mm
from src.game import hit as H
from src.ui_album.bugs import (BUG_SIZE_MAX, SIZE_FACTOR, assign_bug_sizes)


class BugScaleTests(unittest.TestCase):
    def test_radius_is_forty_percent_of_config(self):
        content = ContentSettings()
        self.assertAlmostEqual(BUG_SCALE, 0.4)
        self.assertAlmostEqual(bug_radius_mm(content),
                               0.4 * float(content.game_prop_radius_mm))

    def test_drawn_half_size_never_exceeds_placement_radius(self):
        radius = bug_radius_mm(ContentSettings())
        props = [{"pos_mm": (0.0, 0.0, -100.0), "radius_mm": radius}
                 for _ in range(200)]
        assign_bug_sizes(props, "any-seed")
        for prop in props:
            # BugPainter: quad edge = radius_mm * size_k, centred on the bug.
            half = 0.5 * prop["radius_mm"] * prop["size_k"]
            self.assertLessEqual(half, prop["radius_mm"])
        self.assertLessEqual(0.5 * SIZE_FACTOR * BUG_SIZE_MAX, 1.0)

    def test_drawn_size_is_forty_percent_of_the_old_size(self):
        old_radius = float(ContentSettings().game_prop_radius_mm)
        new_radius = bug_radius_mm(ContentSettings())
        props = [{"pos_mm": (0.0, 0.0, -100.0), "radius_mm": new_radius}]
        assign_bug_sizes(props, "s")
        k = props[0]["size_k"]
        self.assertAlmostEqual(new_radius * k, 0.4 * old_radius * k)

    def test_hit_area_is_the_drawn_quad(self):
        radius, k = bug_radius_mm(ContentSettings()), 1.0
        quad = H.sprite_quad((0.0, 0.0, -100.0), radius, k, (0.0, 0.0, 450.0))
        old = H.sprite_quad((0.0, 0.0, -100.0), radius / BUG_SCALE, k,
                            (0.0, 0.0, 450.0))
        self.assertAlmostEqual(quad.half_w / old.half_w, BUG_SCALE)


if __name__ == "__main__":
    unittest.main()
