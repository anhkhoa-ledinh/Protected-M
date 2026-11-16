"""TASK_pvp_lan_1116 P3: ``src/ui_album/pvp_sync.py`` bug placement <-> wire
round-trip, including the "two different machines" case — same gap-relative
geometry, different absolute plane position/size."""

from __future__ import annotations

import numpy as np
import pytest

from src.content.hidden_props import LayerPlane
from src.ui_album import pvp_sync as sync


def _plane(z_min_m, z_max_m, quad_w_m=0.3, quad_h_m=0.2, centre=(0.0, 0.0)):
    z_m = (z_min_m + z_max_m) / 2.0
    return LayerPlane(alpha=np.zeros((4, 4), dtype=np.uint8), z_m=z_m,
                       quad_w_m=quad_w_m, quad_h_m=quad_h_m,
                       centre_x_m=centre[0], centre_y_m=centre[1],
                       z_min_m=z_min_m, z_max_m=z_max_m)


#: Far -> near, matching layer_planes()'s own ordering.
#: Gap 0 = between planes[0] (relief -0.60..-0.50) and planes[1] (-0.40..-0.30):
#:   open range -0.50 .. -0.40 m  (-500 .. -400 mm), 100 mm wide.
#: Gap 1 = between planes[1] (-0.40..-0.30) and planes[2] (-0.10..0.0):
#:   open range -0.30 .. -0.10 m  (-300 .. -100 mm), 200 mm wide.
PLANES = (
    _plane(-0.60, -0.50),
    _plane(-0.40, -0.30),
    _plane(-0.10, 0.0),
)


def test_round_trip_recovers_the_gap_and_a_close_xy():
    prop = {"pos_mm": (10.0, -5.0, -200.0), "radius_mm": 6.0,  # gap 1 midpoint
            "size_k": 1.3, "camo_rgb": (0.2, 0.4, 0.6)}
    bug = sync.bug_to_wire(0, prop, PLANES)
    assert bug is not None
    assert bug["layer"] == 1
    assert 0.0 <= bug["x"] <= 1.0 and 0.0 <= bug["y"] <= 1.0
    assert bug["z"] == pytest.approx(0.5, abs=0.02)
    assert bug["size_k"] == 1.3
    assert bug["camo_rgb"] == [0.2, 0.4, 0.6]

    back = sync.bug_from_wire(bug, PLANES, radius_mm=6.0)
    assert back is not None
    bx, by, bz = back["pos_mm"]
    assert bx == pytest.approx(10.0, abs=0.5)
    assert by == pytest.approx(-5.0, abs=0.5)
    assert -300.0 <= bz <= -100.0  # still within gap 1's z range


def test_z_is_gap_relative_so_it_survives_different_absolute_bounds():
    """Two machines bake slightly different absolute relief bounds for the
    same photo (independent bakes) — z must still land inside whichever
    machine's own gap bounds, not at the sender's raw mm."""
    prop = {"pos_mm": (0.0, 0.0, -200.0), "radius_mm": 6.0}  # gap 1 midpoint
    bug = sync.bug_to_wire(0, prop, PLANES)
    assert bug["layer"] == 1
    assert bug["z"] == pytest.approx(0.5, abs=0.02)

    other_machine_planes = (
        PLANES[0],
        _plane(-0.50, -0.30),           # same layer, different relief bounds
        _plane(-0.20, 0.0),             # gap 1 now -0.30 .. -0.20 m
    )
    back = sync.bug_from_wire(bug, other_machine_planes, radius_mm=6.0)
    lo, hi = other_machine_planes[1].z_max_m, other_machine_planes[2].z_min_m
    assert lo * 1000.0 <= back["pos_mm"][2] <= hi * 1000.0


def test_bugs_to_wire_ids_match_prop_list_index():
    props = [
        {"pos_mm": (0.0, 0.0, -450.0), "radius_mm": 6.0},   # gap 0
        {"pos_mm": (20.0, 10.0, -200.0), "radius_mm": 6.0},  # gap 1
    ]
    wire = sync.bugs_to_wire(props, PLANES)
    assert [b["id"] for b in wire] == [0, 1]
    assert [b["layer"] for b in wire] == [0, 1]


def test_bugs_from_wire_preserves_id_as_list_index():
    bugs = [
        {"id": 0, "x": 0.5, "y": 0.5, "z": 0.5, "layer": 0,
         "size_k": 1.0, "camo_rgb": [0.5, 0.5, 0.5]},
        {"id": 2, "x": 0.5, "y": 0.5, "z": 0.5, "layer": 1,
         "size_k": 1.0, "camo_rgb": [0.5, 0.5, 0.5]},
    ]
    props = sync.bugs_from_wire(bugs, PLANES, radius_mm=6.0)
    assert len(props) == 3
    assert props[1] is None  # id 1 never sent
    assert props[0] is not None and props[2] is not None


def test_bug_to_wire_none_with_fewer_than_two_planes():
    prop = {"pos_mm": (0.0, 0.0, -400.0), "radius_mm": 6.0}
    assert sync.bug_to_wire(0, prop, PLANES[:1]) is None


def test_bug_from_wire_none_for_an_unknown_layer_gap():
    bug = {"id": 0, "x": 0.5, "y": 0.5, "z": 0.5, "layer": 99,
           "size_k": 1.0, "camo_rgb": [0.5, 0.5, 0.5]}
    assert sync.bug_from_wire(bug, PLANES, radius_mm=6.0) is None


def test_xy_clamped_to_0_1_even_outside_the_anchor_quad():
    prop = {"pos_mm": (10_000.0, 10_000.0, -200.0), "radius_mm": 6.0}
    bug = sync.bug_to_wire(0, prop, PLANES)
    assert bug["x"] == 1.0 and bug["y"] == 0.0
