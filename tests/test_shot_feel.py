"""
TASK_game_feel_1118 Phase 4: ``src/ui_album/shot_feel.py`` — table lookup,
shake decay/cap, kick spring settling, hit marker on hit/crit/miss, crit
impact frame gating. Pure Python, no GL — these never touch
``game_bridge.fire()`` at all, matching the task's own rule that none of
this may change a shot's result.
"""

from __future__ import annotations

import math

import pytest

from src.ui_album.shot_feel import (CRIT_IMPACT_ALPHA, CRIT_IMPACT_S,
                                    KICK_REST_SCALE, SHAKE_TAU_S, WEAPON_FEEL,
                                    ShotFeel, weapon_feel)


def test_weapon_feel_table_lookup():
    precision = weapon_feel("Precision gun")
    assert precision.shake_px == 3.0
    assert precision.kick_scale == 1.35
    assert precision.tracer.lines == 1
    assert precision.tracer.muzzle_width_px == 2.5

    spread = weapon_feel("Spread gun")
    assert spread.shake_px == 4.0
    assert spread.kick_scale == 1.40
    assert spread.tracer.lines == 3
    assert spread.tracer.spread_px == 14.0

    heavy = weapon_feel("Heavy-mag gun")
    assert heavy.shake_px == 1.5
    assert heavy.shake_cap_px == 3.0
    assert heavy.kick_scale == 1.15
    assert heavy.tracer.muzzle_width_px == 1.5

    assert set(WEAPON_FEEL) == {"Precision gun", "Spread gun", "Heavy-mag gun"}


def test_unknown_weapon_falls_back_instead_of_raising():
    assert weapon_feel("Nonexistent gun") is not None


def test_shake_decays_to_about_one_over_e_after_tau():
    feel = ShotFeel(seed=1)
    feel.on_shot("Spread gun", (640.0, 360.0), hit=False, crit=False)
    start = math.hypot(*feel.shake_offset)
    assert start == pytest.approx(4.0, abs=1e-6)
    feel.step(SHAKE_TAU_S)
    after = math.hypot(*feel.shake_offset)
    assert after == pytest.approx(start * math.exp(-1.0), rel=1e-6)


def test_heavy_mag_shake_accumulates_and_caps_at_3():
    feel = ShotFeel(seed=2)
    magnitudes = []
    for _ in range(30):
        feel.on_shot("Heavy-mag gun", (640.0, 360.0), hit=False, crit=False)
        magnitudes.append(math.hypot(*feel.shake_offset))
    # Firing rapidly (no decay between shots) must never exceed the cap —
    # and with 30 shots of 1.5 px each in random directions, it climbs to
    # (and gets held at) the cap well before the end.
    assert all(m <= 3.0 + 1e-6 for m in magnitudes)
    assert max(magnitudes) == pytest.approx(3.0, abs=1e-6)


def test_precision_shake_never_exceeds_its_own_per_shot_amount():
    feel = ShotFeel(seed=3)
    for _ in range(10):
        feel.on_shot("Precision gun", (640.0, 360.0), hit=False, crit=False)
    assert math.hypot(*feel.shake_offset) <= 3.0 + 1e-6


def test_kick_returns_to_1_within_0_4s():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (640.0, 360.0), hit=False, crit=False)
    assert feel.kick_scale == pytest.approx(1.35)
    elapsed = 0.0
    while elapsed < 0.4:
        feel.step(0.002)
        elapsed += 0.002
    assert feel.kick_scale == pytest.approx(KICK_REST_SCALE, abs=0.02)


def test_kick_scale_matches_table_for_each_weapon():
    for name, feel_spec in WEAPON_FEEL.items():
        feel = ShotFeel(seed=0)
        feel.on_shot(name, (640.0, 360.0), hit=False, crit=False)
        assert feel.kick_scale == pytest.approx(feel_spec.kick_scale)


def test_hit_marker_appears_on_hit_not_on_miss():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (100.0, 200.0), hit=False, crit=False)
    assert feel.hit_markers == []
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=False)
    assert len(feel.hit_markers) == 1
    x, y, age, crit = feel.hit_markers[0]
    assert (x, y) == (100.0, 200.0)
    assert age == 0.0
    assert crit is False


def test_hit_marker_crit_flag_set_on_crit():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=True)
    assert len(feel.hit_markers) == 1
    assert feel.hit_markers[0][3] is True


def test_hit_marker_fades_out_after_its_life():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=False)
    feel.step(0.18)
    assert feel.hit_markers == []


def test_crit_impact_only_on_crit():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=False)
    assert feel.crit_impact_alpha == 0.0
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=True)
    assert feel.crit_impact_alpha == CRIT_IMPACT_ALPHA


def test_crit_impact_expires_after_its_life():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=True)
    feel.step(CRIT_IMPACT_S)
    assert feel.crit_impact_alpha == 0.0


def test_no_shake_off_switch_disables_shake_and_crit_impact():
    feel = ShotFeel(seed=0)
    feel.shake_enabled = False
    feel.on_shot("Precision gun", (100.0, 200.0), hit=True, crit=True)
    assert feel.shake_offset == (0.0, 0.0)
    assert feel.crit_impact_alpha == 0.0
    # Kick, tracer and hit marker stay on even with shake off.
    assert feel.kick_scale == pytest.approx(1.35)
    assert len(feel.tracers) == 1
    assert len(feel.hit_markers) == 1


def test_tracer_spans_muzzle_to_crosshair_for_a_single_line_weapon():
    from src.ui_album.shot_feel import MUZZLE_XY

    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (300.0, 400.0), hit=False, crit=False)
    assert len(feel.tracers) == 1
    muzzle, target, age, width = feel.tracers[0]
    assert muzzle == MUZZLE_XY
    assert target == (300.0, 400.0)
    assert age == 0.0
    assert width == 2.5


def test_spread_gun_spawns_3_tracer_lines_offset_at_the_target_end():
    feel = ShotFeel(seed=0)
    feel.on_shot("Spread gun", (300.0, 400.0), hit=False, crit=False)
    assert len(feel.tracers) == 3
    targets = [t[1] for t in feel.tracers]
    # The crosshair itself is one of the three target points; the other
    # two sit 14 px away from it, perpendicular to the shot direction.
    assert (300.0, 400.0) in targets
    import math as _math
    distances = sorted(
        _math.hypot(tx - 300.0, ty - 400.0) for tx, ty in targets)
    assert distances[0] == pytest.approx(0.0, abs=1e-9)
    assert distances[1] == pytest.approx(14.0, abs=1e-6)
    assert distances[2] == pytest.approx(14.0, abs=1e-6)


def test_tracers_expire_after_their_life():
    feel = ShotFeel(seed=0)
    feel.on_shot("Precision gun", (300.0, 400.0), hit=False, crit=False)
    feel.step(0.07)
    assert feel.tracers == []


def test_on_shot_has_no_return_value_to_feed_back_into_fire():
    """``on_shot`` only reacts to a shot already resolved by
    ``game_bridge.fire`` — it must not be able to influence it."""
    feel = ShotFeel(seed=0)
    assert feel.on_shot("Precision gun", (0.0, 0.0), hit=True, crit=True) is None


def test_shot_feel_module_never_imports_game_bridge():
    """Structural guarantee behind the above: no import cycle exists for
    ``on_shot`` to reach back into ``fire()``'s own module."""
    import src.ui_album.shot_feel as module

    assert "game_bridge" not in module.__dict__
    assert not hasattr(module, "fire")
