"""
TASK_weapons_1119: Sniper rifle / Shotgun display names, shotgun pellets,
bug HP, damage table, Heavy-mag 50 rounds.

Pure-logic tests against ``game_bridge`` (no GL) — same ``_Aim``-stub
pattern ``tests/test_ammo_and_marks.py`` already uses for ``fire()``.
"""

from __future__ import annotations

import math
import random
import time
from pathlib import Path
from types import SimpleNamespace

from src.ui_album import v3_flow as F
from src.ui_album.game_bridge import (BUG_HP_BY_LEVEL, HP_BAR_FADE_S,
                                      PELLET_COUNT, PELLET_RADIUS_PX,
                                      WEAPON_DAMAGE, WEAPON_LABEL,
                                      WEAPONS_SPEC, AimContext, AlbumMatch,
                                      fire, pellet_points, weapon_damage)
from src.ui_album.v3 import V3Manifest, V3State
from src.ui_album.v3_values import _hud_weapon

ROOT = Path(".").resolve()


class _FixedAim(AimContext):
    """Bối cảnh ngắm giả: trả cố định MỘT con bọ cho MỌI điểm ngắm — hình
    học của hộp trúng có bộ test riêng (``tests/test_hit_box.py``); ở đây
    chỉ cần biết ``fire`` trừ máu/đếm giết đúng."""

    def __new__(cls, index: int = 0, crit: bool = False):
        one = super().__new__(cls)
        object.__setattr__(one, "eye_mm", (0.0, 0.0, 600.0))
        object.__setattr__(one, "bugs", {})
        object.__setattr__(one, "planes", ())
        object.__setattr__(one, "glass_mm", (344.0, 194.0))
        object.__setattr__(one, "centre_mm", (0.0, 0.0))
        object.__setattr__(one, "alpha_threshold", None)
        object.__setattr__(one, "is_exposed", {})
        object.__setattr__(one, "answer", (index, crit))
        return one

    def __init__(self, index: int = 0, crit: bool = False) -> None:
        pass

    def resolve(self, point):
        return self.answer


class _Props:
    def __init__(self, n: int = 3) -> None:
        self.props = [{"pos_mm": (0.0, 0.0, -500.0)} for _ in range(n)]
        self.found = [False] * n


def _match(level: int, weapon: str, *, mag: int = 50) -> AlbumMatch:
    one = AlbumMatch(level=level, state=_Props(), started_s=time.perf_counter(),
                     weapon_name=weapon, mag_size=mag, ammo_enabled=True)
    one.rounds_left = mag
    return one


# ---- tên hiển thị -----------------------------------------------------

def test_weapon_label_maps_ids_to_the_shown_names():
    assert WEAPON_LABEL["Precision gun"] == "Sniper rifle"
    assert WEAPON_LABEL["Spread gun"] == "Shotgun"
    assert WEAPON_LABEL["Heavy-mag gun"] == "Heavy-mag gun"


def test_hud_weapon_name_shows_the_display_label_not_the_id():
    manifest = V3Manifest(ROOT)
    state = V3State()
    flow = SimpleNamespace(weapon_name=lambda: "Precision gun")
    _hud_weapon(state, flow, None, manifest, F.PLAY)
    assert state.texts["weapon_name"] == "SNIPER RIFLE"


# ---- ghém shotgun -------------------------------------------------------

def test_pellets_are_five_and_all_fall_inside_the_40px_disc():
    centre = (640.0, 360.0)
    points = pellet_points(centre, rng=random.Random(42))
    assert len(points) == PELLET_COUNT == 5
    assert points[0] == centre          # tâm luôn có một viên
    for x, y in points:
        assert math.hypot(x - centre[0], y - centre[1]) < PELLET_RADIUS_PX


def test_pellets_spread_is_seeded_and_reproducible():
    centre = (100.0, 50.0)
    first = pellet_points(centre, rng=random.Random(7))
    second = pellet_points(centre, rng=random.Random(7))
    assert first == second


# ---- bảng sát thương / máu bọ -------------------------------------------

def test_damage_table_matches_the_spec():
    assert weapon_damage("Precision gun", False) == 100.0
    assert weapon_damage("Precision gun", True) == 200.0
    assert weapon_damage("Spread gun", False) == 25.0
    assert weapon_damage("Spread gun", True) == 25.0      # khong thuong dau
    assert weapon_damage("Heavy-mag gun", False) == 34.0
    assert weapon_damage("Heavy-mag gun", True) == 51.0


def test_every_weapon_has_a_damage_entry():
    assert set(WEAPON_DAMAGE) == set(WEAPONS_SPEC)


def test_bug_hp_by_level():
    assert BUG_HP_BY_LEVEL == {1: 100.0, 2: 100.0, 3: 150.0, 4: 200.0, 5: 200.0}


# ---- ngưỡng giết (qua fire() thật) --------------------------------------

def test_sniper_body_shot_kills_at_level_1_not_at_level_3():
    hit = _FixedAim(0, False)
    one = _match(1, "Precision gun")
    assert fire(one, (10.0, 10.0), hit) == (0, False)
    assert one.state.found[0] is True

    three = _match(3, "Precision gun")
    fire(three, (10.0, 10.0), hit)
    assert three.state.found[0] is False


def test_sniper_headshot_kills_at_level_5():
    headshot = _FixedAim(0, True)
    five = _match(5, "Precision gun")
    fire(five, (10.0, 10.0), headshot)
    assert five.state.found[0] is True


def test_heavy_mag_three_body_hits_kill_at_level_1():
    hit = _FixedAim(0, False)
    one = _match(1, "Heavy-mag gun")
    fire(one, (10.0, 10.0), hit)
    fire(one, (10.0, 10.0), hit)
    assert one.state.found[0] is False      # 2 phat chua ha
    fire(one, (10.0, 10.0), hit)
    assert one.state.found[0] is True       # phat thu 3 ha


def test_shotgun_five_pellets_on_one_bug_kill_at_level_1():
    hit = _FixedAim(0, False)        # mọi ghém cùng rơi vào con #0
    one = _match(1, "Spread gun")
    fire(one, (640.0, 360.0), hit)
    assert one.state.found[0] is True


# ---- thanh máu nổi ------------------------------------------------------

def test_hp_bar_shows_on_a_non_lethal_hit_and_fades_after_1_5s():
    hit = _FixedAim(0, False)
    three = _match(3, "Precision gun")   # 150 hp, sat thuong 100 -> con song
    fire(three, (12.0, 34.0), hit)
    assert three.state.found[0] is False

    bars = three.hp_bars(three.shot_at_s + 0.1)
    assert len(bars) == 1
    x, y, frac, alpha = bars[0]
    assert (x, y) == (12.0, 34.0)
    assert 0.0 < frac < 1.0
    assert alpha > 0.0

    assert three.hp_bars(three.shot_at_s + HP_BAR_FADE_S + 0.01) == []


def test_hp_bar_does_not_show_for_a_kill():
    hit = _FixedAim(0, False)
    one = _match(1, "Precision gun")     # 100 hp, sat thuong 100 -> chet
    fire(one, (1.0, 2.0), hit)
    assert one.state.found[0] is True
    assert one.hp_bars(one.shot_at_s + 0.1) == []


# ---- PvP không đổi (một phát hạ) ----------------------------------------

def test_pvp_keeps_one_hit_kills_even_where_solo_hp_would_survive():
    hit = _FixedAim(0, False)
    five = _match(5, "Heavy-mag gun")     # 200 hp; 34 dmg thuong can 6 phat
    fire(five, (1.0, 1.0), hit, pvp=True)
    assert five.state.found[0] is True


def test_solo_does_not_one_hit_kill_the_same_setup():
    hit = _FixedAim(0, False)
    five = _match(5, "Heavy-mag gun")
    fire(five, (1.0, 1.0), hit, pvp=False)
    assert five.state.found[0] is False


# ---- Heavy-mag 50/2.5s ---------------------------------------------------

def test_heavy_mag_spec_is_50_rounds_2_5s_reload_same_cadence():
    spec = WEAPONS_SPEC["Heavy-mag gun"]
    assert spec.mag == 50
    assert spec.reload_s == 2.5
    assert spec.fire_s == 0.18


def test_heavy_mag_fires_50_rounds_then_reloads_for_2_5s():
    spec = WEAPONS_SPEC["Heavy-mag gun"]
    one = AlbumMatch(level=1, state=_Props(), started_s=0.0,
                     weapon_name="Heavy-mag gun", mag_size=spec.mag,
                     ammo_enabled=True, fire_interval_s=0.0,
                     reload_s=spec.reload_s)
    one.rounds_left = spec.mag
    miss = _FixedAim(-1, False)
    for _ in range(spec.mag):
        fire(one, (0.0, 0.0), miss)
    assert one.rounds == (0, spec.mag)
    assert one.reloading(time.perf_counter())
    assert fire(one, (0.0, 0.0), miss) is None     # dang nap, khong ra dan

    one.tick_reload(time.perf_counter() + spec.reload_s + 0.01)
    assert one.rounds == (spec.mag, spec.mag)
    assert not one.reloading(time.perf_counter() + spec.reload_s + 0.01)
