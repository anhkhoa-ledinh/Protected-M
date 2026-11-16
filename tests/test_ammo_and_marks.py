"""
Đếm đạn và vết đạn của giao diện v3.

Hai điều phải giữ bằng mọi giá:

- **Bộ asset cũ không đổi.** Nó là đường lùi cho buổi bảo vệ, nên ``fire()``
  chỉ trừ đạn khi ``ammo_enabled`` bật — và cờ ấy chỉ giao diện v3 bật.
- **Không bao giờ kẹt.** Bắn hết băng mà không có đường nạp thì ván tắc, mà
  bản vẽ không có nút nạp đạn nào; nên hết viên cuối là tự nạp sau ``RELOAD_S``.
"""

from __future__ import annotations

import time

import pytest

from src.ui_album.game_bridge import (DEFAULT_MAG, RELOAD_S, SHOT_MARKS,
                                      WEAPON_MAG, WEAPONS_SPEC, AimContext,
                                      AlbumMatch, fire)


class _Aim(AimContext):
    """
    Bối cảnh ngắm giả: trả thẳng con bọ đã hẹn trước (F1).

    Phần hình học của hộp trúng được ``tests/test_hit_box.py`` chấm riêng
    trên một sprite tổng hợp; ở đây chỉ cần biết ``fire`` trừ đạn, ghi vết và
    đánh dấu con bọ cho đúng.
    """

    def __new__(cls, answer=(-1, False)):
        one = super().__new__(cls)
        object.__setattr__(one, "eye_mm", (0.0, 0.0, 600.0))
        object.__setattr__(one, "bugs", {})
        object.__setattr__(one, "planes", ())
        object.__setattr__(one, "glass_mm", (344.0, 194.0))
        object.__setattr__(one, "centre_mm", (0.0, 0.0))
        object.__setattr__(one, "alpha_threshold", None)
        object.__setattr__(one, "is_exposed", {})
        object.__setattr__(one, "answer", answer)
        return one

    def __init__(self, answer=(-1, False)) -> None:
        pass

    def resolve(self, point):
        return self.answer


MISS = _Aim()


class _Props:
    def __init__(self, n: int = 3) -> None:
        self.props = [{"pos_mm": (0.0, 0.0, -500.0)} for _ in range(n)]
        self.found = [False] * n


def match(*, ammo: bool = True, mag: int = 4) -> AlbumMatch:
    one = AlbumMatch(level=1, state=_Props(), started_s=time.perf_counter(),
                     mag_size=mag, ammo_enabled=ammo)
    one.rounds_left = mag
    return one


def test_a_shot_spends_a_round():
    one = match(mag=4)
    fire(one, (100.0, 100.0), MISS)
    assert one.rounds == (3, 4)


def test_rounds_run_down_to_empty():
    one = match(mag=3)
    for _ in range(3):
        fire(one, (100.0, 100.0), MISS)
    assert one.rounds == (0, 3)


def test_firing_while_empty_is_refused():
    """Trả ``None`` = không bắn được, khác hẳn ``-1`` = bắn mà hụt."""
    one = match(mag=1)
    assert fire(one, (10.0, 10.0), MISS) == (-1, False)      # ban, hut
    assert fire(one, (10.0, 10.0), MISS) is None    # het dan


def test_empty_magazine_reloads_itself():
    """Không có đường nạp thì ván tắc — bản vẽ không có nút nạp đạn nào."""
    one = match(mag=1)
    fire(one, (10.0, 10.0), MISS)
    assert one.reloading(time.perf_counter())
    one.tick_reload(time.perf_counter() + RELOAD_S + 0.01)
    assert one.rounds == (1, 1)
    assert not one.reloading(time.perf_counter() + RELOAD_S + 0.01)


def test_cannot_fire_mid_reload():
    one = match(mag=1)
    fire(one, (10.0, 10.0), MISS)
    assert fire(one, (10.0, 10.0), MISS) is None


def test_album_pack_still_fires_without_limit():
    """``ammo_enabled`` tắt: bộ cũ bắn bao nhiêu phát cũng được."""
    one = match(ammo=False, mag=1)
    for _ in range(5):
        assert fire(one, (10.0, 10.0), MISS) == (-1, False)
    assert not one.reloading(time.perf_counter())


def test_a_hit_also_spends_a_round():
    """Trúng hay hụt đều tốn đạn."""
    one = match(mag=4)
    assert fire(one, (10.0, 10.0), _Aim((0, False))) == (0, False)
    assert one.rounds == (3, 4)


def test_shot_marks_record_where_and_whether_it_hit():
    one = match(mag=9)
    fire(one, (120.0, 240.0), MISS)                      # hut
    fire(one, (300.0, 100.0), _Aim((0, False)))     # trung
    assert [(s[0], s[1], s[3]) for s in one.shots] == [
        (120.0, 240.0, False), (300.0, 100.0, True)]


def test_only_the_last_few_marks_are_kept():
    one = match(mag=50)
    for i in range(SHOT_MARKS + 4):
        fire(one, (float(i), 0.0), MISS)
    assert len(one.shots) == SHOT_MARKS
    assert one.shots[-1][0] == float(SHOT_MARKS + 3)   # moi nhat o cuoi


def test_refused_shots_leave_no_mark():
    """Hết đạn thì không có vết nào mọc ra."""
    one = match(mag=1)
    fire(one, (10.0, 10.0), MISS)
    fire(one, (99.0, 99.0), MISS)
    assert len(one.shots) == 1


@pytest.mark.parametrize("weapon,size", sorted(WEAPON_MAG.items()))
def test_each_weapon_has_its_own_magazine(weapon, size):
    """Súng chính xác 'few rounds', súng băng nặng 'big magazine'."""
    assert size > 0
    assert WEAPON_MAG["Heavy-mag gun"] > WEAPON_MAG["Precision gun"]
    assert WEAPON_MAG["Precision gun"] > WEAPON_MAG["Spread gun"]
    _ = weapon, size


# ---- nhịp bắn --------------------------------------------------------
def test_cadence_blocks_a_second_shot_too_soon():
    """Bấm nhanh hơn súng thì phát thứ hai không ra."""
    one = match(mag=9)
    one.fire_interval_s = 0.5
    assert fire(one, (10.0, 10.0), MISS) == (-1, False)
    assert fire(one, (10.0, 10.0), MISS) is None
    assert one.rounds == (8, 9)        # chi tru DUNG mot vien


def test_cadence_lets_the_next_shot_through_after_the_wait():
    one = match(mag=9)
    one.fire_interval_s = 0.02
    fire(one, (10.0, 10.0), MISS)
    time.sleep(0.03)
    assert fire(one, (10.0, 10.0), MISS) == (-1, False)
    assert one.rounds == (7, 9)


def test_zero_cadence_fires_as_fast_as_you_click():
    """``fire_interval_s = 0`` là lối cũ: không chặn nhịp."""
    one = match(mag=9)
    one.fire_interval_s = 0.0
    for _ in range(3):
        assert fire(one, (10.0, 10.0), MISS) == (-1, False)
    assert one.rounds == (6, 9)


@pytest.mark.parametrize("weapon", sorted(WEAPONS_SPEC))
def test_every_weapon_declares_all_three_numbers(weapon):
    spec = WEAPONS_SPEC[weapon]
    assert spec.mag > 0 and spec.fire_s > 0.0 and spec.reload_s > 0.0


def test_the_three_guns_actually_feel_different():
    """
    Ba khẩu phải khác nhau thật, không phải cùng số với ba cái tên.

    Súng băng nặng bắn nhanh nhất và nạp lâu nhất; súng chính xác chậm nhất.
    """
    precision = WEAPONS_SPEC["Precision gun"]
    spread = WEAPONS_SPEC["Spread gun"]
    heavy = WEAPONS_SPEC["Heavy-mag gun"]
    assert heavy.fire_s < spread.fire_s < precision.fire_s
    assert heavy.reload_s > precision.reload_s
    assert len({s.mag for s in (precision, spread, heavy)}) == 3


#: Trần tỉ lệ thời gian ĐỨNG NẠP. Bộ số đầu vượt xa trần này (40 % và 43 %)
#: và đo được trên một phiên chơi thật: 59 phát ra, 45 cú bấm bị nuốt.
MAX_DOWNTIME = 0.25


@pytest.mark.parametrize("weapon", sorted(WEAPONS_SPEC))
def test_no_weapon_spends_too_long_reloading(weapon):
    """
    Không khẩu nào được đứng nạp quá 25 % thời gian.

    Đây là ràng buộc ĐO ĐƯỢC chứ không phải khẩu vị: bắn hết băng trong 2,4 s
    rồi đứng nạp 1,6 s nghĩa là cứ 5 giây có 2 giây bấm chuột không ra gì.
    """
    spec = WEAPONS_SPEC[weapon]
    firing = spec.mag * spec.fire_s
    downtime = spec.reload_s / (firing + spec.reload_s)
    assert downtime <= MAX_DOWNTIME, f"{weapon} dung nap {downtime:.0%}"


def test_reload_time_is_per_weapon():
    """Nạp lại dùng ``reload_s`` của khẩu đang cầm, không phải hằng chung."""
    one = match(mag=1)
    one.reload_s = 0.05
    fire(one, (10.0, 10.0), MISS)
    one.tick_reload(time.perf_counter() + 0.06)
    assert one.rounds == (1, 1)


def test_default_magazine_covers_an_unknown_weapon():
    assert WEAPON_MAG.get("Khong co sung nay", DEFAULT_MAG) == DEFAULT_MAG


def test_a_fresh_match_starts_loaded():
    """``__post_init__`` nạp đầy; quên thì ván mới bắt đầu với 0 viên."""
    one = AlbumMatch(level=1, state=_Props(), started_s=0.0, mag_size=7)
    assert one.rounds == (7, 7)
