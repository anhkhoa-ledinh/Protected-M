"""A3: R reloads to a full magazine any time (not only when empty)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_weapons_1119 import _match  # noqa: E402


def test_r_on_a_full_magazine_does_nothing():
    match = _match(1, "Precision gun", mag=12)
    assert match.start_reload(1.0) is False
    assert not match.reloading(1.0)


def test_r_in_the_middle_refills_after_reload_s():
    match = _match(1, "Precision gun", mag=12)
    for index in range(4):
        assert match.take_round(1.0 + index)
    assert match.rounds_left == 8
    assert match.start_reload(10.0) is True
    assert match.reloading(10.0 + match.reload_s - 0.01)
    assert match.rounds_left == 8                      # not refilled yet
    match.tick_reload(10.0 + match.reload_s + 0.01)
    assert match.rounds_left == match.mag_size
    assert not match.reloading(10.0 + match.reload_s + 0.01)


def test_second_r_during_a_reload_does_not_restart_it():
    match = _match(1, "Precision gun", mag=12)
    match.take_round(1.0)
    assert match.start_reload(5.0)
    until = match.reload_until_s
    assert match.start_reload(6.0) is False
    assert match.reload_until_s == until


def test_shooting_during_a_reload_is_still_blocked():
    match = _match(1, "Precision gun", mag=12)
    match.take_round(1.0)
    match.start_reload(5.0)
    assert match.take_round(5.0 + match.reload_s / 2) is False
    assert match.rounds_left == match.mag_size - 1
