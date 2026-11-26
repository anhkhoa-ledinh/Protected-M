"""Match score: seconds left x level; using an item costs nothing; lost = 0."""

import time
from types import SimpleNamespace

import pytest

from src.game.match import AlbumMatch
from src.game.levels import LEVELS

def _match(level, seconds_left, used=None, ended=""):
    state = SimpleNamespace(props=[1], found=[True])
    played = LEVELS[level].seconds - seconds_left
    match = AlbumMatch(level=level, state=state,
                       started_s=time.perf_counter() - played,
                       used=dict(used or {}))
    match.ended = ended
    return match


def test_level1_win_150s_left_no_items():
    assert _match(1, 150).score() == 150


def test_level5_win_60s_left_one_flare():
    assert _match(5, 60, {"Flare": 1}).score() == 60 * 5 == 300


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_won_match_with_an_item_used_scores_exactly_seconds_left_times_level(level):
    used = {"Flare": 1, "Thermal scan": 2}
    assert _match(level, 77, used).score() == _match(level, 77).score() == 77 * level


@pytest.mark.parametrize("level", [1, 2, 3, 4, 5])
def test_lost_match_scores_zero(level):
    assert _match(level, 100, {"Flare": 1}, ended="lose").score() == 0


def test_never_negative():
    assert _match(1, 1, {"Flare": 5}).score() == 1       # items cost no score
    assert _match(1, 0, {"Flare": 5}).score() == 0
    assert _match(5, -20, {"Flare": 1}).score() == 0   # overtime penalty


def test_level5_overtime_penalty_stays():
    # 10 s over: max(0, 0 x 5 - 5 x 10) = 0, never below
    assert _match(5, -10).score() == 0
