"""Leaderboard data: every account, ranked by lifetime points."""

from src.ui_album import board


def _data(*rows):
    return {"accounts": {
        name.lower(): {"display": name, "earned_total": earned,
                       "levels_cleared": cleared}
        for name, earned, cleared in rows}}


def test_ranked_by_earned_total_ties_share_rank_and_zeros_last():
    entries = board.collect(_data(
        ("Zed", 0, []), ("Bo", 500, [1]), ("Al", 900, [1, 2]),
        ("Cy", 500, [1, 2, 3]), ("Ana", 0, [])))
    assert [(e.name, e.rank) for e in entries] == [
        ("Al", 1), ("Bo", 2), ("Cy", 2), ("Ana", 4), ("Zed", 4)]
    assert [board.points(e) for e in entries] == ["900", "500", "500", "—", "—"]


def test_guest_is_not_listed_and_spending_never_lowers_rank():
    data = _data(("Al", 900, [1]))
    data["accounts"]["guest"] = {"display": "guest", "earned_total": 5000}
    data["accounts"]["al"]["wallet"] = 0                  # spent everything
    entries = board.collect(data)
    assert [e.name for e in entries] == ["Al"] and entries[0].earned == 900


def test_badge_is_highest_level_cleared_or_empty():
    entries = board.collect(_data(("A", 10, [1, 5, 3]), ("B", 0, [])))
    assert board.badge(entries[0]) == "Level 5 clear"
    assert board.badge(entries[1]) == ""


def test_rank_of_none_when_nothing_won():
    data = _data(("A", 10, [1]), ("B", 0, []))
    assert board.rank_of(data, "a") == 1
    assert board.rank_of(data, "b") is None and board.rank_of(data, "x") is None


def test_empty_and_missing_store():
    assert board.collect(None) == [] and board.collect({"accounts": {}}) == []
