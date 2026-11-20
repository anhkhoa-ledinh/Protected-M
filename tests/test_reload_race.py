"""
TASK_game_feel_fix_1118 Fix 3: the reload re-trigger race, fixed at the
root — ``AlbumMatch.take_round()`` calls ``tick_reload(now)`` itself,
first thing, instead of relying on each CALLER (``_tick_auto_fire``,
a click) to do it first. Direct tests against ``AlbumMatch``, no app.py
involved — the bug and the fix both live entirely in ``game_bridge.py``.
"""

from __future__ import annotations

from src.game.match import AlbumMatch

MAG, FIRE_S, RELOAD_S = 24, 0.18, 1.3


def _reloading_match() -> AlbumMatch:
    """A match whose magazine JUST ran out — ``reload_until_s`` set, but
    ``tick_reload`` has not run since (as if this is the very next call
    after the shot that emptied the mag)."""
    match = AlbumMatch(level=1, state=None, started_s=0.0,
                       weapon_name="Heavy-mag gun")
    match.ammo_enabled = True
    match.mag_size = MAG
    match.rounds_left = 0
    match.fire_interval_s = FIRE_S
    match.reload_s = RELOAD_S
    match.reload_until_s = 10.0
    match.next_shot_s = 0.0
    return match


def test_a_click_on_the_exact_reload_deadline_frame_fires_not_re_reloads():
    """TASK_game_feel_fix_1118 Fix 3's own scenario: ``now`` is EXACTLY
    the frame the reload deadline passes — before this fix,
    ``reloading(now)`` already reads False (deadline passed) but
    ``rounds_left`` is still 0 (nothing refilled it yet), so
    ``take_round`` used to hit the "out of rounds" branch and re-arm
    ANOTHER full reload right there, as if the player's click had never
    landed. A single ``take_round`` call must now refill AND fire in the
    same call."""
    match = _reloading_match()
    now = match.reload_until_s           # deadline passes on THIS exact call
    assert match.reloading(now) is False  # the race's precondition
    assert match.rounds_left == 0         # not yet refilled

    fired = match.take_round(now)

    assert fired is True
    assert match.rounds_left == MAG - 1
    assert match.reload_until_s == 0.0    # not re-armed
    assert match.next_shot_s == now + FIRE_S


def test_a_click_one_frame_before_the_deadline_still_reloads():
    """Sanity check the fix didn't just always refill — one tick before
    the deadline, it's still reloading, as before."""
    match = _reloading_match()
    now = match.reload_until_s - 0.001
    assert match.take_round(now) is False
    assert match.reloading(now) is True
    assert match.rounds_left == 0


def test_held_auto_fire_across_the_reload_deadline_resumes_without_rearming():
    """The full-auto regression from TASK_game_feel_1118 P7, re-proved at
    the ``AlbumMatch`` level (``tests/test_auto_fire.py`` already proves
    it through the whole ``_tick_auto_fire`` path — kept, per Fix 3's own
    instruction)."""
    match = _reloading_match()
    # Simulate several consecutive auto-fire frames straddling the deadline.
    now = 9.9
    assert match.take_round(now) is False      # still reloading
    now = 10.0                                  # deadline passes this frame
    assert match.take_round(now) is True       # fires immediately, no re-arm
    now = 10.0 + FIRE_S
    assert match.take_round(now) is True       # cadence continues normally
    assert match.reload_until_s == 0.0
