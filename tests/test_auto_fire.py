"""
TASK_game_feel_1118 Phase 5: ``src/engine/app.py``'s ``_tick_auto_fire`` —
full-auto for the Heavy-mag gun. Pure-logic tests via
``HcpdApplication.__new__`` + a fake clock, the same pattern
``tests/test_abandon_match.py``/``tests/test_pvp_app_integration.py``
already use — ``_album_shoot`` itself (hit resolution, fx, PvP) is
replaced with a stub that just consumes a round, so these tests isolate
the NEW firing-cadence logic (readiness gating, reload resume, release,
weapon gating) from Phase 1-4's unrelated mechanics.
"""

from __future__ import annotations

from src.engine.app import HcpdApplication
from src.ui_album import v3_flow as F
from src.ui_album.game_bridge import AlbumMatch
from src.ui_album.options import AlbumOptions

#: The task's own fake spec: (mag, fire_s, reload_s).
MAG, FIRE_S, RELOAD_S = 24, 0.18, 1.3


class _FakeAlbum:
    def __init__(self, weapon_name: str, screen: str = F.PLAY) -> None:
        self.flow = F
        self.state = F.V3Flow()
        self.state.screen = screen
        self.mouse = (640.0, 360.0)
        self.mouse_down = False
        self.game = AlbumMatch(level=1, state=None, started_s=0.0,
                               weapon_name=weapon_name)
        self.game.ammo_enabled = True
        self.game.mag_size = MAG
        self.game.rounds_left = MAG
        self.game.fire_interval_s = FIRE_S
        self.game.reload_s = RELOAD_S
        self.bloom_calls: list = []

    def bump_auto_bloom(self) -> None:
        self.bloom_calls.append("bump")

    def release_auto_bloom(self) -> None:
        self.bloom_calls.append("release")


def make_app(weapon_name: str = "Heavy-mag gun", screen: str = F.PLAY) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album_options = AlbumOptions(enabled=True, ui="v3")
    app._album = _FakeAlbum(weapon_name, screen)
    app._heavy_mag_was_held = False
    app._shot_count = 0

    def fake_shoot(now: float) -> bool:
        # Stands in for ``fire()``'s own readiness gate + round-consuming
        # side effect — ``_tick_auto_fire`` already checked readiness
        # before calling this, so ``take_round`` should always succeed.
        if app._album.game.take_round(now):
            app._shot_count += 1
        return True

    app._album_shoot = fake_shoot
    return app


def _hold_for(app: HcpdApplication, seconds: float, *, dt: float = 1.0 / 60.0,
             start: float = 0.0) -> float:
    """Simulate the trigger held for ``seconds`` of fake time, ticking at
    a realistic ~60fps cadence. Returns the fake clock's end value.

    ``tick_reload`` is real gameplay's job (``AlbumMatch.tick()``, called
    every frame from ``_tick_album_match`` — but that method reads REAL
    ``time.perf_counter()``, incompatible with a fake clock), so this
    calls it directly with the fake ``now`` instead, same effect."""
    now = start
    end = start + seconds
    app._album.mouse_down = True
    while now < end:
        app._album.game.tick_reload(now)
        app._tick_auto_fire(now)
        now += dt
    return now


def test_holding_5s_fires_exactly_the_spec_implied_count():
    """mag=24, fire_s=0.18: the 24th shot lands at t=23*0.18=4.14s, well
    inside 5s; the mag is then empty and reload (1.3s) doesn't finish
    before t=5.0s, so no 25th shot happens in this window."""
    app = make_app()
    _hold_for(app, 5.0)
    assert app._shot_count == MAG


def test_holding_through_a_reload_resumes():
    app = make_app()
    now = _hold_for(app, 5.0)          # empties the mag, reload starts
    assert app._shot_count == MAG
    # Still held — once the reload (1.3s) finishes, firing must resume.
    _hold_for(app, 2.0, start=now)
    assert app._shot_count > MAG


def test_releasing_stops_firing_and_calls_release_once():
    app = make_app()
    now = _hold_for(app, 1.0)          # a handful of shots
    fired_before_release = app._shot_count
    assert fired_before_release > 0
    app._album.mouse_down = False
    app._tick_auto_fire(now)
    assert app._album.bloom_calls[-1] == "release"
    release_count = app._album.bloom_calls.count("release")
    # Holding the (now released) button state steady for a while longer
    # must not fire again, and must not call release a second time.
    still = now
    for _ in range(120):
        still += 1.0 / 60.0
        app._tick_auto_fire(still)
    assert app._shot_count == fired_before_release
    assert app._album.bloom_calls.count("release") == release_count


def test_other_guns_do_not_auto_fire():
    for weapon in ("Precision gun", "Spread gun"):
        app = make_app(weapon_name=weapon)
        _hold_for(app, 5.0)
        assert app._shot_count == 0, weapon


def test_pvp_hunt_screen_also_auto_fires():
    """The task scopes this to PLAY and PVP_HUNT both."""
    app = make_app(screen=F.PVP_HUNT)
    _hold_for(app, 1.0)
    assert app._shot_count > 0


def test_not_held_does_not_fire():
    app = make_app()
    app._album.mouse_down = False
    now = 0.0
    for _ in range(60):
        app._tick_auto_fire(now)
        now += 1.0 / 60.0
    assert app._shot_count == 0


def test_bump_auto_bloom_called_once_per_shot():
    app = make_app()
    _hold_for(app, 1.0)
    assert app._album.bloom_calls.count("bump") == app._shot_count
