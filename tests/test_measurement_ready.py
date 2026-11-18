"""
TASK_game_feel_fix_1118 "Measurement hygiene": ``_measurement_ready()`` —
a bench/evidence hook's measured window only starts once fx preload is
done, depth layers are built, and 2s have passed since. Fake app, fake
clock — no GL, no real bake.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.engine.app import HcpdApplication


class _FakeFxLibrary:
    def __init__(self, remaining: int) -> None:
        self.preload_remaining = remaining


def make_app(*, fx_enabled: bool = True, preload_remaining: int = 0,
            waiting_for_depth: bool = False) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album = SimpleNamespace(
        fx_enabled=fx_enabled, _fx_library=_FakeFxLibrary(preload_remaining))
    app._renderer = SimpleNamespace(waiting_for_depth=waiting_for_depth)
    return app


def test_not_ready_while_fx_preload_remaining(monkeypatch):
    app = make_app(preload_remaining=3)
    monkeypatch.setattr("src.engine.app.time.perf_counter", lambda: 100.0)
    assert app._measurement_ready() is False


def test_not_ready_while_waiting_for_depth(monkeypatch):
    app = make_app(waiting_for_depth=True)
    monkeypatch.setattr("src.engine.app.time.perf_counter", lambda: 100.0)
    assert app._measurement_ready() is False


def test_not_ready_until_2s_after_both_hold(monkeypatch):
    app = make_app()
    clock = {"t": 100.0}
    monkeypatch.setattr("src.engine.app.time.perf_counter", lambda: clock["t"])
    assert app._measurement_ready() is False     # first check — starts the 2s timer
    clock["t"] = 101.5
    assert app._measurement_ready() is False     # 1.5s in — not yet
    clock["t"] = 102.1
    assert app._measurement_ready() is True       # 2.1s in — ready


def test_becoming_unready_resets_the_2s_timer(monkeypatch):
    app = make_app()
    clock = {"t": 100.0}
    monkeypatch.setattr("src.engine.app.time.perf_counter", lambda: clock["t"])
    app._measurement_ready()                      # starts timer at t=100
    clock["t"] = 101.0
    app._renderer.waiting_for_depth = True         # un-ready again (layer rebuild)
    assert app._measurement_ready() is False
    app._renderer.waiting_for_depth = False
    clock["t"] = 102.5                             # ready again — timer restarts here
    assert app._measurement_ready() is False
    clock["t"] = 103.5                              # only 1.0s since the restart
    assert app._measurement_ready() is False
    clock["t"] = 104.6                              # 2.1s since the restart
    assert app._measurement_ready() is True


def test_fx_disabled_does_not_block_readiness(monkeypatch):
    app = make_app(fx_enabled=False, preload_remaining=99)
    clock = {"t": 100.0}
    monkeypatch.setattr("src.engine.app.time.perf_counter", lambda: clock["t"])
    app._measurement_ready()
    clock["t"] = 102.1
    assert app._measurement_ready() is True


def test_log_slowest_frame_stage_does_not_crash_on_empty(capsys):
    app = HcpdApplication.__new__(HcpdApplication)
    app._log_slowest_frame_stage([], [], None)    # must not raise
