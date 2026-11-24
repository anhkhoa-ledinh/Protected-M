"""
TASK_game_feel_fix_1118 "Measurement hygiene": ``_measurement_ready()`` —
a bench/evidence hook's measured window only starts once fx preload is
done, depth layers are built, and 2s have passed since. Fake app, fake
clock — no GL, no real bake.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.devtools import DevTools
from src.engine.app import HcpdApplication


class _FakeFxLibrary:
    def __init__(self, remaining: int) -> None:
        self.preload_remaining = remaining


def make_devtools(*, fx_enabled: bool = True, preload_remaining: int = 0,
                  waiting_for_depth: bool = False) -> DevTools:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album = SimpleNamespace(
        fx_enabled=fx_enabled, _fx_library=_FakeFxLibrary(preload_remaining))
    app._renderer = SimpleNamespace(waiting_for_depth=waiting_for_depth)
    return DevTools(app)


def test_not_ready_while_fx_preload_remaining(monkeypatch):
    dt = make_devtools(preload_remaining=3)
    monkeypatch.setattr("src.devtools.time.perf_counter", lambda: 100.0)
    assert dt._measurement_ready() is False


def test_not_ready_while_waiting_for_depth(monkeypatch):
    dt = make_devtools(waiting_for_depth=True)
    monkeypatch.setattr("src.devtools.time.perf_counter", lambda: 100.0)
    assert dt._measurement_ready() is False


def test_not_ready_until_2s_after_both_hold(monkeypatch):
    dt = make_devtools()
    clock = {"t": 100.0}
    monkeypatch.setattr("src.devtools.time.perf_counter", lambda: clock["t"])
    assert dt._measurement_ready() is False     # first check — starts the 2s timer
    clock["t"] = 101.5
    assert dt._measurement_ready() is False     # 1.5s in — not yet
    clock["t"] = 102.1
    assert dt._measurement_ready() is True       # 2.1s in — ready


def test_becoming_unready_resets_the_2s_timer(monkeypatch):
    dt = make_devtools()
    clock = {"t": 100.0}
    monkeypatch.setattr("src.devtools.time.perf_counter", lambda: clock["t"])
    dt._measurement_ready()                      # starts timer at t=100
    clock["t"] = 101.0
    dt._app.renderer.waiting_for_depth = True    # un-ready again (layer rebuild)
    assert dt._measurement_ready() is False
    dt._app.renderer.waiting_for_depth = False
    clock["t"] = 102.5                             # ready again — timer restarts here
    assert dt._measurement_ready() is False
    clock["t"] = 103.5                              # only 1.0s since the restart
    assert dt._measurement_ready() is False
    clock["t"] = 104.6                              # 2.1s since the restart
    assert dt._measurement_ready() is True


def test_fx_disabled_does_not_block_readiness(monkeypatch):
    dt = make_devtools(fx_enabled=False, preload_remaining=99)
    clock = {"t": 100.0}
    monkeypatch.setattr("src.devtools.time.perf_counter", lambda: clock["t"])
    dt._measurement_ready()
    clock["t"] = 102.1
    assert dt._measurement_ready() is True


def test_log_slowest_frame_stage_does_not_crash_on_empty(capsys):
    app = HcpdApplication.__new__(HcpdApplication)
    DevTools(app)._log_slowest_frame_stage([], [], None)    # must not raise
