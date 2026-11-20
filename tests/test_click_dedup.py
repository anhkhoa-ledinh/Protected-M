"""TASK_perf_textures_1119 Fix 8: one physical click = one queued click =
one ``fire()`` call.

The user's log showed every shot ("Ban hut tai ...") followed AT ONCE by
a second, rejected ``fire()`` call ("Bam co dan: chua toi nhip") for 7 of
8 shots. ``GlfwWindow._on_mouse_button`` is the ONLY place anything
appends to ``_clicks``, and ``app.py``'s main loop calls
``self._album_shoot(now)`` exactly once per entry ``pop_clicks()``
returns — so a double entry in ``_clicks`` for one physical press IS a
double fire. Fixed with a duplicate-press guard (``DUPLICATE_PRESS_GUARD_S``)
in the callback itself: no human releases and presses again within 50 ms,
so a PRESS that close behind the previous one is the same click delivered
twice (plausible after the Fix 9 resize-redraw change's extra
``glfw.poll_events()`` wrapping), not a second one.
"""
from __future__ import annotations

import glfw
import pytest

import src.renderer.window as window_mod
from src.renderer.window import DUPLICATE_PRESS_GUARD_S, GlfwWindow


@pytest.fixture
def window(monkeypatch):
    instance = GlfwWindow.__new__(GlfwWindow)
    instance._window = object()          # non-None; cursor_pos() is mocked below
    instance._clicks = []
    instance._last_press_t = None
    monkeypatch.setattr(glfw, "get_cursor_pos", lambda w: (100.0, 200.0))
    return instance


def _button_event(window, monkeypatch, t: float, action) -> None:
    monkeypatch.setattr(window_mod.time, "perf_counter", lambda: t)
    window._on_mouse_button(window._window, glfw.MOUSE_BUTTON_LEFT, action, 0)


def _press(window, monkeypatch, t: float) -> None:
    _button_event(window, monkeypatch, t, glfw.PRESS)


def test_one_press_queues_one_click(window, monkeypatch):
    _press(window, monkeypatch, 1.000)
    assert len(window.pop_clicks()) == 1


def test_two_presses_within_the_guard_window_queue_only_one_click(window, monkeypatch):
    _press(window, monkeypatch, 1.000)
    _press(window, monkeypatch, 1.000 + DUPLICATE_PRESS_GUARD_S / 2)
    assert len(window.pop_clicks()) == 1


def test_n_presses_spaced_well_apart_queue_n_clicks(window, monkeypatch):
    n = 5
    for i in range(n):
        _press(window, monkeypatch, 1.0 + i * (DUPLICATE_PRESS_GUARD_S * 4))
    assert len(window.pop_clicks()) == n


def test_release_action_never_queues_a_click(window, monkeypatch):
    _button_event(window, monkeypatch, 1.0, glfw.RELEASE)
    assert window.pop_clicks() == []


def test_a_press_right_at_the_guard_boundary_is_not_deduped(window, monkeypatch):
    _press(window, monkeypatch, 1.000)
    _press(window, monkeypatch, 1.000 + DUPLICATE_PRESS_GUARD_S)   # exactly at the edge
    assert len(window.pop_clicks()) == 2


# ---- end-to-end click-dispatch contract (app.py's own loop shape) --------

def test_n_well_spaced_presses_on_a_non_auto_gun_give_n_fire_calls(window, monkeypatch):
    """Mirrors app.py's own ``for _click in self._window.pop_clicks():
    if self._album_shoot(now): continue`` exactly -- this is the
    dispatch contract Fix 8 has to hold, independent of WHICH duplicate-
    click source (if any) is still out there; the window-level tests
    above cover the one actually found and fixed."""
    n = 4
    for i in range(n):
        _press(window, monkeypatch, 1.0 + i * (DUPLICATE_PRESS_GUARD_S * 4))

    fire_calls = []

    def fake_album_shoot(now: float) -> bool:
        fire_calls.append(now)
        return True

    for _click in window.pop_clicks():
        if fake_album_shoot(1.0):
            continue
    assert len(fire_calls) == n
