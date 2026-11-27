"""Virtual room keys (no GL): B toggles, + = - zoom, 0 resets; demo only."""

from __future__ import annotations

import dataclasses
from unittest import mock

import glfw

from src.config.schema import RendererSettings
from src.renderer import window as window_mod
from src.renderer.window import ALBUM_KEYS, DEMO_KEYS, V3_KEYS, GlfwWindow

ROOM = (glfw.KEY_B, glfw.KEY_EQUAL, glfw.KEY_KP_ADD, glfw.KEY_MINUS,
        glfw.KEY_KP_SUBTRACT, glfw.KEY_0)


def _press(window, keys):
    for key in keys:
        window._on_key(None, key, 0, glfw.PRESS, 0)
    with mock.patch.object(window_mod.glfw, "poll_events"), \
         mock.patch.object(window_mod.glfw, "get_key",
                           side_effect=lambda _w, _k: glfw.RELEASE):
        return window.poll_input()


def _room_flags(state):
    d = dataclasses.asdict(state)
    return {k: v for k, v in d.items() if k.startswith("room_") and v}


def test_demo_maps_each_key_to_its_room_action():
    mk = lambda: GlfwWindow(RendererSettings(), demo=True)  # noqa: E731
    assert _room_flags(_press(mk(), [glfw.KEY_B])) == {"room_toggle": True}
    for key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
        assert _room_flags(_press(mk(), [key])) == {"room_zoom_in": True}
    for key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
        assert _room_flags(_press(mk(), [key])) == {"room_zoom_out": True}
    assert _room_flags(_press(mk(), [glfw.KEY_0])) == {"room_zoom_reset": True}


def test_demo_zoom_keys_do_not_touch_the_sandbox_zoom_offset():
    st = _press(GlfwWindow(RendererSettings(), demo=True),
                [glfw.KEY_EQUAL, glfw.KEY_MINUS])
    assert not st.zoom_in and not st.zoom_out


def test_g_still_starts_the_rest_eye_measure_in_demo():
    st = _press(GlfwWindow(RendererSettings(), demo=True), [glfw.KEY_G])
    assert st.game_toggle and not _room_flags(st)


def test_sandbox_keeps_its_zoom_offset_and_has_no_room_keys():
    st = _press(GlfwWindow(RendererSettings()), ROOM)
    assert st.zoom_in and st.zoom_out
    assert not _room_flags(st)


def test_album_ignores_every_room_key():
    for kwargs in ({"demo": False, "album": True},
                   {"demo": True, "album": True},
                   {"demo": False, "album": True, "ui_v3": True}):
        st = _press(GlfwWindow(RendererSettings(), **kwargs), ROOM)
        assert not _room_flags(st), kwargs
        assert not st.zoom_in and not st.zoom_out, kwargs


def test_key_sets_keep_album_and_v3_as_on_main():
    assert set(ROOM) & ALBUM_KEYS == set()
    assert set(ROOM) & V3_KEYS == set()
    assert set(ROOM) <= DEMO_KEYS


# --- F10 relief toggle in the demo (user rule 27/11) -------------------------

#: Sandbox research keys that stay blocked in the demo.
STILL_BLOCKED = (glfw.KEY_F1, glfw.KEY_F2, glfw.KEY_F3, glfw.KEY_F7,
                 glfw.KEY_F8, glfw.KEY_F9)


def test_demo_f10_sets_the_relief_toggle():
    st = _press(GlfwWindow(RendererSettings(), demo=True), [glfw.KEY_F10])
    assert st.toggle_relief
    assert glfw.KEY_F10 in DEMO_KEYS


def test_f10_is_not_in_the_album_or_v3_key_sets():
    from src.renderer.window import _DEMO_BASE_KEYS

    assert glfw.KEY_F10 not in _DEMO_BASE_KEYS
    assert glfw.KEY_F10 not in ALBUM_KEYS
    assert glfw.KEY_F10 not in V3_KEYS
    assert ALBUM_KEYS == frozenset({
        glfw.KEY_G, glfw.KEY_T, glfw.KEY_F6, glfw.KEY_F4, glfw.KEY_F9,
        glfw.KEY_F12, glfw.KEY_C, glfw.KEY_BACKSPACE, glfw.KEY_ENTER,
        glfw.KEY_KP_ENTER, glfw.KEY_TAB})


def test_album_and_demo_album_ignore_f10():
    for kwargs in ({"demo": False, "album": True},
                   {"demo": True, "album": True},
                   {"demo": False, "album": True, "ui_v3": True}):
        st = _press(GlfwWindow(RendererSettings(), **kwargs), [glfw.KEY_F10])
        assert not st.toggle_relief, kwargs


def test_sandbox_f10_still_toggles_relief():
    assert _press(GlfwWindow(RendererSettings()), [glfw.KEY_F10]).toggle_relief


def test_other_ab_keys_stay_ignored_in_the_demo():
    st = dataclasses.asdict(_press(GlfwWindow(RendererSettings(), demo=True),
                                   STILL_BLOCKED))
    on = [k for k, v in st.items() if v is True and k != "quit_app"]
    assert on == []
