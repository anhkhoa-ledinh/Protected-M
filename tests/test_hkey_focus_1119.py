"""
TASK_hkey_profile_1119 Part 1: the char callback (typing into a text
field) and the single-letter shortcut keys (H, C, 1-3 -- see
``window.py``'s ``V3_KEYS``) fire from TWO INDEPENDENT GLFW callbacks
for the SAME keypress. Typing "h" into a username used to also toggle
the HUD (``app._tick_v3_input``'s ``hud_hide_toggle`` branch had no
focus/screen guard at all); typing a digit or "c" hit the item-slot/
recalibrate shortcuts the same way.

Pure-logic tests via ``HcpdApplication.__new__`` + minimal fakes, same
pattern as ``tests/test_auto_fire.py``/``tests/test_scope_app.py``.
Typing itself goes through the REAL ``v3_flow.type_char``/
``forgot_type_char`` -- the exact functions the real char callback
reaches via ``V3UI.key()``.
"""

from __future__ import annotations

from types import SimpleNamespace

from src.engine.app import HcpdApplication
from src.renderer.window import InputState
from src.ui_album import v3_flow as F
from src.ui_album.options import AlbumOptions


class _FakeAlbum:
    def __init__(self, screen: str) -> None:
        self.flow = F
        self.state = F.V3Flow()
        self.state.screen = screen
        self.hud_toggled = 0

    def toggle_hud_hidden(self) -> bool:
        self.hud_toggled += 1
        return True

    def move_focus(self, delta: int) -> None:
        pass

    def pop_account_corrupt_toast(self) -> bool:
        return False

    def set_camera_frame(self, frame) -> None:
        pass


def make_app(screen: str = F.SIGN_IN, *, is_v3: bool = True) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album_options = AlbumOptions(enabled=True, ui="v3" if is_v3 else "album")
    app._album = _FakeAlbum(screen)
    app._camera = SimpleNamespace(peek_frame=lambda: None)
    app._window = SimpleNamespace(display_mode="windowed")
    app._renderer = SimpleNamespace(layer_records=lambda: [])
    app._content = SimpleNamespace(layer_count=3)
    app._v3_live = False
    return app


def _type(state, text: str) -> None:
    for ch in text:
        F.type_char(state, ch)


# ---- _v3_field_focused() itself ------------------------------------------

def test_field_focused_true_with_a_sign_in_field_selected():
    app = make_app(F.SIGN_IN)
    app._album.state.focus = "field_user"
    assert app._v3_field_focused() is True


def test_field_focused_true_while_the_forgot_dialog_is_open():
    app = make_app(F.SIGN_IN)
    app._album.state.focus = None
    app._album.state.forgot_open = True
    assert app._v3_field_focused() is True


def test_field_focused_false_with_nothing_selected():
    app = make_app(F.SIGN_IN)
    assert app._v3_field_focused() is False


def test_field_focused_always_false_for_the_legacy_album_ui():
    app = make_app(F.SIGN_IN, is_v3=False)
    app._album.state.focus = "field_user"          # would be True under v3
    assert app._v3_field_focused() is False


# ---- typing "h"/digits/"c" into a field does not fire the shortcut -------

def test_typing_khoa_h_into_the_username_field_is_exact_and_does_not_toggle():
    app = make_app(F.SIGN_IN)
    app._album.state.focus = "field_user"
    _type(app._album.state, "Khoa h")
    app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
    assert app._album.state.player == "Khoa h"
    assert app._album.hud_toggled == 0


def test_typing_hhh_into_the_password_field_is_exact_and_does_not_toggle():
    app = make_app(F.SIGN_UP)
    app._album.state.focus = "field_pass"
    _type(app._album.state, "hhh")
    app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
    assert app._album.state.password == "hhh"
    assert app._album.hud_toggled == 0


def test_typing_hhh_into_the_forgot_password_user_field_is_exact_and_does_not_toggle():
    app = make_app(F.SIGN_IN)
    app._album.state.forgot_open = True
    app._album.state.forgot_focus = "field_forgot_user"
    for ch in "hhh":
        F.forgot_type_char(app._album.state, ch)
    app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
    assert app._album.state.forgot_user == "hhh"
    assert app._album.hud_toggled == 0


def test_typing_a_digit_into_a_field_does_not_select_an_item_slot():
    app = make_app(F.SIGN_UP)
    app._album.state.focus = "field_user"
    before = app._album.state.slot_selected
    _type(app._album.state, "2")           # the char callback: a literal "2"
    app._tick_v3_input(InputState(item_slot=2), 0.0)   # the key callback
    assert app._album.state.player == "2"
    assert app._album.state.slot_selected == before


# ---- H still works where it is supposed to --------------------------------

def test_h_toggles_the_hud_in_play_without_a_focused_field():
    app = make_app(F.PLAY)
    app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
    assert app._album.hud_toggled == 1


def test_h_toggles_the_hud_in_pvp_hunt_and_pvp_hide():
    for screen in (F.PVP_HUNT, F.PVP_HIDE):
        app = make_app(screen)
        app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
        assert app._album.hud_toggled == 1


def test_h_does_not_toggle_the_hud_in_play_while_a_field_has_focus():
    """Task's own caveat: "if a field can have focus there" -- the join
    field is the one real case (PvP room code can be typed mid-flow)."""
    app = make_app(F.PLAY)
    app._album.state.focus = "field_join"
    app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
    assert app._album.hud_toggled == 0


def test_h_does_not_toggle_the_hud_outside_a_match_screen_even_unfocused():
    for screen in (F.SIGN_IN, F.MENU, F.LOADOUT):
        app = make_app(screen)
        app._tick_v3_input(InputState(hud_hide_toggle=True), 0.0)
        assert app._album.hud_toggled == 0


# ---- item slots still work in PLAY ----------------------------------------

def test_item_slot_still_applies_in_play_without_focus():
    app = make_app(F.PLAY)
    app._tick_v3_input(InputState(item_slot=3), 0.0)
    assert app._album.state.slot_selected == 3
