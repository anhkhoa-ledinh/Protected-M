"""
TASK_ui_sakuga_1117 Phase 7: ``src/ui_album/v3_hover.py``.

``cursor_for`` is pure (no GL) — tested directly. ``HoverFx`` spring
bookkeeping is also pure-Python (``step``/``value``) and tested without a
GL context; its ``draw`` method needs real GL and is exercised in
``tests/test_v3ui_fx_hooks.py`` instead.
"""

from src.ui_album import v3_flow as F
from src.ui_album.v3_hover import HoverFx, cursor_for


class _FakeState:
    shot = False
    disabled = frozenset()


def test_cursor_arrow_when_nothing_hovered():
    assert cursor_for(F.MENU, None, _FakeState()) == "arrow"


def test_cursor_hand_over_clickable_hotspot():
    assert cursor_for(F.MENU, "btn_Solo", _FakeState()) == "hand"


def test_cursor_ibeam_over_field():
    assert cursor_for(F.SIGN_IN, "field_user", _FakeState()) == "ibeam"


def test_cursor_arrow_on_match_screens_regardless_of_hover():
    for screen in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE):
        assert cursor_for(screen, "btn_whatever", _FakeState()) == "arrow"


def test_cursor_arrow_during_shot_mode():
    state = _FakeState()
    state.shot = True
    assert cursor_for(F.MENU, "btn_Solo", state) == "arrow"


def test_cursor_arrow_for_disabled_hotspot():
    state = _FakeState()
    state.disabled = frozenset({"btn_Solo"})
    assert cursor_for(F.MENU, "btn_Solo", state) == "arrow"


def test_hover_spring_rises_on_enter_and_decays_on_leave():
    hover = HoverFx()
    for _ in range(200):
        hover.step(F.MENU, "tab_Solo", 1.0 / 60.0)
    assert hover.value("tab_Solo") > 0.99
    for _ in range(200):
        hover.step(F.MENU, None, 1.0 / 60.0)
    assert hover.value("tab_Solo") < 0.01


def test_hover_springs_cleared_entering_match_screen():
    hover = HoverFx()
    for _ in range(200):
        hover.step(F.MENU, "tab_Solo", 1.0 / 60.0)
    assert hover.value("tab_Solo") > 0.99
    hover.step(F.PLAY, None, 1.0 / 60.0)
    assert hover.value("tab_Solo") == 0.0


def test_hover_value_zero_for_untracked_name():
    hover = HoverFx()
    assert hover.value("never_hovered") == 0.0
