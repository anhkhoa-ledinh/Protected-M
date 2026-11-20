"""TASK_ui_sakuga_1117 Phase 7: ``src/fx/screens.py`` per-screen scoping."""

from src.fx import screens as S
from src.ui_album import v3_flow as F


def test_match_screens_get_sheets_only_set():
    for screen in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE):
        assert S.effects_for_screen(screen) == S.MATCH_SET
        assert S.is_match_screen(screen)


def test_loading_screens_get_menu_plus_loading_set():
    expected = S.MENU_SET | {"load_edge", "load_done", "star_pop"}
    for screen in (F.LOADING, F.LOADING_READY):
        assert S.effects_for_screen(screen) == expected
        assert S.words_for_screen(screen) == frozenset({("gogo", "ink")})
        assert not S.is_match_screen(screen)


def test_tilt_and_loadout_get_menu_plus_star_pop():
    expected = S.MENU_SET | {"star_pop"}
    assert S.effects_for_screen(F.TILT) == expected
    assert S.effects_for_screen(F.LOADOUT) == expected


def test_win_and_pvp_result_get_menu_plus_stars():
    expected = S.MENU_SET | {"star_five", "star_pop"}
    assert S.effects_for_screen(F.WIN) == expected
    assert S.effects_for_screen(F.PVP_RESULT) == expected


def test_every_other_screen_is_menu_set():
    others = (F.SIGN_IN, F.SIGN_UP, F.MENU, F.SETTINGS, F.LEVELS, F.PHOTOS,
             F.PVP_MODE, F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND, F.BOARD,
             F.LOSE, F.OFFLINE)
    for screen in others:
        assert S.effects_for_screen(screen) == S.MENU_SET
        assert S.words_for_screen(screen) == frozenset()


def test_every_real_screen_name_resolves_without_error():
    from src.ui_album.v3 import V3Manifest
    from pathlib import Path

    manifest = V3Manifest(Path("."))
    for screen in manifest.screen_names():
        S.effects_for_screen(screen)
        S.words_for_screen(screen)
        S.is_match_screen(screen)
