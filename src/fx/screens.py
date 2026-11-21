"""
Which Sakuga effect NAMES are in scope on each screen (TASK_ui_sakuga_1117's
per-screen table). This only names effects, not individual textures —
``FxPlayer.active_textures()`` already works out exactly which sheet/word
PNGs a *live* effect needs, respecting whatever ``words=`` the spawner
passed; what this module adds is screen-level scoping, so Phase 4's
``FxLibrary`` knows what's safe to free when the screen changes (free
anything outside the new screen's set that no live effect still needs),
and in particular so the menu set and the match set are never resident
together.

"Match screens ... sheets only (no word)" (the task's own wording) means:
on ``PLAY``/``PVP_HUNT``/``PVP_HIDE``, the caller spawns ``hit_kill``/
``hit_crit``/``hit_block``/``star_pop`` with ``words=False`` (Phase 6) —
so their word PNGs never enter ``active_textures()`` there, and this
module does not need to reserve room for them.
"""

from __future__ import annotations

#: Screen names (``src.ui_album.v3_flow``'s own values) duplicated here,
#: not imported (TASK_arch_refactor_1120 R2: ``fx`` must not depend on
#: ``ui_album``) — a plain string, changes only if the Penpot screen
#: itself is renamed, same risk as any other literal in this table.
_PLAY = "v3_08_gameplay_hud"
_PVP_HUNT = "v3_12a_pvp_hunter_hud"
_PVP_HIDE = "v3_12b_pvp_bug_hud"
_LOADING = "v3_07_loading"
_LOADING_READY = "v3_07_loading_ready"
_TILT = "v3_06_head_tilt_check"
_LOADOUT = "v3_05_loadout"
_WIN = "v3_09a_result_win"
_PVP_RESULT = "v3_13_pvp_result"

#: Effects available on every screen that isn't special-cased below.
MENU_SET = frozenset({"ui_confirm", "ui_tap", "toast_in"})

#: PLAY/PVP_HUNT/PVP_HIDE: sprite sheets only, no word effect.
MATCH_SET = frozenset({"hit_kill", "hit_crit", "hit_block", "star_pop"})

#: TASK_game_feel_fix_1118 Fix 7: WIN/PVP_RESULT — ``star_pop`` is
#: already in ``MATCH_SET`` above; ``star_five`` was NOT in any
#: priority-preload set (only ``MENU_SET``/``MATCH_SET`` get decoded
#: before the low-priority "everything else" catch-all), and WIN is a
#: screen every player reaches in their very first match, same as the
#: menu/match screens — a real (if rare) chance the player gets there
#: before "star_five" has decoded, exactly why ``fx_win.png`` showed
#: nothing in the 1118 report's own screenshot sequence (fast, automated,
#: reaching WIN well before the low-priority tier finished).
RESULT_SET = frozenset({"star_five"})

_MATCH_SCREENS = (_PLAY, _PVP_HUNT, _PVP_HIDE)
_LOADING_SCREENS = (_LOADING, _LOADING_READY)
_TILT_LOADOUT_SCREENS = (_TILT, _LOADOUT)
_WIN_RESULT_SCREENS = (_WIN, _PVP_RESULT)

_SCREEN_SETS = {}
for _screen in _MATCH_SCREENS:
    _SCREEN_SETS[_screen] = MATCH_SET
for _screen in _LOADING_SCREENS:
    _SCREEN_SETS[_screen] = MENU_SET | {"load_edge", "load_done", "star_pop"}
for _screen in _TILT_LOADOUT_SCREENS:
    _SCREEN_SETS[_screen] = MENU_SET | {"star_pop"}
for _screen in _WIN_RESULT_SCREENS:
    _SCREEN_SETS[_screen] = MENU_SET | {"star_five", "star_pop"}


#: Bare WORD textures (no sheet/effect behind them) scoped to specific
#: screens, as ``(word, style)`` pairs — the loading bar's "gogo" is
#: spawned directly via ``FxPlayer.spawn_word``, so it never appears in
#: ``_SCREEN_SETS`` above (that table is effect NAMES only).
_WORD_SETS = {_screen: frozenset({("gogo", "ink")}) for _screen in _LOADING_SCREENS}


def effects_for_screen(screen: str) -> frozenset:
    """Effect names in scope on ``screen``. Every screen not listed above
    falls back to :data:`MENU_SET`."""
    return _SCREEN_SETS.get(screen, MENU_SET)


def words_for_screen(screen: str) -> frozenset:
    """Bare ``(word, style)`` pairs in scope on ``screen`` — empty for
    every screen except the loading ones."""
    return _WORD_SETS.get(screen, frozenset())


def is_match_screen(screen: str) -> bool:
    """``PLAY``/``PVP_HUNT``/``PVP_HIDE`` — no menu-set effects, no toast,
    and Phase 5's hover/press feedback is skipped entirely."""
    return screen in _MATCH_SCREENS
