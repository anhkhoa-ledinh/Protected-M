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

from src.ui_album import v3_flow as F

#: Effects available on every screen that isn't special-cased below.
MENU_SET = frozenset({"ui_confirm", "ui_tap", "toast_in"})

#: PLAY/PVP_HUNT/PVP_HIDE: sprite sheets only, no word effect.
_MATCH_SET = frozenset({"hit_kill", "hit_crit", "hit_block", "star_pop"})

_MATCH_SCREENS = (F.PLAY, F.PVP_HUNT, F.PVP_HIDE)
_LOADING_SCREENS = (F.LOADING, F.LOADING_READY)
_TILT_LOADOUT_SCREENS = (F.TILT, F.LOADOUT)
_WIN_RESULT_SCREENS = (F.WIN, F.PVP_RESULT)

_SCREEN_SETS = {}
for _screen in _MATCH_SCREENS:
    _SCREEN_SETS[_screen] = _MATCH_SET
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
