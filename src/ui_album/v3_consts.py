"""Module-level constants shared by ``V3UI`` and its mixins (moved out of
``v3_ui.py`` unchanged)."""

from __future__ import annotations

import logging
from src.ui_album import v3_flow as F
from src.ui_album import ids as ID

logger = logging.getLogger("src.ui_album.v3_ui")

#: TASK_perf_textures_1119 Fix 4: framebuffer width above which the fx@2x
#: pack (baked at ``image_scale=2``, 2560 basis) looks better than the
#: 1.5x default — below it, fx@2x is wasted decode/upload/GPU-RAM for no
#: visible gain.
FX_WIDE_THRESHOLD_PX = 1920

#: TASK_ui_sakuga_1117 Phase 6 — hotspot names/prefixes that get a plain
#: ``ui_tap`` on click (the PRIMARY button of the screen gets
#: ``ui_confirm`` instead; anything not listed here gets nothing —
#: fields/slider/dropdown/dropzone/switch/links/icons/recent_memory
#: already have Phase 5's hover/press feedback, a tap on top would be
#: noise on a text-entry/drag control).
_UI_TAP_PREFIXES = ("btn_", "tab_", "seg_", "mode_", "display_", "theme_",
                    "level_row_", "weapon_", "sticker_", "slot_")
_UI_TAP_EXACT = frozenset({ID.LEAVE, ID.CONTINUE})
#: A tap on something this tall/wide shows at the CLICK point, not the
#: hotspot centre (a 300px-wide ``mode_`` card, or the full-screen
#: ``continue`` hotspot) — centred would land far from where the player
#: actually looked.
_UI_TAP_BIG_HEIGHT_PX, _UI_TAP_BIG_WIDTH_PX = 100.0, 360.0

#: "kills" counter text per screen, the wordless star_pop's anchor on
#: every kill. ``PVP_HIDE`` has none — the hider doesn't shoot.
_KILL_STAR_TEXT = {F.PLAY: "kills", F.PVP_HUNT: "kills_you"}

#: Loading screen (``_step_loading_fx``): the progress bar's own design-px
#: span, so the edge/word effects track the fill exactly.
_LOAD_BAR_X0, _LOAD_BAR_W, _LOAD_BAR_Y = 48.0, 1184.0, 665.0
_LOAD_EDGE_INTERVAL_S = 0.25
_LOAD_DONE_XY = (1232.0, 665.0)
_GOGO_X_RANGE = (200.0, 1080.0)
_GOGO_Y = 600.0
_STEP_DOT_X = (53.0, 349.0, 645.0, 941.0)
_STEP_DOT_Y = 687.0

#: Tilt checklist marker centres (Phase 2's ``CHECK_ROWS_Y`` is the row's
#: TEXT baseline; this is the task's own, slightly different, marker-glyph
#: centre — kept as its own constant rather than reusing one for the other).
_TILT_STAR_X, _TILT_STAR_Y0, _TILT_STAR_DY = 783.0, 497.0, 32.0

#: TASK_game_feel_1118 P3: the OLD anchor, (1084, 370), was the exact
#: centre of the ``score`` text rect ((984, 308, 200, 125) on
#: v3_09a_result_win) — the burst's "pon" word always landed right on
#: the score digits. Moved straight up, clear of both the score rect AND
#: the row_value text just above it — checked with
#: ``src/fx/layout.py``'s rect-intersection helpers (see
#: ``tests/test_fx_layout.py``), not eyeballed.
_WIN_SCORE_XY = (1084.0, 220.0)
_PVP_CROWN_XY = (640.0, 50.0)
_RANK_UP_DELAY_S = 0.5

_TOAST_XY = (640.0, 648.0)
_TOAST_PHOTOS_XY = (640.0, 596.0)

#: TASK_game_feel_fix_1118 Fix 10.2/10.3: HUD card fade — unscoped
#: proximity (crosshair within a card's rect, grown by this much) drops
#: that one card to this alpha; scoped drops EVERY fadeable card to 0.
#: Spring constants are the task's own numbers, shared by both targets —
#: only ``.to`` differs.
_HUD_FADE_GROW_PX = 48.0
_HUD_FADE_PROXIMITY_ALPHA = 0.22
_HUD_FADE_SPRING_K, _HUD_FADE_SPRING_D = 420.0, 26.0

#: TASK_hud_fix_1119 Fix 2: H hide/show — a single Spring (not per-card;
#: this one drives ``state.hud_hidden_alpha``, the multiplier every card
#: and "free" HUD element fades with) tuned to settle in ~0.15s: k=600,
#: d=40 reaches within 2% of its target at t≈0.15s (simulated directly
#: against ``Spring.step``, 1/60s substeps), with negligible overshoot
#: (<0.005 past 0 or 1).
_HUD_HIDE_SPRING_K, _HUD_HIDE_SPRING_D = 600.0, 40.0
_HUD_HIDDEN_TOAST = "HUD hidden — press H to show"
