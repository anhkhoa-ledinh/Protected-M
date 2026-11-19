"""TASK_perf_textures_1119 Fix 7: compact HUD keeps the Pause pill and
full-size-but-scaled item slots, drops the broken heart glyph, every
compact pill string renders with glyphs the pack fonts actually have,
and the "press H" toast fires worded for whichever mode is active.
"""
from __future__ import annotations

from pathlib import Path

import pytest
from fontTools.ttLib import TTFont

from src.engine.app import hud_compact_hint_text
from src.ui_album.v3 import (_COMPACT_PAUSE_TEXT, _COMPACT_PILL_FONT,
                             V3State, _draw_compact_hud)
from src.ui_album.v3_values import _hud_items, _scale_rect, _scaled_runs

ROOT = Path(__file__).resolve().parents[1]


class _FakeCompactPainter:
    def __init__(self) -> None:
        self.pills: list = []

    def pill(self, record, text, style, alpha: float = 1.0) -> None:
        self.pills.append((record, text, style, alpha))

    def text_width(self, runs, texts) -> float:
        return 100.0


# ---- Pause pill ------------------------------------------------------------

def test_pause_pill_is_always_drawn_in_compact_mode():
    state = V3State()
    state.texts = {}
    painter = _FakeCompactPainter()
    _draw_compact_hud(painter, state)
    assert any(text == _COMPACT_PAUSE_TEXT for _rec, text, _style, _a in painter.pills)


def test_pause_pill_text_is_plain_ascii():
    assert _COMPACT_PAUSE_TEXT.isascii()


# ---- no broken heart glyph --------------------------------------------------

def test_hp_pill_never_contains_the_heart_character():
    state = V3State()
    state.texts = {"timer": "01:00", "hp_value": "72", "st_value": "48"}
    painter = _FakeCompactPainter()
    _draw_compact_hud(painter, state)
    texts = [text for _rec, text, _style, _a in painter.pills]
    assert not any("♥" in t for t in texts)
    assert any(t.startswith("HP ") for t in texts)


# ---- glyph coverage: every compact pill string, against the real font -----

def _font_path(family: str, weight: int) -> Path:
    import json

    manifest = json.loads((ROOT / "assets" / "ui_v3" / "manifest.json").read_text(encoding="utf-8"))
    table = manifest["fonts"]["files"][family]
    key = str(weight)
    if key not in table:
        key = min(table, key=lambda k: abs(int(k) - weight))
    return ROOT / "assets" / "ui_v3" / table[key]


def _missing_glyphs(text: str, family: str, weight: int) -> set[str]:
    font = TTFont(_font_path(family, weight))
    cmap = font.getBestCmap()
    return {ch for ch in text if ch != " " and ord(ch) not in cmap}


@pytest.mark.parametrize("state_texts", [
    {"level_tab": "LEVEL 3 · Layer 2/4", "layer": "layer 2/4", "timer": "02:20",
     "kills": ["1", "/4"], "intact_value": ["84", "%"], "hp_value": "100",
     "st_value": "50", "ammo": ["9", "/12"], "weapon_name": "Precision gun"},
    {"layer": "layer 1/4", "timer": "01:00", "kills_you": ["0", "/4"]},
    {"timer": "00:30"},
    {"timer": "15:59", "hp_value": "0", "st_value": "0",
     "exposure_value": "100%"},
], ids=["full_solo", "pvp_hunt", "minimal", "exposure"])
def test_every_compact_pill_string_only_uses_glyphs_the_font_has(state_texts):
    state = V3State()
    state.texts = dict(state_texts)
    painter = _FakeCompactPainter()
    _draw_compact_hud(painter, state)
    family, weight = _COMPACT_PILL_FONT["ff"], _COMPACT_PILL_FONT["fw"]
    for _rec, text, _style, _alpha in painter.pills:
        missing = _missing_glyphs(text, family, weight)
        assert not missing, f"pill {text!r} uses glyphs missing from {family} {weight}: {missing}"


def test_heart_character_itself_is_confirmed_missing_from_the_pack_font():
    """Sanity check for the regression this fix addresses: the font
    really doesn't have it, so dropping it was necessary, not cosmetic."""
    missing = _missing_glyphs("♥", _COMPACT_PILL_FONT["ff"], _COMPACT_PILL_FONT["fw"])
    assert missing == {"♥"}


# ---- item-slot 80% scale (TASK_perf_textures_1119 Fix 7) -------------------

def _play_flow_and_match():
    from src.content.prop_game import GameState
    from src.ui_album import v3_flow as F
    from src.ui_album.game_bridge import AlbumMatch

    flow = F.V3Flow()
    flow.go(F.PLAY)
    flow.level = 5
    flow.weapon = 0
    flow.items = ["Compass", "Energy drink"]
    flow.slot_selected = 1
    props = [{"pos_mm": (0.0, 0.0, -180.0), "radius_mm": 40.0} for _ in range(12)]
    game_state = GameState(image="t.png", props=props, tracking_on=False,
                           started_at_s=0.0, found=[True] * 7 + [False] * 5)
    match = AlbumMatch(level=5, state=game_state, started_s=0.0,
                       weapon_name="Precision gun", items=["Compass", "Energy drink"],
                       used={"Energy drink": 1}, hp=72.0, stamina=48.0,
                       layer_index=2, layer_total=4, mag_size=10, rounds_left=9,
                       ammo_enabled=True)
    return flow, match


def test_item_slots_are_unscaled_when_not_compact():
    from src.ui_album.v3 import V3Manifest

    manifest = V3Manifest(ROOT)
    flow, match = _play_flow_and_match()
    flow.hud_compact = False
    state = V3State()
    _hud_items(state, flow, match, manifest, flow.screen)
    assert state.borrowed["islot#0"][2][2:] == (108.0, 100.0)
    assert state.borrowed["ikey#0"][2][2:] == (20.0, 20.0)


def test_item_slots_shrink_to_80_percent_when_compact():
    from src.ui_album.v3 import V3Manifest

    manifest = V3Manifest(ROOT)
    flow, match = _play_flow_and_match()
    flow.hud_compact = True
    state = V3State()
    _hud_items(state, flow, match, manifest, flow.screen)
    w, h = state.borrowed["islot#0"][2][2:]
    assert w == pytest.approx(108.0 * 0.80)
    assert h == pytest.approx(100.0 * 0.80)
    kw, kh = state.borrowed["ikey#0"][2][2:]
    assert kw == pytest.approx(20.0 * 0.80)
    assert kh == pytest.approx(20.0 * 0.80)


def test_item_slot_text_font_size_also_shrinks_when_compact():
    from src.ui_album.v3 import V3Manifest

    manifest = V3Manifest(ROOT)
    flow, match = _play_flow_and_match()
    flow.hud_compact = True
    state = V3State()
    _hud_items(state, flow, match, manifest, flow.screen)
    key_number_entry = next(e for e in state.extra if e[2] == ["1"])
    runs = key_number_entry[1]
    assert all(run["fs"] == pytest.approx(16.0 * 0.80, rel=0.2) or run["fs"] < 16.0
              for run in runs)


def test_item_slot_stays_centred_on_its_original_centre_when_shrunk():
    """Shrinking must not visibly drift the slot group -- each slot
    scales about its OWN centre, so the centre point itself is
    invariant under the transform."""
    rect = (490.0, 578.0, 108.0, 100.0)
    centre = (490.0 + 54.0, 578.0 + 50.0)
    scaled = _scale_rect(rect, 0.80, centre)
    scaled_centre = (scaled[0] + scaled[2] / 2.0, scaled[1] + scaled[3] / 2.0)
    assert scaled_centre == pytest.approx(centre)


def test_scale_rect_identity_at_scale_one():
    rect = (1.0, 2.0, 3.0, 4.0)
    assert _scale_rect(rect, 1.0, (0.0, 0.0)) == rect


def test_scaled_runs_identity_at_scale_one():
    runs = [{"fs": 16.0, "ff": "Nunito"}]
    assert _scaled_runs(runs, 1.0) is runs


def test_scaled_runs_scales_font_size_only():
    runs = [{"fs": 16.0, "ff": "Nunito", "fw": 800}]
    scaled = _scaled_runs(runs, 0.5)
    assert scaled[0]["fs"] == 8.0
    assert scaled[0]["ff"] == "Nunito"
    assert scaled[0]["fw"] == 800
    assert runs[0]["fs"] == 16.0          # original untouched


# ---- "press H" toast, worded for the CURRENT mode --------------------------

def test_hint_text_for_full_hud_mode():
    assert hud_compact_hint_text(False) == "Press H for a compact HUD"


def test_hint_text_for_compact_hud_mode():
    assert hud_compact_hint_text(True) == "Press H for the full HUD"
