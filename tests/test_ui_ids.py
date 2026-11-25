"""B1: ``src/ui_album/ids.py`` is the one list of ui ids."""

from __future__ import annotations

import json
from pathlib import Path

from src.ui_album import ids

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = json.loads((ROOT / "assets/ui_v3/manifest.json").read_text("utf-8"))
SCREENS = set(MANIFEST["screens"])
HOTSPOTS = {h["n"] for s in MANIFEST["screens"].values() for h in s.get("hotspots", [])}
#: Ids the code adds itself (modal buttons, forgot-password fields): not in the manifest.
CODE_ONLY = {"btn_confirm_no", "btn_confirm_yes", "btn_forgot_cancel", "btn_forgot_submit",
             "btn_recovery_copy", "btn_recovery_saved", "field_forgot_code",
             "field_forgot_pass", "field_forgot_pass2", "field_forgot_user"}


def _constants() -> dict[str, str]:
    return {k: v for k, v in vars(ids).items() if k.isupper() and isinstance(v, str)}


def test_every_constant_is_a_manifest_id_or_a_known_code_only_id():
    unknown = {k: v for k, v in _constants().items()
               if v not in SCREENS | HOTSPOTS | CODE_ONLY}
    assert unknown == {}


def test_no_two_constants_share_a_value():
    values = list(_constants().values())
    assert len(values) == len(set(values))


def test_every_manifest_screen_and_hotspot_is_used_or_listed_as_not_used():
    constants = set(_constants().values())
    missing = (SCREENS | HOTSPOTS) - constants - ids.NOT_USED_IN_CODE
    assert missing == set(), "add a constant to ids.py or list it in NOT_USED_IN_CODE"
    stale = ids.NOT_USED_IN_CODE & constants
    assert stale == set(), "listed as not used but has a constant"
    assert ids.NOT_USED_IN_CODE <= SCREENS | HOTSPOTS


def test_the_fx_copy_of_the_screen_names_matches_ids():
    """``src.fx`` may not import ``ui_album`` (architecture), so
    ``fx/screens.py`` keeps its own copies; they must stay equal."""
    from src.fx import screens as fx_screens

    pairs = {"_PLAY": ids.PLAY, "_PVP_HUNT": ids.PVP_HUNT, "_PVP_HIDE": ids.PVP_HIDE,
             "_LOADING": ids.LOADING, "_LOADING_READY": ids.LOADING_READY,
             "_TILT": ids.TILT, "_LOADOUT": ids.LOADOUT, "_WIN": ids.WIN,
             "_PVP_RESULT": ids.PVP_RESULT}
    for name, value in pairs.items():
        assert getattr(fx_screens, name) == value, name
