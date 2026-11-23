"""
TASK_hud_fix_1119 Fix 3: don't start an unwinnable match.

``place_props``'s own reveal scan (``_candidate``/``_reveal_offset`` in
``hidden_props_search.py``) only walks ONE axis at a time, never the 2D
peek range a real lean actually covers. A spot can pass that scan yet
still never expose during real play -- exactly ``t22.png``'s case in
``logs/game_20261119_150429.log`` (1 bug behind the cat, 0/191 frames
with it exposed, every shot alpha=0.00).

``exposure_count`` double-checks a kept prop against a real 5x5 grid
over the peek range; ``_start_album_match`` drops any prop it returns 0
for, and refuses to start the match at all if that empties ``kept``.

Pure-Python throughout -- synthetic ``LayerPlane`` stacks, same helpers
as ``test_hidden_props.py``, no OpenGL needed.
"""

from __future__ import annotations

import numpy as np

from src.content.hidden_props import LayerPlane
from src.content.hidden_props_search import exposure_count
from src.ui_album import v3_flow as F


def _plane(z_m, alpha, w=0.344, h=0.258, cx=0.0, cy=0.0, zmin=None, zmax=None):
    z0 = z_m if zmin is None else zmin
    z1 = z_m if zmax is None else zmax
    return LayerPlane(alpha, z_m, w, h, cx, cy, z0, z1)


def _opaque_patch(size=64, lo=0, hi=64):
    a = np.zeros((size, size), np.uint8)
    a[lo:hi, lo:hi] = 255
    return a


# ---- exposure_count on a synthetic layer stack ----------------------------

def test_a_spot_covered_across_the_whole_peek_range_is_rejected():
    """Near layer opaque everywhere -- no eye position in the 5x5 grid
    ever exposes the prop behind it, matching t22.png's 0/191 frames."""
    near = _plane(-0.100, _opaque_patch(lo=0, hi=64), w=0.400, h=0.400)
    planes = (near, _plane(-0.300, np.zeros((8, 8), np.uint8), w=0.400, h=0.400))
    count = exposure_count(planes, (0.0, 0.0, -300.0), 12.0,
                           rest_xy=(0.0, 0.0), z=450.0)
    assert count == 0


def test_a_spot_covered_only_at_its_centre_is_accepted():
    """Near layer opaque only in a small centre patch -- leaning far
    enough (within the 5x5 grid's span) exposes the prop at some eye
    positions, even though the straight-ahead eye still sees nothing."""
    near = _plane(-0.100, _opaque_patch(size=64, lo=28, hi=36), w=0.400, h=0.400)
    planes = (near, _plane(-0.300, np.zeros((8, 8), np.uint8), w=0.400, h=0.400))
    count = exposure_count(planes, (0.0, 0.0, -300.0), 12.0,
                           rest_xy=(0.0, 0.0), z=450.0)
    assert count > 0


# ---- _start_album_match: refuses to start when nothing is winnable -------

class _Camera:
    screen_width_m = 0.344
    screen_height_m = 0.258
    screen_center_m = (0.0, 0.0)


class _Manifest:
    screens: dict = {}


class _UI:
    flow = F
    manifest = _Manifest()

    def __init__(self) -> None:
        self.state = F.V3Flow()
        self.state.go(F.LOADOUT)
        self.game = None


class _Renderer:
    def __init__(self) -> None:
        self.toasts: list = []

    def set_props(self, props) -> None:
        self._props = props

    def layer_records(self):
        return []

    def show_toast(self, text, is_error=False) -> None:
        self.toasts.append(text)


def _make_app(prop):
    """Minimal ``HcpdApplication`` surface ``_start_album_match`` touches,
    with ``_start_prop_game`` stubbed out -- placement itself is
    ``place_props``'s own concern (``test_hidden_props.py``); this test is
    only about the NEW exposure double-check and its refuse-to-start path.
    """
    from src.content.prop_game import GameState
    from src.engine.app import HcpdApplication

    app = HcpdApplication.__new__(HcpdApplication)
    app._album_options = type("Opts", (), {"is_v3": True})()
    app._album = _UI()
    app._renderer = _Renderer()
    app._virtual_camera = _Camera()
    app._rest_eye = None
    app._tracking_off = True
    app._bugs = None
    app._content = type("Content", (), {
        "overscan_design_eye_mm": (0.0, 0.0),
        "game_prop_radius_mm": 12.0,
        "content_path": "unknown.png",
    })()
    app._game_planes = ()

    def _stub_start_prop_game(count=0, **_kw):
        app._game = GameState(image="unknown.png", props=[dict(prop)],
                              tracking_on=True, started_at_s=0.0)
        app._game_planes = (
            _plane(-0.100, _opaque_patch(lo=0, hi=64), w=0.400, h=0.400),
            _plane(-0.300, np.zeros((8, 8), np.uint8), w=0.400, h=0.400),
        )

    app._start_prop_game = _stub_start_prop_game
    return app


def test_a_photo_with_no_valid_spot_does_not_start(monkeypatch):
    import src.engine.app as app_module

    prop = {"pos_mm": (0.0, 0.0, -300.0), "radius_mm": 12.0}
    app = _make_app(prop)
    monkeypatch.setattr(
        "src.game.viewport_filter.filter_props",
        lambda props, *a, **k: (list(props), 0))
    monkeypatch.setattr(
        "src.content.hidden_props_search.exposure_count", lambda *a, **k: 0)

    app._start_album_match(0.0)

    assert app._album.game is None
    assert app._album.state.screen == F.PHOTOS
    assert "no spot a bug can be found" in app._album.state.message


def test_a_photo_with_one_winnable_spot_keeps_it(monkeypatch):
    prop = {"pos_mm": (0.0, 0.0, -300.0), "radius_mm": 12.0}
    unwinnable = {"pos_mm": (50.0, 0.0, -300.0), "radius_mm": 12.0}
    app = _make_app(prop)

    def _stub_start_prop_game(count=0, **_kw):
        from src.content.prop_game import GameState

        app._game = GameState(image="unknown.png", props=[prop, unwinnable],
                              tracking_on=True, started_at_s=0.0)
        app._game_planes = ()

    app._start_prop_game = _stub_start_prop_game
    monkeypatch.setattr(
        "src.game.viewport_filter.filter_props",
        lambda props, *a, **k: (list(props), 0))
    monkeypatch.setattr(
        "src.content.hidden_props_search.exposure_count",
        lambda planes, pos_mm, *a, **k: 0 if pos_mm is unwinnable["pos_mm"] else 3)
    monkeypatch.setattr(
        "src.game.match.start_match",
        lambda level, game, weapon, items, item_weight: type(
            "Match", (), {"layer_total": 0, "ammo_enabled": False})())

    app._start_album_match(0.0)

    assert app._album.game is not None
    assert len(app._game.props) == 1
    assert app._game.props[0] is prop
    assert app._album.state.screen == F.PLAY
