"""
TASK_game_feel_1118 Phase 6: ``src/engine/app.py``'s ``_tick_scope`` —
entering/exiting around holding the right mouse button, dropping and
resuming around a reload, and gating to the Precision gun only. Same
fake-clock pattern as ``tests/test_auto_fire.py``.
"""

from __future__ import annotations

import numpy as np

from src.engine.app import HcpdApplication
from src.renderer.camera import VirtualCamera
from src.ui_album import v3_flow as F
from src.game.match import AlbumMatch
from src.ui_album.options import AlbumOptions
from src.ui_album.scope import ScopeState

GLASS_MM = (344.0, 194.0)
CENTRE_MM = (0.0, 0.0)
RELOAD_S = 1.3


class _FakePainter:
    def map(self, x, y):
        return (x, y)      # 1:1 — only the call shape matters here


class _FakeAlbum:
    """Mirrors the ``V3UI`` scope methods ``_tick_scope`` calls, backed
    by a REAL ``ScopeState`` so entering/exiting/finer-aim behave exactly
    as the real game does."""

    def __init__(self, weapon_name: str, screen: str = F.PLAY) -> None:
        self.flow = F
        self.state = F.V3Flow()
        self.state.screen = screen
        self.mouse = (640.0, 360.0)
        self.painter = _FakePainter()
        self.game = AlbumMatch(level=1, state=None, started_s=0.0,
                               weapon_name=weapon_name)
        self.game.ammo_enabled = True
        self.game.reload_s = RELOAD_S
        self._scope = ScopeState()

    @property
    def is_scoped(self) -> bool:
        return self._scope.active

    @property
    def scope_visually_active(self) -> bool:
        return self._scope.visually_active

    @property
    def scope_zoom(self) -> float:
        return self._scope.z

    def enter_scope(self) -> None:
        self._scope.enter(self.mouse)

    def exit_scope(self) -> None:
        self._scope.exit()

    def step_scope(self, dt: float) -> None:
        self._scope.step(dt)

    def apply_scope_aim(self, raw_mouse):
        return self._scope.update_crosshair(self.mouse, raw_mouse)


def make_app(weapon_name: str = "Precision gun", screen: str = F.PLAY) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._album_options = AlbumOptions(enabled=True, ui="v3")
    app._album = _FakeAlbum(weapon_name, screen)
    app._scope_last_t = None
    app._window = type("W", (), {"is_right_mouse_down": lambda self: app.right_held,
                                 "set_cursor_pos": lambda self, x, y: None})()
    app.right_held = False
    app._virtual_camera = VirtualCamera(
        screen_width_m=GLASS_MM[0] / 1000.0, screen_height_m=GLASS_MM[1] / 1000.0,
        screen_center_m=np.array([0.0, 0.0, 0.0]))
    app._glass_mm = lambda: (GLASS_MM, CENTRE_MM)
    return app


def _tick(app: HcpdApplication, now: float, *, held: bool,
         mouse: tuple = (640.0, 360.0)) -> None:
    app.right_held = held
    app._album.mouse = app._tick_scope(now, mouse)


def test_holding_right_mouse_enters_scope_on_precision_gun():
    app = make_app()
    _tick(app, 0.0, held=True)
    assert app._album.is_scoped is True
    assert app._virtual_camera.scope_eye_anchor_mm is not None


def test_releasing_right_mouse_exits_scope():
    app = make_app()
    _tick(app, 0.0, held=True)
    assert app._album.is_scoped is True
    _tick(app, 0.1, held=False)
    assert app._album.is_scoped is False


def test_other_guns_never_scope():
    for weapon in ("Spread gun", "Heavy-mag gun"):
        app = make_app(weapon_name=weapon)
        _tick(app, 0.0, held=True)
        assert app._album.is_scoped is False, weapon


def test_pvp_hunt_screen_also_scopes():
    app = make_app(screen=F.PVP_HUNT)
    _tick(app, 0.0, held=True)
    assert app._album.is_scoped is True


def test_starting_a_reload_drops_the_scope():
    app = make_app()
    _tick(app, 0.0, held=True)
    assert app._album.is_scoped is True
    app._album.game.rounds_left = 0
    app._album.game.reload_until_s = 0.1 + RELOAD_S   # reload starts at t=0.1
    _tick(app, 0.1, held=True)                          # still held, but reloading
    assert app._album.is_scoped is False


def test_scope_resumes_after_the_reload_finishes_if_still_held():
    app = make_app()
    _tick(app, 0.0, held=True)
    app._album.game.rounds_left = 0
    app._album.game.reload_until_s = 0.1 + RELOAD_S
    _tick(app, 0.1, held=True)                          # reload starts, scope drops
    assert app._album.is_scoped is False
    _tick(app, 0.1 + RELOAD_S + 0.01, held=True)         # reload just finished
    assert app._album.is_scoped is True


def test_scope_does_not_resume_if_released_during_the_reload():
    app = make_app()
    _tick(app, 0.0, held=True)
    app._album.game.rounds_left = 0
    app._album.game.reload_until_s = 0.1 + RELOAD_S
    _tick(app, 0.1, held=False)                          # released mid-reload
    assert app._album.is_scoped is False
    _tick(app, 0.1 + RELOAD_S + 0.01, held=False)
    assert app._album.is_scoped is False
