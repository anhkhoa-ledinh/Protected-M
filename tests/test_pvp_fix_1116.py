"""TASK_pvp_fix_1116: three UI bugs the reviewer found in results/pvp_1116/
screenshots — the fake "Player 2" name on PVP_FOUND, the "is splitting
layers..." message surviving past the lobby, and the yellow debug spheres
drawn over real bug sprites in a PvP match."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from src.engine.app import HcpdApplication
from src.ui_album import v3_flow as F
from src.ui_album import v3_values


# ---- Fix 1: real opponent name/initial on PVP_FOUND ------------------------

class PvpFoundShowsTheRealOpponentNameTest(unittest.TestCase):
    def _state(self, opponent_name: str):
        flow = F.V3Flow()
        flow.screen = F.PVP_FOUND
        flow.player = "me"
        flow.pvp_opponent_name = opponent_name
        flow.found_at_s = 0.0
        state = SimpleNamespace(texts={}, focus=None)
        v3_values._pvp_lobby(state, flow, now=0.5)
        return state

    def test_opp_name_is_the_real_opponent(self) -> None:
        state = self._state("BotHunter")
        self.assertEqual(state.texts["opp_name"], "BotHunter")
        self.assertEqual(state.texts["opp_initial"], "B")

    def test_opp_name_falls_back_to_opponent_when_empty(self) -> None:
        state = self._state("")
        self.assertEqual(state.texts["opp_name"], "Opponent")
        self.assertEqual(state.texts["opp_initial"], "O")

    def test_never_shows_the_old_hardcoded_player_2(self) -> None:
        state = self._state("Alice")
        self.assertNotEqual(state.texts["opp_name"], "Player 2")


# ---- Fix 2: "splitting layers" / "waiting for" only render on PVP_FOUND ---

class PvpLobbyMessageTest(unittest.TestCase):
    """``_pvp_lobby_message`` replaces the one-shot ``F.say`` (30 s timed
    toast, drawn on every screen via the generic ``_draw_toast`` mechanism)
    with state recomputed fresh every frame from the app's own booleans, so
    it can never survive a screen change the way a timed toast can."""

    def _app(self, screen, *, baking, ready_sent, started):
        app = HcpdApplication.__new__(HcpdApplication)
        app._album = SimpleNamespace(
            flow=F, state=F.V3Flow(screen=screen, pvp_opponent_name="Bob",
                                   pvp_started=started))
        app._renderer = SimpleNamespace(waiting_for_depth=baking)
        app._pvp_ready_sent = ready_sent
        return app

    def test_baking_shows_splitting_layers(self) -> None:
        app = self._app(F.PVP_FOUND, baking=True, ready_sent=False, started=False)
        self.assertEqual(app._pvp_lobby_message(), "Splitting layers…")

    def test_ready_but_not_started_shows_waiting_for_name(self) -> None:
        app = self._app(F.PVP_FOUND, baking=False, ready_sent=True, started=False)
        self.assertEqual(app._pvp_lobby_message(), "Waiting for Bob…")

    def test_started_shows_nothing(self) -> None:
        app = self._app(F.PVP_FOUND, baking=False, ready_sent=True, started=True)
        self.assertEqual(app._pvp_lobby_message(), "")

    def test_neither_baking_nor_ready_shows_nothing(self) -> None:
        app = self._app(F.PVP_FOUND, baking=False, ready_sent=False, started=False)
        self.assertEqual(app._pvp_lobby_message(), "")

    def test_never_rendered_outside_pvp_found(self) -> None:
        for screen in (F.PVP_HUNT, F.PVP_HIDE, F.PVP_RESULT, F.PVP_MODE):
            with self.subTest(screen=screen):
                app = self._app(screen, baking=True, ready_sent=False, started=False)
                self.assertEqual(app._pvp_lobby_message(), "")


class PvpValuesRendersTheComputedLobbyMessageTest(unittest.TestCase):
    """``v3_values._pvp_lobby`` must show ``flow.pvp_lobby_message`` on
    ``PVP_FOUND`` when set, and never invent its own text."""

    def _state(self, pvp_lobby_message: str):
        flow = F.V3Flow()
        flow.screen = F.PVP_FOUND
        flow.player = "me"
        flow.found_at_s = 0.0
        flow.pvp_lobby_message = pvp_lobby_message
        state = SimpleNamespace(texts={}, focus=None, message="")
        v3_values._pvp_lobby(state, flow, now=0.5)
        return state

    def test_shows_the_computed_message(self) -> None:
        state = self._state("Splitting layers…")
        self.assertEqual(state.message, "Splitting layers…")

    def test_empty_computed_message_leaves_state_message_untouched(self) -> None:
        """Empty means "nothing to add" — must not clobber another toast
        (e.g. "This photo can't hide any bugs for PvP") already placed into
        state.message by state_for()'s generic F.say-driven assignment."""
        state = SimpleNamespace(texts={}, focus=None, message="some other toast")
        flow = F.V3Flow()
        flow.screen = F.PVP_FOUND
        flow.player = "me"
        flow.found_at_s = 0.0
        flow.pvp_lobby_message = ""
        v3_values._pvp_lobby(state, flow, now=0.5)
        self.assertEqual(state.message, "some other toast")


# ---- Fix 3: no yellow debug spheres handed to the renderer in PvP ---------

class PvpNeverSendsDebugSpheresToTheRendererTest(unittest.TestCase):
    """``_renderer.set_props`` with non-empty content is solo's debug-sphere
    path (``prop_game.prop_items``); PvP draws real sprites via
    ``BugPainter`` instead and must always pass ``set_props([])``."""

    class _Renderer:
        def __init__(self) -> None:
            self.props_calls: list[list] = []
            self.layers_revision = 1

        def set_props(self, props) -> None:
            self.props_calls.append(list(props))

        def layer_records(self):
            return []

        def set_prop_found(self, index: int) -> None:
            pass

    def _app(self):
        app = HcpdApplication.__new__(HcpdApplication)
        app._root = None
        app._renderer = self._Renderer()
        app._bugs = SimpleNamespace(reset=lambda: None)
        app._content = SimpleNamespace(game_prop_radius_mm=6.0,
                                       overscan_design_eye_mm=(0.0, 85.0))
        app._album = SimpleNamespace(game=None, flow=F, state=F.V3Flow())
        app._game = None
        app._game_planes = ()
        app._game_layers_revision = 0
        app._pvp_active = False
        return app

    def _fake_record(self, z_min_m, z_max_m, quad_w_m=0.3, quad_h_m=0.2,
                     centre=(0.0, 0.0)):
        import numpy as np

        return SimpleNamespace(alpha=np.zeros((4, 4), dtype=np.uint8),
                               z_world_m=(z_min_m + z_max_m) / 2.0,
                               quad_w_m=quad_w_m, quad_h_m=quad_h_m,
                               centre_x_m=centre[0], centre_y_m=centre[1],
                               relief_z_min_m=z_min_m, relief_z_max_m=z_max_m)

    def test_pvp_handle_start_never_sends_non_empty_props(self) -> None:
        from src.content.hidden_props import layer_planes
        from src.ui_album import pvp_sync

        app = self._app()
        records = [self._fake_record(-0.60, -0.50), self._fake_record(-0.20, 0.0)]
        app._renderer.layer_records = lambda: records
        app._content.content_path = "photo.jpg"
        app._album.state.level = 1
        app._album.state.weapon = 0
        planes = layer_planes(records)
        props = [{"pos_mm": (0.0, 0.0, -100.0), "radius_mm": 6.0,
                 "size_k": 1.0, "camo_rgb": (0.5, 0.5, 0.5)}]
        wire = pvp_sync.bugs_to_wire(props, planes)
        self.assertTrue(wire, "fixture bug didn't anchor to a gap")
        app._pvp_handle_start({"bugs": wire, "duration_s": 180}, 0.0)
        for call in app._renderer.props_calls:
            self.assertEqual(call, [], "PvP must never send real geometry "
                                        "to the solo debug-sphere renderer")

    def test_host_placement_path_never_sends_non_empty_props(self) -> None:
        app = self._app()
        app._album.state.level = 1
        app._content.content_path = "photo.jpg"
        app._rest_eye = None
        app._tracking_off = False
        app._pvp = SimpleNamespace(send=lambda msg: None)

        def fake_start_prop_game(count=0):
            from src.content.prop_game import GameState

            app._game = GameState(image="t", props=[{
                "pos_mm": (0.0, 0.0, -200.0), "radius_mm": 6.0}],
                tracking_on=True, started_at_s=0.0)
            app._game_planes = ()
            # Mirrors the real _start_prop_game, which DOES send real
            # geometry mid-call (same path solo uses) — the regression
            # this test guards is that something AFTER this still clears
            # it with set_props([]), not that it's never called.
            from src.content.prop_game import prop_items
            app._renderer.set_props(prop_items(app._game, 6.0))

        app._start_prop_game = fake_start_prop_game
        app._viewport_rect = lambda screen=None: (0.0, 0.0, 1280.0, 720.0)
        app._glass_mm = lambda: ((344.0, 194.0), (0.0, 0.0))
        app._photo_rgb_for_camo = lambda: None

        app._pvp_place_bugs(0.0)
        self.assertEqual(app._renderer.props_calls[-1], [],
                         "the LAST set_props call must clear the debug "
                         "spheres _start_prop_game left behind")


if __name__ == "__main__":
    unittest.main()
