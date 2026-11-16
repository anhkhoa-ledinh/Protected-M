"""TASK_pvp_lan_1116 P3: ``src/engine/app.py``'s PvP message handlers —
pure-logic tests via ``HcpdApplication.__new__`` + stubs, the same pattern
``test_abandon_match.py``/``test_hit_box.py`` already use. No GL, no real
sockets (the real-socket path is covered end to end by
``tests/test_pvp_bot.py``)."""

from __future__ import annotations

import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from src.content.prop_game import GameState
from src.engine.app import HcpdApplication
from src.net import protocol as proto
from src.ui_album import hit as H
from src.ui_album import v3_flow as F
from src.ui_album.game_bridge import AlbumMatch


class _Renderer:
    def __init__(self) -> None:
        self.props = None
        self.found: list[int] = []
        self.toasts: list[str] = []
        self.waiting_for_depth = False
        self.layers_revision = 1

    def set_props(self, props) -> None:
        self.props = props

    def set_prop_found(self, index: int) -> None:
        self.found.append(index)

    def layer_records(self):
        return []

    def show_toast(self, text, is_error: bool = False) -> None:
        self.toasts.append(text)


class _BugsPainter:
    def __init__(self) -> None:
        self.reset_calls = 0
        self.marked: list[tuple] = []

    def reset(self) -> None:
        self.reset_calls += 1

    def mark_found(self, index, pos_m) -> None:
        self.marked.append((index, pos_m))

    def sprite_mask(self, level):
        return object()


class _FakePvp:
    def __init__(self) -> None:
        self.sent: list[dict] = []
        self.is_connected = True
        self.closed = False

    def send(self, msg) -> None:
        self.sent.append(msg)

    def poll(self):
        return []

    def close(self) -> None:
        self.closed = True


def _app(**extra) -> HcpdApplication:
    app = HcpdApplication.__new__(HcpdApplication)
    app._root = Path(".")
    app._renderer = _Renderer()
    app._bugs = _BugsPainter()
    app._content = SimpleNamespace(game_prop_radius_mm=6.0,
                                   overscan_design_eye_mm=(0.0, 85.0))
    app._album = SimpleNamespace(game=None, flow=F, state=F.V3Flow())
    app._game = None
    app._game_planes = ()
    app._game_layers_revision = 0
    app._pvp = _FakePvp()
    app._pvp_pending = None
    app._pvp_my_name = "Alice"
    app._pvp_is_host = False
    app._pvp_photo_sent = False
    app._pvp_bugs_sent = False
    app._pvp_bugs_attempted = False
    app._pvp_ready_sent = False
    app._pvp_active = False
    for key, value in extra.items():
        setattr(app, key, value)
    return app


def _fake_record(z_min_m, z_max_m, quad_w_m=0.3, quad_h_m=0.2, centre=(0.0, 0.0)):
    """Shape ``layer_planes()`` reads off a renderer's ``_LayerRenderable``."""
    return SimpleNamespace(alpha=np.zeros((4, 4), dtype=np.uint8),
                           z_world_m=(z_min_m + z_max_m) / 2.0,
                           quad_w_m=quad_w_m, quad_h_m=quad_h_m,
                           centre_x_m=centre[0], centre_y_m=centre[1],
                           relief_z_min_m=z_min_m, relief_z_max_m=z_max_m)


# ---- _pvp_handle_photo -----------------------------------------------------

class PvpHandlePhotoTest(unittest.TestCase):
    def test_saves_under_cache_pvp_named_by_the_locally_recomputed_sha1(self) -> None:
        import base64
        import hashlib
        import tempfile

        data = b"\xff\xd8\xff fake jpeg bytes"
        sha1 = hashlib.sha1(data).hexdigest()
        with tempfile.TemporaryDirectory() as tmp:
            app = _app(_root=Path(tmp))
            app._try_load_media = lambda path, now: None
            app._pvp_handle_photo(
                {"jpeg_b64": base64.b64encode(data).decode("ascii"), "sha1": sha1},
                time.perf_counter())
            expected = Path(tmp) / "assets" / "cache" / "pvp" / f"{sha1}.jpg"
            self.assertTrue(expected.exists())
            self.assertEqual(expected.read_bytes(), data)
            self.assertEqual(str(app._album.state.photo), str(expected))

    def test_rejects_a_mismatched_sha1_and_writes_nothing(self) -> None:
        import base64
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            app = _app(_root=Path(tmp))
            app._try_load_media = lambda path, now: (_ for _ in ()).throw(
                AssertionError("must not bake a rejected photo"))
            app._pvp_handle_photo(
                {"jpeg_b64": base64.b64encode(b"some bytes").decode("ascii"),
                 "sha1": "0" * 40}, time.perf_counter())
            cache_dir = Path(tmp) / "assets" / "cache" / "pvp"
            self.assertFalse(cache_dir.exists() and list(cache_dir.iterdir()))

    def test_rejects_bad_base64_without_raising(self) -> None:
        app = _app()
        app._try_load_media = lambda path, now: (_ for _ in ()).throw(
            AssertionError("must not bake"))
        app._pvp_handle_photo({"jpeg_b64": "!!!not base64!!!", "sha1": "a" * 40},
                              time.perf_counter())  # must not raise

    def test_ignores_non_string_fields(self) -> None:
        app = _app()
        app._try_load_media = lambda path, now: (_ for _ in ()).throw(
            AssertionError("must not bake"))
        app._pvp_handle_photo({"jpeg_b64": 12345, "sha1": None}, time.perf_counter())


# ---- _pvp_handle_killed -----------------------------------------------------

class PvpHandleKilledTest(unittest.TestCase):
    def _match_app(self, found_before: bool = False):
        app = _app()
        app._game = GameState(image="pvp", props=[{"pos_mm": (0.0, 0.0, -300.0)}],
                              tracking_on=True, started_at_s=0.0,
                              found=[found_before])
        match = AlbumMatch(level=1, state=app._game, started_s=0.0)
        app._album.game = match
        return app, match

    def test_confirmed_kill_by_me_increments_my_kills_and_marks_found(self) -> None:
        app, match = self._match_app()
        app._pvp_handle_killed({"bug": 0, "by": "Alice", "late": False}, 1.0)
        self.assertTrue(app._game.found[0])
        self.assertEqual(match.my_kills, 1)
        self.assertEqual(match.bot_found, 0)
        self.assertEqual(app._renderer.found, [0])

    def test_confirmed_kill_by_opponent_increments_bot_found(self) -> None:
        app, match = self._match_app()
        app._pvp_handle_killed({"bug": 0, "by": "Bob", "late": False}, 1.0)
        self.assertTrue(app._game.found[0])
        self.assertEqual(match.bot_found, 1)
        self.assertEqual(match.my_kills, 0)

    def test_late_duplicate_changes_no_kill_counts(self) -> None:
        app, match = self._match_app(found_before=True)
        app._pvp_handle_killed({"bug": 0, "by": "Alice", "late": True}, 1.0)
        self.assertEqual(match.my_kills, 0)
        self.assertEqual(match.bot_found, 0)

    def test_already_found_locally_still_credits_the_confirmed_killer(self) -> None:
        """My own optimistic fire() already set found=True; the server's
        confirmation must still award the kill, without re-marking found."""
        app, match = self._match_app(found_before=True)
        app._pvp_handle_killed({"bug": 0, "by": "Alice", "late": False}, 1.0)
        self.assertEqual(match.my_kills, 1)
        self.assertEqual(app._renderer.found, [])  # not re-marked

    def test_unknown_bug_id_is_ignored(self) -> None:
        app, match = self._match_app()
        app._pvp_handle_killed({"bug": 99, "by": "Alice", "late": False}, 1.0)
        self.assertEqual(match.my_kills, 0)
        self.assertEqual(match.bot_found, 0)

    def test_no_active_match_is_a_no_op(self) -> None:
        app = _app()
        app._pvp_handle_killed({"bug": 0, "by": "Alice"}, 1.0)  # must not raise


# ---- _pvp_handle_tick -------------------------------------------------------

class PvpHandleTickTest(unittest.TestCase):
    def test_recomputes_started_s_so_local_time_left_matches_the_server(self) -> None:
        app = _app()
        state = GameState(image="pvp", props=[], tracking_on=True, started_at_s=0.0)
        match = AlbumMatch(level=1, state=state, started_s=time.perf_counter())
        app._album.game = match
        app._pvp_handle_tick({"time_left": 100.0}, time.perf_counter())
        self.assertAlmostEqual(match.time_left_s, 100.0, delta=0.1)

    def test_no_active_match_is_a_no_op(self) -> None:
        app = _app()
        app._pvp_handle_tick({"time_left": 10.0}, 1.0)  # must not raise


# ---- _pvp_handle_end --------------------------------------------------------

class PvpHandleEndTest(unittest.TestCase):
    def test_populates_last_result_and_goes_to_result_screen(self) -> None:
        app = _app()
        app._album.state.screen = F.PVP_HUNT
        app._album.game = object()
        app._game = object()
        msg = {"winner": "you", "scores": {"you": 42.0, "opp": 10.0},
              "rows": [["Bugs killed", "4 × 10"], ["Time left", 2]],
              "opponent_rows": [["Bugs killed", "1 × 10"], ["Time left", 2]],
              "opp_name": "Bob"}
        app._pvp_handle_end(msg, time.perf_counter())
        result = app._album.state.last_result
        self.assertEqual(result["score"], 42.0)
        self.assertEqual(result["opponent_score"], 10.0)
        self.assertEqual(result["winner"], "you")
        self.assertEqual(result["opp_name"], "Bob")
        self.assertEqual(result["rows"], msg["rows"])
        self.assertEqual(result["opponent_rows"], msg["opponent_rows"])
        self.assertEqual(app._album.state.screen, F.PVP_RESULT)
        self.assertIsNone(app._album.game)
        self.assertIsNone(app._game)
        self.assertFalse(app._pvp_active)


# ---- _pvp_handle_start -------------------------------------------------------

class PvpHandleStartTest(unittest.TestCase):
    def _app_with_planes(self) -> HcpdApplication:
        app = _app()
        records = [_fake_record(-0.60, -0.50), _fake_record(-0.40, -0.30),
                  _fake_record(-0.10, 0.0)]
        app._renderer.layer_records = lambda: records
        app._album.state.level = 1
        app._album.state.weapon = 0
        app._content.content_path = "photo.jpg"
        return app

    def test_builds_game_and_album_match_from_wire_bugs(self) -> None:
        from src.content.hidden_props import layer_planes
        from src.ui_album import pvp_sync

        app = self._app_with_planes()
        planes = layer_planes(app._renderer.layer_records())
        props = [{"pos_mm": (0.0, 0.0, -450.0), "radius_mm": 6.0, "size_k": 1.0,
                 "camo_rgb": (0.5, 0.5, 0.5)},
                {"pos_mm": (10.0, 5.0, -200.0), "radius_mm": 6.0, "size_k": 1.2,
                 "camo_rgb": (0.2, 0.3, 0.4)}]
        wire = pvp_sync.bugs_to_wire(props, planes)
        app._pvp_handle_start({"bugs": wire, "duration_s": 180},
                              time.perf_counter())
        self.assertTrue(app._pvp_active)
        self.assertTrue(app._album.state.pvp_started)
        match = app._album.game
        self.assertIsNotNone(match)
        self.assertEqual(match.bugs_total, 2)
        self.assertEqual(app._game.found, [False, False])
        self.assertIsNotNone(app._renderer.props)   # set_props() was called

    def test_empty_bugs_is_the_hunt_and_hide_setup_phase_not_a_match_start(self) -> None:
        app = self._app_with_planes()
        app._album.state.mode = F.PVP_MODES[1]
        app._album.state.pvp_role = "hider"
        app._pvp_handle_start({"bugs": [], "duration_s": 20}, time.perf_counter())
        self.assertFalse(app._pvp_active)
        self.assertFalse(app._album.state.pvp_started)
        self.assertGreater(app._album.state.pvp_setup_deadline_s, 0.0)
        self.assertIsNone(app._album.game)

    def test_a_bug_whose_layer_does_not_exist_locally_becomes_a_dead_placeholder(self) -> None:
        app = self._app_with_planes()
        wire = [{"id": 0, "x": 0.5, "y": 0.5, "z": 0.5, "layer": 99,
                "size_k": 1.0, "camo_rgb": [0.5, 0.5, 0.5]}]
        app._pvp_handle_start({"bugs": wire, "duration_s": 180},
                              time.perf_counter())
        self.assertEqual(app._game.found, [True])
        self.assertIsNotNone(app._game.props[0])  # placeholder dict, not None


# ---- _pvp_send_when_ready / _tick_pvp ---------------------------------------

class PvpSendWhenReadyTest(unittest.TestCase):
    def test_sends_immediately_when_already_connected(self) -> None:
        app = _app()
        app._pvp.is_connected = True
        msg = {"t": proto.QUEUE, "mode": "Competitive", "level": 1}
        app._pvp_send_when_ready(msg)
        self.assertEqual(app._pvp.sent, [msg])
        self.assertIsNone(app._pvp_pending)

    def test_queues_when_not_connected_yet_and_a_later_tick_flushes_it(self) -> None:
        app = _app()
        app._pvp.is_connected = False
        msg = {"t": proto.QUEUE, "mode": "Competitive", "level": 1}
        app._pvp_send_when_ready(msg)
        self.assertEqual(app._pvp.sent, [])
        self.assertEqual(app._pvp_pending, msg)
        app._pvp.is_connected = True
        app._tick_pvp(0.0)
        self.assertEqual(app._pvp.sent, [msg])
        self.assertIsNone(app._pvp_pending)


# ---- _album_shoot sends `hit` in an active PvP match ------------------------

_EYE = (0.0, 0.0, 600.0)
_GLASS = (344.0, 194.0)
_PROP = (0.0, 0.0, -180.0)
_RADIUS_MM = 12.0


class AlbumShootSendsHitWhenPvpActiveTest(unittest.TestCase):
    def test_a_confirmed_hit_sends_a_hit_message(self) -> None:
        mask = H.SpriteMask.from_rgba(np.full((16, 16, 4), 255, dtype=np.uint8))
        quad = H.sprite_quad(_PROP, _RADIUS_MM, 1.0, _EYE, _GLASS)
        app = _app()
        app._bugs = SimpleNamespace(sprite_mask=lambda level: mask,
                                    mark_found=lambda *a, **k: None)
        app._virtual_camera = SimpleNamespace(eye_position_mm=np.array(_EYE))
        app._glass_mm = lambda: (_GLASS, (0.0, 0.0))
        app._content = SimpleNamespace(game_prop_radius_mm=_RADIUS_MM,
                                       layer_alpha_threshold=0.05)
        app._album.state.level = 1
        app._album.state.screen = F.PVP_HUNT
        app._album.mouse = (quad.centre_x, quad.centre_y)
        app._game = GameState(image="pvp",
                              props=[{"pos_mm": _PROP, "radius_mm": _RADIUS_MM,
                                     "size_k": 1.0}],
                              tracking_on=True, started_at_s=0.0)
        app._game_planes = ()
        match = AlbumMatch(level=1, state=app._game, started_s=0.0)
        app._album.game = match
        app._pvp_active = True

        handled = app._album_shoot(time.perf_counter())

        self.assertTrue(handled)
        self.assertEqual(app._pvp.sent, [{"t": proto.HIT, "bug": 0}])

    def test_solo_play_never_sends_a_hit_message(self) -> None:
        mask = H.SpriteMask.from_rgba(np.full((16, 16, 4), 255, dtype=np.uint8))
        quad = H.sprite_quad(_PROP, _RADIUS_MM, 1.0, _EYE, _GLASS)
        app = _app()
        app._bugs = SimpleNamespace(sprite_mask=lambda level: mask,
                                    mark_found=lambda *a, **k: None)
        app._virtual_camera = SimpleNamespace(eye_position_mm=np.array(_EYE))
        app._glass_mm = lambda: (_GLASS, (0.0, 0.0))
        app._content = SimpleNamespace(game_prop_radius_mm=_RADIUS_MM,
                                       layer_alpha_threshold=0.05)
        app._album.state.level = 1
        app._album.state.screen = F.PLAY
        app._album.mouse = (quad.centre_x, quad.centre_y)
        app._game = GameState(image="solo",
                              props=[{"pos_mm": _PROP, "radius_mm": _RADIUS_MM,
                                     "size_k": 1.0}],
                              tracking_on=True, started_at_s=0.0)
        app._game_planes = ()
        match = AlbumMatch(level=1, state=app._game, started_s=0.0)
        app._album.game = match
        app._pvp_active = False          # solo: never set

        handled = app._album_shoot(time.perf_counter())

        self.assertTrue(handled)
        self.assertEqual(app._pvp.sent, [])


if __name__ == "__main__":
    unittest.main()
