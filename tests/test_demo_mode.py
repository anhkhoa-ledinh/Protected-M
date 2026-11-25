"""
PB-055: chế độ demo — chỉ G, T, F6, F4, Esc; đọc content.demo.json; không ghi config.

Quyết định 2026-10-11: sandbox giữ phím A/B và F1, demo thì không. Không OpenGL:
cửa sổ dựng bằng ``GlfwWindow(...)`` (hàm khởi tạo không gọi GLFW) và ``glfw``
được thay bằng đồ giả khi đọc phím.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import glfw

from src.config.schema import RendererSettings
from src.renderer import window as window_mod
from src.renderer.window import GlfwWindow

#: Mọi phím sandbox mà demo phải bỏ qua (F1, F2, F3, F7–F10, [ ] , . ; ' +/-).
BLOCKED = (glfw.KEY_F1, glfw.KEY_F2, glfw.KEY_F3, glfw.KEY_F7, glfw.KEY_F8,
           glfw.KEY_F9, glfw.KEY_F10, glfw.KEY_LEFT_BRACKET, glfw.KEY_RIGHT_BRACKET,
           glfw.KEY_COMMA, glfw.KEY_PERIOD, glfw.KEY_SEMICOLON, glfw.KEY_APOSTROPHE,
           glfw.KEY_EQUAL, glfw.KEY_MINUS, glfw.KEY_KP_ADD, glfw.KEY_KP_SUBTRACT)
#: Phím đang giữ khi đọc: D, W, Q (dịch pose) và Esc.
HELD = {glfw.KEY_D, glfw.KEY_W, glfw.KEY_Q, glfw.KEY_ESCAPE}


def _press(window, keys):
    """Bấm ``keys`` rồi đọc một khung, với D/W/Q/Esc đang được giữ."""
    for key in keys:
        window._on_key(None, key, 0, glfw.PRESS, 0)
    held = lambda _w, k: glfw.PRESS if k in HELD else glfw.RELEASE  # noqa: E731
    with mock.patch.object(window_mod.glfw, "poll_events"), \
         mock.patch.object(window_mod.glfw, "get_key", side_effect=held):
        return window.poll_input()


class DemoKeysTest(unittest.TestCase):
    """Đỏ trên code cũ: ``GlfwWindow`` không có chế độ demo, F1 luôn bật HUD."""

    def test_demo_keeps_only_g_t_f6_f4_and_esc(self) -> None:
        state = dataclasses.asdict(_press(GlfwWindow(RendererSettings(), demo=True), BLOCKED))
        on = [k for k, v in state.items() if v is True and k != "quit_app"]
        self.assertEqual(on, [], "phim sandbox van an trong demo")
        self.assertEqual((state["move_x"], state["move_y"], state["move_z"]), (0.0, 0.0, 0.0))
        self.assertTrue(state["quit_app"], "Esc phai con chay")
        allowed = _press(GlfwWindow(RendererSettings(), demo=True),
                         (glfw.KEY_G, glfw.KEY_T, glfw.KEY_F6, glfw.KEY_F4))
        self.assertTrue(allowed.game_toggle and allowed.toggle_tracking
                        and allowed.open_content_file and allowed.toggle_webcam_preview)

    def test_sandbox_keeps_f1_the_ab_keys_and_manual_pose(self) -> None:
        state = _press(GlfwWindow(RendererSettings()), (glfw.KEY_F1, glfw.KEY_F8))
        self.assertTrue(state.toggle_debug and state.toggle_atmospheric)
        self.assertEqual((state.move_x, state.move_y, state.move_z), (1.0, 1.0, 1.0))


class AlbumAloneKeysTest(unittest.TestCase):
    """TASK_account_persist_1121: ``--album`` alone (no ``--demo``) must lock
    the same research keys and manual pose nudge as ``--demo --album`` did
    before -- the game cannot depend on ``--demo`` for this any more."""

    def test_album_alone_keeps_only_player_keys_and_blocks_manual_pose(self) -> None:
        window = GlfwWindow(RendererSettings(), demo=False, album=True, ui_v3=True)
        state = dataclasses.asdict(_press(window, BLOCKED))
        # v3 doi nghia Esc thanh "lui mot buoc" (khong quit_app) -- bo qua
        # key nay, muc dich cua test la cac phim sandbox khac.
        on = [k for k, v in state.items()
              if v is True and k not in ("quit_app",)]
        self.assertEqual(on, [], "phim sandbox van an trong album")
        self.assertEqual((state["move_x"], state["move_y"], state["move_z"]), (0.0, 0.0, 0.0))

    def test_album_alone_blocks_wasd_qe_manual_pose(self) -> None:
        window = GlfwWindow(RendererSettings(), demo=False, album=True, ui_v3=True)
        state = _press(window, ())
        self.assertEqual((state.move_x, state.move_y, state.move_z), (0.0, 0.0, 0.0))


class DemoConfigTest(unittest.TestCase):
    """``--demo`` đọc content.demo.json; nạp ảnh (F6/kéo-thả) không ghi file config."""

    @staticmethod
    def _app(root, demo, album_enabled=False):
        from src.engine.app import HcpdApplication
        from src.ui_album.options import AlbumOptions

        app = HcpdApplication.__new__(HcpdApplication)
        app._demo, app._root = demo, root
        app._album_options = AlbumOptions(enabled=album_enabled)
        app._content = app._load_content(root)
        app._renderer = SimpleNamespace(
            load_content_file=lambda p, label=None: (True, f"Loaded: {label}"),
            show_toast=lambda *a, **k: None, set_props=lambda items: None,
            # FR-DEPTH-03: _try_load_media hỏi thông báo chờ sau mỗi lần nạp.
            take_content_notice=lambda: None)
        app._game, app._game_planes, app._watch_until_s = None, (), None
        return app

    def test_demo_reads_the_demo_file_and_writes_no_config(self) -> None:
        import main

        self.assertTrue(main.parse_args(["--demo"]).demo)
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            main.parse_args(["--demo", "--calibrate"])    # wizard ghi user.json
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "assets" / "config"
            cfg.mkdir(parents=True)
            for name, dd in (("content.json", 600), ("content.demo.json", 450)):
                (cfg / name).write_text(json.dumps({"overscan_design_distance_mm": dd}),
                                        encoding="utf-8")
            image = root / "pic.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")      # chỉ cần tồn tại, đuôi hợp lệ
            before = {p.name: p.read_bytes() for p in cfg.iterdir()}

            demo = self._app(root, True)
            self.assertEqual(demo._content.overscan_design_distance_mm, 450)
            demo._try_load_media(image, 0.0)
            self.assertEqual({p.name: p.read_bytes() for p in cfg.iterdir()}, before)

            sandbox = self._app(root, False)              # đối chứng: sandbox vẫn ghi
            self.assertEqual(sandbox._content.overscan_design_distance_mm, 600)
            sandbox._try_load_media(image, 0.0)
            self.assertNotEqual((cfg / "content.json").read_bytes(), before["content.json"])

    def test_album_alone_writes_no_research_config(self) -> None:
        """TASK_account_persist_1121: ``--album`` không ghi content.json, dù
        ``--demo`` có bật hay không -- ván chơi không cần nhớ ảnh mở qua
        file research này."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            cfg = root / "assets" / "config"
            cfg.mkdir(parents=True)
            (cfg / "content.json").write_text(
                json.dumps({"overscan_design_distance_mm": 600}), encoding="utf-8")
            image = root / "pic.png"
            image.write_bytes(b"\x89PNG\r\n\x1a\n")
            before = (cfg / "content.json").read_bytes()

            album = self._app(root, demo=False, album_enabled=True)
            album._try_load_media(image, 0.0)
            self.assertEqual((cfg / "content.json").read_bytes(), before)

            album_and_demo = self._app(root, demo=True, album_enabled=True)
            album_and_demo._try_load_media(image, 0.0)
            self.assertEqual((cfg / "content.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()
