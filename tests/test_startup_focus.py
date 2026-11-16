"""
PB-051: khởi động không để cửa sổ "Not Responding", và báo đang chờ tách lớp.

Bằng chứng (16/10): run8 cửa sổ toàn màn bị thu nhỏ (0x0) từ ngay sau khởi
động tới khi có lớp; run12 dwm thay cửa sổ bằng bản ma "Not Responding" 0,3 s
sau khi nó xuất hiện. Không OpenGL: app dựng bằng ``__new__`` với đồ giả.
"""

from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


class StartupOrderTest(unittest.TestCase):
    """
    Đỏ trên code cũ: từ ``create()`` tới khung đầu không lấy message nào.

    Thứ tự cũ (cửa sổ trước camera) giữ nguyên: run13 thử mở camera trước thì
    phím không ăn.
    """

    def test_startup_pumps_messages_after_every_slow_step(self) -> None:
        from src.engine import app as app_mod

        calls = []

        def rec(name):
            return lambda *a, **k: calls.append(name)

        window = mock.MagicMock()
        window.create.side_effect = rec("window.create")
        window.pump_events.side_effect = rec("pump")
        window.should_close = True                    # không vào vòng vẽ
        camera, tracking, renderer = mock.MagicMock(), mock.MagicMock(), mock.MagicMock()
        camera.start.side_effect = rec("camera.start")
        tracking.start.side_effect = rec("tracking.start")
        renderer.initialize.side_effect = rec("renderer.initialize")

        app = app_mod.HcpdApplication.__new__(app_mod.HcpdApplication)
        app._window, app._camera, app._tracking = window, camera, tracking
        app._shader_dir = app._root = Path(".")
        app._virtual_camera = app._ui_settings = app._content = None
        app._game = app._renderer = None
        build = lambda *a, **k: calls.append("SceneRenderer") or renderer  # noqa: E731
        with mock.patch.object(app_mod, "SceneRenderer", side_effect=build), \
             mock.patch.object(app_mod.HcpdApplication, "_build_scene", lambda self: None):
            self.assertEqual(app.run(), 0)
        self.assertEqual(calls, ["window.create", "pump", "SceneRenderer", "pump",
                                 "renderer.initialize", "pump", "camera.start", "pump",
                                 "tracking.start", "pump"])


class WaitingForDepthTest(unittest.TestCase):
    """Chỉ "đang chờ" khi lớp của ĐÚNG ảnh hiện tại chưa có và worker còn sống."""

    def test_waiting_ends_when_this_image_has_layers_or_the_worker_failed(self) -> None:
        from src.content.depth_worker import DepthWorker
        from src.renderer import content_plane as cp

        r = cp.ContentPlaneRenderer.__new__(cp.ContentPlaneRenderer)
        r._content = SimpleNamespace(depth_enabled=True)
        r._depth_worker, r._bake_worker = SimpleNamespace(failed=False), None
        r._current_path = Path("t22.png").resolve()
        r._depth_mesh_key = None
        self.assertTrue(r.waiting_for_depth)
        r._depth_mesh_key = DepthWorker.media_key(r._current_path)
        self.assertFalse(r.waiting_for_depth)
        r._depth_mesh_key, r._depth_worker = None, SimpleNamespace(failed=True)
        self.assertFalse(r.waiting_for_depth)


class LayerWaitTickTest(unittest.TestCase):
    """Toast mỗi 2 s, một dòng log khi xong, hết toast sau trần; không nghỉ khung."""

    def test_toasts_every_two_seconds_and_logs_the_wait(self) -> None:
        from src.engine import app as app_mod

        toasts = []
        renderer = SimpleNamespace(waiting_for_depth=True,
                                   show_toast=lambda m, **k: toasts.append(m))
        app = app_mod.HcpdApplication.__new__(app_mod.HcpdApplication)
        app._renderer, app._layer_wait_since, app._layer_wait_toast_s = renderer, None, 0.0
        with mock.patch.object(app_mod.time, "sleep") as sleep, \
             self.assertLogs(app_mod.logger, "INFO") as logs:
            for now in (10.0, 11.0, 12.1, 13.0, 14.2):
                app._tick_layer_wait(now)
            renderer.waiting_for_depth = False
            app._tick_layer_wait(15.0)
            renderer.waiting_for_depth = True
            app._tick_layer_wait(100.0)
            app._tick_layer_wait(100.0 + app_mod.LAYER_WAIT_MAX_S + 1.0)
        self.assertEqual(toasts, ["Dang tach lop anh... 2 s", "Dang tach lop anh... 4 s"],
                         "sau tran LAYER_WAIT_MAX_S khong con toast")
        self.assertIn("Cho tach lop: 5.0 s", logs.output[0])
        sleep.assert_not_called()      # nghỉ 20 ms/khung đã bỏ (A/B chưa chứng minh)


if __name__ == "__main__":
    unittest.main()
