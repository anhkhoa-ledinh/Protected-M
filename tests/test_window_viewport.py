"""Kích thước viewport phải theo framebuffer, không theo config (PB-038)."""

from __future__ import annotations

import unittest
from types import SimpleNamespace

import glfw

from src.config.schema import RendererSettings
from src.renderer.window import GlfwWindow, fullscreen_target, viewport_size


def _video_mode(width: int, height: int, refresh: int = 40):
    """Giả lập ``GLFWvidmode``; màn của máy demo là 1366x768 @ 8/8/8, 40 Hz."""
    return SimpleNamespace(
        size=SimpleNamespace(width=width, height=height),
        bits=SimpleNamespace(red=8, green=8, blue=8),
        refresh_rate=refresh,
    )


class ViewportSizeTest(unittest.TestCase):
    """
    Trước bản sửa, ``GlfwWindow.width/height`` trả thẳng
    ``settings.window_width/height`` và không bao giờ hỏi GLFW, nên
    ``glViewport`` ở ``renderer.py:214`` chạy theo con số cấu hình. Cũng không
    có resize callback nào, nên cửa sổ đổi cỡ thì viewport đứng im mãi mãi.

    Đo 2026-10-11 trên màn 1366x768: client area **1304x744** nhưng cảnh vẽ
    **1280x720** neo góc dưới-trái, chừa dải đen 24 px ở trên và bên phải.
    """

    def test_framebuffer_size_wins_over_config(self) -> None:
        self.assertEqual(viewport_size((1304, 744), (1280, 720)), (1304, 744))

    def test_width_and_height_follow_a_resize(self) -> None:
        """
        Dựng ``GlfwWindow`` mà **không** gọi ``create()`` — ``__init__`` không
        đụng GLFW hay OpenGL — rồi gọi phần ghi nhận kích thước, vốn tách riêng
        khỏi lời gọi ``glViewport`` đúng để test được không cần context.
        """
        window = GlfwWindow(RendererSettings(window_width=1280, window_height=720))
        self.assertEqual((window.width, window.height), (1280, 720))

        window._set_framebuffer_size(1304, 744)
        self.assertEqual((window.width, window.height), (1304, 744))

        window._set_framebuffer_size(1920, 1080)
        self.assertEqual((window.width, window.height), (1920, 1080))


class FullscreenTargetTest(unittest.TestCase):
    """
    Cờ ``fullscreen`` quyết định truyền màn nào vào ``glfw.create_window``
    (PB-042). Trước bản sửa, đối số monitor luôn là ``None`` (window.py:93) nên
    app không bao giờ chạy toàn màn hình; maximize chỉ cho 1304x744 và vẫn chừa
    taskbar, trong khi Kooima giả định viewport phủ đúng 344x194 mm.
    """

    MONITOR = object()

    def test_disabled_keeps_the_windowed_size_and_no_monitor(self) -> None:
        monitor, size, hints = fullscreen_target(
            False, self.MONITOR, _video_mode(1366, 768), (1280, 720)
        )
        self.assertIsNone(monitor, "tắt cờ thì không được truyền monitor")
        self.assertEqual(size, (1280, 720))
        self.assertEqual(hints, {})

    def test_enabled_takes_the_monitor_current_video_mode(self) -> None:
        monitor, size, hints = fullscreen_target(
            True, self.MONITOR, _video_mode(1366, 768), (1280, 720)
        )
        self.assertIs(monitor, self.MONITOR)
        self.assertEqual(size, (1366, 768), "phải theo chế độ đang chạy của màn")
        # Hint khớp chế độ hiện tại = GLFW không đổi độ phân giải, chỉ bỏ viền.
        self.assertEqual(
            hints,
            {
                glfw.RED_BITS: 8,
                glfw.GREEN_BITS: 8,
                glfw.BLUE_BITS: 8,
                glfw.REFRESH_RATE: 40,
            },
        )


if __name__ == "__main__":
    unittest.main()
