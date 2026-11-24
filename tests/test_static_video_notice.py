"""FR-DEPTH-03: báo rõ video ở chế độ lớp chỉ lấy depth từ khung đầu."""

from __future__ import annotations

import unittest

from src.renderer.content_plane import (
    STATIC_VIDEO_NOTICE,
    ContentPlaneRenderer,
    static_video_notice,
)


class TestStaticVideoNotice(unittest.TestCase):
    """
    Chỉ báo khi **vừa** là video **vừa** đang ở chế độ lớp.

    Ảnh thì không có gì đứng yên để phải báo; video ở chế độ mặt phẳng vẫn
    chạy bình thường (depth dùng cho displacement, không tách lớp).
    """

    def test_video_in_layer_mode_is_reported(self) -> None:
        self.assertEqual(static_video_notice("video", True), STATIC_VIDEO_NOTICE)

    def test_wording_names_the_first_frame(self) -> None:
        # Tiêu chí FR-DEPTH-03: log/HUD phải ghi "video: depth khung đầu".
        self.assertIn("first frame", STATIC_VIDEO_NOTICE.lower())
        self.assertIn("video", STATIC_VIDEO_NOTICE.lower())

    def test_video_without_layer_mode_is_silent(self) -> None:
        self.assertIsNone(static_video_notice("video", False))

    def test_image_is_silent(self) -> None:
        self.assertIsNone(static_video_notice("image", True))
        self.assertIsNone(static_video_notice("image", False))


class TestNoticeReachesTheToast(unittest.TestCase):
    """
    Nạp file xong thì thông báo phải đi vào **toast**, không chỉ vào log.

    Dựng app bằng ``__new__`` với renderer giả, cùng cách ``test_demo_mode``
    làm, để không cần cửa sổ hay context OpenGL.
    """

    def _app(self, notice):
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        from src.engine.app import HcpdApplication

        app = HcpdApplication.__new__(HcpdApplication)
        # demo = True để _try_load_media không ghi file config nào.
        app._demo, app._root = True, Path(tempfile.gettempdir())
        self.toasts: list[str] = []
        app._content = SimpleNamespace(content_path="", content_type="", mode="")
        app._renderer = SimpleNamespace(
            load_content_file=lambda p, label=None: (True, f"Loaded: {label}"),
            show_toast=lambda m, **k: self.toasts.append(m),
            set_props=lambda items: None,
            take_content_notice=lambda: notice)
        app._game, app._game_planes, app._watch_until_s = None, (), None
        return app

    def _load(self, notice):
        from pathlib import Path
        from unittest import mock

        app = self._app(notice)
        clip = Path(__file__).resolve().parents[1] / "assets" / "images" / "test_pattern.jpg"
        with mock.patch.object(app, "_abandon_game", lambda *a, **k: None):
            app._try_load_media(clip, 0.0)
        return self.toasts

    def test_toast_carries_the_notice(self) -> None:
        toasts = self._load(STATIC_VIDEO_NOTICE)
        self.assertEqual(len(toasts), 1, toasts)
        self.assertIn(STATIC_VIDEO_NOTICE, toasts[0])

    def test_toast_unchanged_when_there_is_no_notice(self) -> None:
        toasts = self._load(None)
        self.assertEqual(toasts, ["Loaded: test_pattern.jpg"])


class TestPendingNoticeIsOneShot(unittest.TestCase):
    """Toast chỉ hiện MỘT lần: lấy xong thì hết, không nhắc mãi mỗi khung."""

    def test_notice_is_taken_once(self) -> None:
        # Không dựng cả renderer (cần context OpenGL); chỉ cần đúng thuộc tính
        # mà take_pending_notice đọc.
        plane = ContentPlaneRenderer.__new__(ContentPlaneRenderer)
        plane._pending_notice = STATIC_VIDEO_NOTICE
        self.assertEqual(plane.take_pending_notice(), STATIC_VIDEO_NOTICE)
        self.assertIsNone(plane.take_pending_notice())


if __name__ == "__main__":
    unittest.main()
