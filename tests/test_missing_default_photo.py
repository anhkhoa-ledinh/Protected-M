import logging
import unittest
from pathlib import Path
from types import SimpleNamespace

from src.renderer.content_plane import ContentPlaneRenderer

MISSING = Path("D:\\PycharmProjects\\test_images\\no_such_photo.png")


class MissingPhotoLogLevelTests(unittest.TestCase):
    def test_startup_default_missing_is_not_an_error(self) -> None:
        with self.assertNoLogs("src.renderer.content_plane", level=logging.ERROR):
            ok, _msg = ContentPlaneRenderer.load_file(SimpleNamespace(), MISSING, strict=False)
        self.assertFalse(ok)

    def test_user_picked_missing_file_is_still_an_error(self) -> None:
        with self.assertLogs("src.renderer.content_plane", level=logging.ERROR):
            ok, _msg = ContentPlaneRenderer.load_file(SimpleNamespace(), MISSING, strict=True)
        self.assertFalse(ok)


if __name__ == "__main__":
    unittest.main()
