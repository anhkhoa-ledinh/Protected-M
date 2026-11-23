import unittest
from unittest import mock

import cv2

from src.camera import capture
from src.camera.capture import CameraCapture, CaptureConfig


class FakeCap:
    def __init__(self, *_args) -> None:
        self.calls: list[int] = []

    def isOpened(self) -> bool:
        return True

    def set(self, prop: int, _value: float) -> bool:
        self.calls.append(prop)
        return True

    def get(self, _prop: int) -> float:
        return 0.0

    def release(self) -> None:
        pass


def _open_with(platform: str) -> list[int]:
    fake = FakeCap()
    with mock.patch.object(capture.cv2, "VideoCapture", return_value=fake), \
            mock.patch.object(capture.sys, "platform", platform):
        CameraCapture(CaptureConfig()).open()
    return fake.calls


class FourccOrderTests(unittest.TestCase):
    def test_linux_sets_pixel_format_before_size_and_fps(self) -> None:
        calls = _open_with("linux")
        self.assertLess(calls.index(cv2.CAP_PROP_FOURCC),
                        calls.index(cv2.CAP_PROP_FRAME_WIDTH))
        self.assertLess(calls.index(cv2.CAP_PROP_FOURCC),
                        calls.index(cv2.CAP_PROP_FPS))

    def test_windows_order_is_unchanged(self) -> None:
        calls = _open_with("win32")
        self.assertEqual(calls[-1], cv2.CAP_PROP_FOURCC)
        self.assertEqual(calls[:4], [cv2.CAP_PROP_BUFFERSIZE, cv2.CAP_PROP_FRAME_WIDTH,
                                     cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS])


if __name__ == "__main__":
    unittest.main()
