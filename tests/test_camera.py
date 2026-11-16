"""Unit tests for camera timing and configuration."""

from __future__ import annotations

import time
import unittest

from src.camera.capture import CaptureConfig, FramePacket
from src.camera.timing import FrameTimer


class TestCaptureConfig(unittest.TestCase):
    """CaptureConfig serialization tests."""

    def test_round_trip_dict(self) -> None:
        config = CaptureConfig(device_id=1, width=640, height=480, fps=30)
        restored = CaptureConfig.from_dict(config.to_dict())
        self.assertEqual(restored, config)

    def test_defaults(self) -> None:
        config = CaptureConfig()
        self.assertEqual(config.device_id, 0)
        self.assertEqual(config.width, 1280)
        self.assertEqual(config.fps, 60)


class TestFrameTimer(unittest.TestCase):
    """FrameTimer statistical behavior."""

    def test_tick_computes_fps(self) -> None:
        timer = FrameTimer(window_size=10)
        for _ in range(5):
            timer.tick(capture_latency_ms=1.0)
            time.sleep(0.01)

        self.assertGreater(timer.stats.frame_count, 0)
        self.assertGreater(timer.stats.fps, 0.0)
        self.assertGreater(timer.stats.avg_capture_latency_ms, 0.0)

    def test_reset_clears_state(self) -> None:
        timer = FrameTimer()
        timer.tick(capture_latency_ms=2.0)
        timer.reset()
        self.assertEqual(timer.stats.frame_count, 0)
        self.assertEqual(timer.stats.fps, 0.0)


class TestFramePacket(unittest.TestCase):
    """FramePacket property accessors."""

    def test_dimensions(self) -> None:
        import numpy as np

        image = np.zeros((480, 640, 3), dtype=np.uint8)
        packet = FramePacket(
            image=image,
            frame_index=1,
            timestamp=0.0,
            capture_latency_ms=1.5,
        )
        self.assertEqual(packet.width, 640)
        self.assertEqual(packet.height, 480)
        self.assertEqual(packet.channels, 3)


if __name__ == "__main__":
    unittest.main()
