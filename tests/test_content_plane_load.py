"""Tests for content plane media loading (no OpenGL required)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from src.renderer.video_player import VideoPlayer


class TestVideoPlayerPrime(unittest.TestCase):
    def test_prime_first_frame_reads_without_dt(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "color.mp4"
            frame = np.zeros((120, 160, 3), dtype=np.uint8)
            frame[:, :80] = (255, 0, 0)
            frame[:, 80:] = (0, 255, 0)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (160, 120),
            )
            for _ in range(3):
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            player = VideoPlayer(path, loop=True)
            self.assertFalse(player.is_procedural)
            rgb = player.prime_first_frame()
            self.assertIsNotNone(rgb)
            assert rgb is not None
            self.assertEqual(rgb.shape[:2], (120, 160))
            self.assertGreater(int(rgb[0, 0, 0]), 200)
            player.close()

    def test_tick_returns_first_frame_immediately(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "one.mp4"
            frame = np.full((64, 64, 3), (0, 0, 255), dtype=np.uint8)
            writer = cv2.VideoWriter(
                str(path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                10.0,
                (64, 64),
            )
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            writer.release()

            player = VideoPlayer(path)
            rgb = player.tick(0.0)
            self.assertIsNotNone(rgb)
            player.close()


if __name__ == "__main__":
    unittest.main()
