"""TASK_pvp_lan_1116 P2 — ``--no-webcam``.

Two laptops are one machine during a local PvP test (only one process can
open the real webcam), so the second game instance must run the HCPD
pipeline without ever opening a device. ``TrackingWorker`` already falls
back to a simulated pose when ``CameraWorker.pop_frame()`` returns ``None``
(``_hold_or_simulate``/``_simulated_pose`` in ``tracking_worker.py``) — so
``--no-webcam`` only has to guarantee CameraWorker is never started, not
reimplement the fallback.
"""

from __future__ import annotations

import unittest

import main
from src.camera.capture import CaptureConfig
from src.engine.app import HcpdApplication
from src.engine.camera_worker import CameraWorker


class NoWebcamCliTest(unittest.TestCase):
    def test_flag_defaults_off(self) -> None:
        self.assertFalse(main.parse_args([]).no_webcam)

    def test_flag_parses(self) -> None:
        self.assertTrue(main.parse_args(["--no-webcam"]).no_webcam)


class NoWebcamAppDefaultTest(unittest.TestCase):
    def test_class_default_is_real_webcam(self) -> None:
        # Instances built via __new__ (every other test in this suite) must
        # keep today's behaviour unless __init__ says otherwise.
        self.assertFalse(HcpdApplication._no_webcam)


class CameraWorkerNeverStartedTest(unittest.TestCase):
    """The contract ``--no-webcam`` relies on: no ``start()`` -> no frames."""

    def test_pop_and_peek_are_none_without_start(self) -> None:
        worker = CameraWorker(CaptureConfig())
        self.assertIsNone(worker.pop_frame())
        self.assertIsNone(worker.peek_frame())

    def test_stop_without_start_does_not_raise(self) -> None:
        CameraWorker(CaptureConfig()).stop()


if __name__ == "__main__":
    unittest.main()
