"""Thread-safe latest pose slot with age tracking."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np

from src.math.pose_utils import Pose6DoF


@dataclass(frozen=True)
class PoseSnapshot:
    """Pose sample with monotonic publish time."""

    pose: Pose6DoF
    published_at: float
    sequence: int

    def age_ms(self, now: float | None = None) -> float:
        """Milliseconds since this pose was published."""
        current = now if now is not None else time.perf_counter()
        return max(0.0, (current - self.published_at) * 1000.0)


class PoseBus:
    """Lock-protected slot for the latest head pose."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._sequence = 0
        self._snapshot = PoseSnapshot(
            pose=Pose6DoF(
                translation_mm=np.array([0.0, 0.0, 600.0], dtype=np.float64),
                rotation=np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64),
                timestamp=time.perf_counter(),
            ),
            published_at=time.perf_counter(),
            sequence=0,
        )

    def publish(self, pose: Pose6DoF) -> None:
        """Write the latest pose from tracking thread."""
        with self._lock:
            self._sequence += 1
            self._snapshot = PoseSnapshot(
                pose=pose,
                published_at=time.perf_counter(),
                sequence=self._sequence,
            )

    def latest(self) -> PoseSnapshot:
        """Read the latest pose snapshot from render thread."""
        with self._lock:
            return self._snapshot

    def pose_age_ms(self) -> float:
        """Age of the latest pose in milliseconds."""
        return self.latest().age_ms()
