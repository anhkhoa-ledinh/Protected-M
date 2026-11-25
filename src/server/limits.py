"""Per-IP connection counting and rate limiting with bounded memory."""

from __future__ import annotations

import time
from collections import deque

#: Idle IP entries are dropped once the table grows past this.
_MAX_TRACKED_IPS = 10_000


class IpLimiter:
    def __init__(self, max_per_ip: int, rate_per_s: float, burst: int,
                 connects_per_min: int, clock=time.monotonic) -> None:
        self._max_per_ip = max_per_ip
        self._rate = rate_per_s
        self._burst = burst
        self._connects_per_min = connects_per_min
        self._clock = clock
        self._open: dict[str, int] = {}
        self._bucket: dict[str, tuple[float, float]] = {}   # ip -> (tokens, t)
        self._connects: dict[str, deque] = {}
        self._last_prune = 0.0

    def allow_connect(self, ip: str) -> bool:
        """Reserve a connection slot for ``ip``; ``release`` must follow."""
        now = self._clock()
        if self._open.get(ip, 0) >= self._max_per_ip:
            return False
        recent = self._connects.setdefault(ip, deque())
        while recent and now - recent[0] > 60.0:
            recent.popleft()
        if len(recent) >= self._connects_per_min:
            return False
        recent.append(now)
        self._open[ip] = self._open.get(ip, 0) + 1
        self._prune(now)
        return True

    def release(self, ip: str) -> None:
        left = self._open.get(ip, 0) - 1
        if left > 0:
            self._open[ip] = left
        else:
            self._open.pop(ip, None)

    def allow_message(self, ip: str) -> bool:
        now = self._clock()
        tokens, last = self._bucket.get(ip, (float(self._burst), now))
        tokens = min(float(self._burst), tokens + (now - last) * self._rate)
        if tokens < 1.0:
            self._bucket[ip] = (tokens, now)
            return False
        self._bucket[ip] = (tokens - 1.0, now)
        return True

    def _prune(self, now: float) -> None:
        """Bound the tables. Scans at most once a second (an attacker with
        many source addresses must not turn every connect into a full scan);
        if idle entries are not old enough, the oldest are evicted anyway."""
        if len(self._bucket) + len(self._connects) < _MAX_TRACKED_IPS:
            return
        if now - self._last_prune < 1.0:
            return
        self._last_prune = now
        for ip in [i for i, q in self._connects.items()
                   if i not in self._open and (not q or now - q[-1] > 60.0)]:
            del self._connects[ip]
        for table in (self._connects, self._bucket):
            excess = len(table) - _MAX_TRACKED_IPS // 2
            if excess > 0:
                for ip in [i for i in table if i not in self._open][:excess]:
                    del table[ip]

    def tracked(self) -> int:
        return len(self._bucket) + len(self._connects) + len(self._open)
