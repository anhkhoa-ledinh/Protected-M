"""
Spring physics and sound-effect-word motion — an EXACT port of
``tools/sakuga_bake/fxref.py`` (same constants, same substep). Copied, not
imported from ``tools/``: that directory is the bake tool's own reference
copy, kept untouched per TASK_ui_sakuga_1117's rules, and the game must
not depend on it at runtime.

Word motion is a line-by-line port of core.js ``sfx()``: a spring from 0.2
to 1 (k=520, d=17) that scales the word, a jitter of
``sin(t*31)*0.8`` / ``cos(t*27)*0.8`` px, a wobble
``rot + sin(t*6)*0.03``, and a shrink over the last 14% of life.
"""

from __future__ import annotations

import math


class Spring:
    def __init__(self, x: float = 0.0, k: float = 300.0, d: float = 20.0) -> None:
        self.x, self.v, self.to, self.k, self.d = x, 0.0, x, k, d

    def step(self, dt: float) -> float:
        if not dt > 0:
            return self.x
        n = max(1, math.ceil(dt / 0.008))
        h = dt / n
        for _ in range(n):
            self.v += (-self.k * (self.x - self.to) - self.d * self.v) * h
            self.x += self.v * h
        return self.x


def clamp(v: float, a: float = 0.0, b: float = 1.0) -> float:
    return min(b, max(a, v))


def inn(t: float) -> float:
    return t * t * t


class WordAnim:
    """One sound-effect word. ``size`` = width in design px of the TRACED
    word (not of the padded PNG)."""

    def __init__(self, traced_w: float, x: float, y: float, size: float,
                 rot: float, life: float) -> None:
        self.s = Spring(0.2, 520, 17)
        self.s.to = 1.0
        self.k0 = size / traced_w
        self.x, self.y, self.rot, self.life = x, y, rot, life
        self.t = 0.0

    def step(self, dt: float):
        """Advance; returns (cx, cy, scale, angle_rad) in design px, or
        None when finished."""
        self.t += dt
        if self.t >= self.life:
            return None
        k = self.t / self.life
        sc = self.s.step(dt) * (1 - inn(clamp((k - 0.86) / 0.14))) * self.k0
        t = self.t
        return (self.x + math.sin(t * 31) * 0.8, self.y + math.cos(t * 27) * 0.8,
                sc, self.rot + math.sin(t * 6) * 0.03)
