"""Reference player for baked Sakuga effects (pure Python, no GL): used for previews and as the spec the game port must match.

Word motion is a line-by-line port of core.js sfx(): a spring from 0.2 to 1 (k=520, d=17) that scales the word,
a jitter of sin(t*31)*0.8 / cos(t*27)*0.8 px, a wobble rot + sin(t*6)*0.03, and a shrink over the last 14 % of life.
"""
import math

class Spring:
    def __init__(self, x=0.0, k=300.0, d=20.0):
        self.x, self.v, self.to, self.k, self.d = x, 0.0, x, k, d

    def step(self, dt):
        if not dt > 0:
            return self.x
        n = max(1, math.ceil(dt / 0.008))
        h = dt / n
        for _ in range(n):
            self.v += (-self.k * (self.x - self.to) - self.d * self.v) * h
            self.x += self.v * h
        return self.x

def clamp(v, a=0.0, b=1.0):
    return min(b, max(a, v))

def inn(t):
    return t * t * t

class WordAnim:
    """One sound-effect word. size = width in design px of the traced word (not of the padded PNG)."""
    def __init__(self, traced_w, x, y, size, rot, life):
        self.s = Spring(0.2, 520, 17); self.s.to = 1.0
        self.k0 = size / traced_w
        self.x, self.y, self.rot, self.life = x, y, rot, life
        self.t = 0.0

    def step(self, dt):
        """Advance; returns (cx, cy, scale, angle_rad) in design px, or None when finished."""
        self.t += dt
        if self.t >= self.life:
            return None
        k = self.t / self.life
        sc = self.s.step(dt) * (1 - inn(clamp((k - 0.86) / 0.14))) * self.k0
        t = self.t
        return (self.x + math.sin(t * 31) * 0.8, self.y + math.cos(t * 27) * 0.8, sc, self.rot + math.sin(t * 6) * 0.03)
