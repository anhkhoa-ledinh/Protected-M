"""
TASK_game_feel_1118 Phase 4: shooting feel for the match screens (``PLAY``,
``PVP_HUNT``) — crosshair kick, tracer, screen shake, hit marker, crit
impact frame. Pure Python (no GL); ``src/ui_album/v3.py`` draws what this
module computes, and ``app.py`` calls ``ShotFeel.on_shot`` once per shot,
right AFTER ``game_bridge.fire()`` has already decided the result — never
before, and nothing here feeds back into that result.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from src.fx.anim import Spring

#: Crosshair kick: one spring, same stiffness for every weapon — only the
#: value it gets KICKED TO (the table's "Crosshair kick (scale)") differs.
KICK_SPRING_K, KICK_SPRING_D = 600.0, 30.0
KICK_REST_SCALE = 1.0

#: Tracer: muzzle = the HUD weapon icon centre (design px), fixed life and
#: target-end width for every weapon — only the muzzle-end width (and,
#: for the Spread gun, the line count/spread) is per-weapon.
MUZZLE_XY = (968.0, 620.0)
TRACER_LIFE_S = 0.07
TRACER_TARGET_WIDTH_PX = 1.0
TRACER_CORE_COL, TRACER_EDGE_COL = "#FFFFFF", "#24415F"
#: The white core sits inside the coloured edge — this fraction of the
#: edge's width at both ends.
TRACER_CORE_FRACTION = 0.55

#: Screen shake: exponential decay, time constant 0.08 s, same for every
#: weapon — only the per-shot amount (and Heavy-mag's cap) differs.
SHAKE_TAU_S = 0.08

#: Hit marker — weapon-independent.
HITMARK_INNER_PX, HITMARK_OUTER_PX = 10.0, 20.0
HITMARK_WIDTH_PX, HITMARK_OUTLINE_PX = 2.5, 1.5
HITMARK_LIFE_S = 0.18
HITMARK_COL, HITMARK_OUTLINE_COL = "#FFFFFF", "#24415F"
HITMARK_CRIT_COL = "#E8676B"
HITMARK_CRIT_SCALE = 1.3
HITMARK_ANGLES_DEG = (45.0, 135.0, 225.0, 315.0)

#: Crit impact frame — weapon-independent.
CRIT_IMPACT_S = 0.05
CRIT_IMPACT_COL = "#24415F"
CRIT_IMPACT_ALPHA = 0.55


@dataclass(frozen=True)
class TracerSpec:
    lines: int = 1
    muzzle_width_px: float = 3.0
    #: Spread gun only: the TARGET end of each extra line sits this many
    #: px either side of the crosshair, perpendicular to the shot line —
    #: the muzzle end of every line still meets at one point.
    spread_px: float = 0.0


@dataclass(frozen=True)
class WeaponFeel:
    shake_px: float
    shake_cap_px: float
    kick_scale: float
    tracer: TracerSpec


#: TASK_game_feel_1118 P4's own table. Heavy-mag is the only one whose
#: shake ADDS UP across consecutive shots (full-auto, Phase 5) rather than
#: just being capped at its own single-shot amount — Precision/Spread
#: fire slowly enough that one shot's shake has mostly decayed before the
#: next, so their cap is simply their own per-shot amount.
WEAPON_FEEL = {
    "Precision gun": WeaponFeel(
        shake_px=3.0, shake_cap_px=3.0, kick_scale=1.35,
        tracer=TracerSpec(lines=1, muzzle_width_px=2.5)),
    "Spread gun": WeaponFeel(
        shake_px=4.0, shake_cap_px=4.0, kick_scale=1.40,
        tracer=TracerSpec(lines=3, muzzle_width_px=3.0, spread_px=14.0)),
    "Heavy-mag gun": WeaponFeel(
        shake_px=1.5, shake_cap_px=3.0, kick_scale=1.15,
        tracer=TracerSpec(lines=1, muzzle_width_px=1.5)),
}

#: Fallback for an unrecognised weapon name — keeps ``on_shot`` from
#: raising on a stale name; the three real weapons above never hit it.
_DEFAULT_FEEL = WeaponFeel(shake_px=3.0, shake_cap_px=3.0, kick_scale=1.3,
                           tracer=TracerSpec())


def weapon_feel(name: str) -> WeaponFeel:
    """The shake/kick/tracer spec for ``name`` (one of ``v3_flow.WEAPONS``)."""
    return WEAPON_FEEL.get(name, _DEFAULT_FEEL)


@dataclass
class _Tracer:
    muzzle: tuple
    target: tuple
    width_px: float
    age: float = 0.0


@dataclass
class _HitMarker:
    x: float
    y: float
    crit: bool
    age: float = 0.0


class ShotFeel:
    """Per-session stateful "shot feel". One instance lives as long as the
    ``V3UI`` — ``app.py`` never resets it between matches, so a shot right
    after loading a new level behaves exactly like one mid-match.
    ``step()`` advances it with real dt every frame (``V3UI.tick``);
    ``on_shot()`` is called once per resolved shot, from ``app.py``'s
    ``_album_shoot`` — strictly AFTER ``game_bridge.fire()`` already
    decided the result, so nothing here can feed back into it.
    """

    def __init__(self, *, seed: int | None = 0) -> None:
        self._rng = random.Random(seed)
        self._kick = Spring(KICK_REST_SCALE, KICK_SPRING_K, KICK_SPRING_D)
        self._kick.to = KICK_REST_SCALE
        self._shake_x = 0.0
        self._shake_y = 0.0
        self._tracers: list[_Tracer] = []
        self._hit_markers: list[_HitMarker] = []
        self._crit_impact_age: float | None = None
        #: ``"screen_shake": false`` in prefs, or ``HCPD_NO_SHAKE=1`` —
        #: the caller sets this once. Disables shake AND the crit impact
        #: frame; kick, tracer and hit marker are unaffected (P4's own
        #: off-switch rule).
        self.shake_enabled = True

    # ---- per-shot ----------------------------------------------------------
    def on_shot(self, weapon_name: str, crosshair: tuple, *, hit: bool,
                crit: bool) -> None:
        feel = weapon_feel(weapon_name)
        self._kick.v = 0.0
        self._kick.x = feel.kick_scale
        if self.shake_enabled:
            angle = self._rng.uniform(0.0, 2.0 * math.pi)
            self._shake_x += feel.shake_px * math.cos(angle)
            self._shake_y += feel.shake_px * math.sin(angle)
            mag = math.hypot(self._shake_x, self._shake_y)
            if mag > feel.shake_cap_px > 0.0:
                scale = feel.shake_cap_px / mag
                self._shake_x *= scale
                self._shake_y *= scale
        self._spawn_tracers(feel.tracer, crosshair)
        if hit:
            x, y = float(crosshair[0]), float(crosshair[1])
            self._hit_markers.append(_HitMarker(x, y, bool(crit)))
            if crit and self.shake_enabled:
                self._crit_impact_age = 0.0

    def _spawn_tracers(self, tracer: TracerSpec, crosshair: tuple) -> None:
        mx, my = MUZZLE_XY
        tx, ty = float(crosshair[0]), float(crosshair[1])
        if tracer.lines <= 1 or tracer.spread_px <= 0.0:
            self._tracers.append(_Tracer((mx, my), (tx, ty), tracer.muzzle_width_px))
            return
        dx, dy = tx - mx, ty - my
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        half = (tracer.lines - 1) / 2.0
        for i in range(tracer.lines):
            offset = (i - half) * (tracer.spread_px / half)
            self._tracers.append(_Tracer(
                (mx, my), (tx + nx * offset, ty + ny * offset),
                tracer.muzzle_width_px))

    # ---- per-frame -----------------------------------------------------
    def step(self, dt: float) -> None:
        if not dt > 0.0:
            return
        self._kick.step(dt)
        decay = math.exp(-dt / SHAKE_TAU_S)
        self._shake_x *= decay
        self._shake_y *= decay
        for tracer in self._tracers:
            tracer.age += dt
        self._tracers = [t for t in self._tracers if t.age < TRACER_LIFE_S]
        for marker in self._hit_markers:
            marker.age += dt
        self._hit_markers = [m for m in self._hit_markers if m.age < HITMARK_LIFE_S]
        if self._crit_impact_age is not None:
            self._crit_impact_age += dt
            if self._crit_impact_age >= CRIT_IMPACT_S:
                self._crit_impact_age = None

    # ---- read for drawing ------------------------------------------------
    @property
    def kick_scale(self) -> float:
        return self._kick.x

    @property
    def shake_offset(self) -> tuple:
        return (self._shake_x, self._shake_y)

    @property
    def tracers(self) -> list:
        """``[(muzzle, target, age, muzzle_width_px), ...]`` for drawing."""
        return [(t.muzzle, t.target, t.age, t.width_px) for t in self._tracers]

    @property
    def hit_markers(self) -> list:
        """``[(x, y, age, crit), ...]`` for drawing."""
        return [(m.x, m.y, m.age, m.crit) for m in self._hit_markers]

    @property
    def crit_impact_alpha(self) -> float:
        return CRIT_IMPACT_ALPHA if self._crit_impact_age is not None else 0.0
