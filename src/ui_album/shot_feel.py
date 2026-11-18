"""
TASK_game_feel_1118 Phase 4: shooting feel for the match screens (``PLAY``,
``PVP_HUNT``) — crosshair kick, screen shake, hit marker, crit impact
frame. Pure Python (no GL); ``src/ui_album/v3.py`` draws what this module
computes, and ``app.py`` calls ``ShotFeel.on_shot`` once per shot, right
AFTER ``game_bridge.fire()`` has already decided the result — never
before, and nothing here feeds back into that result.

TASK_game_feel_fix_1118 Fix 6: the tracer (a muzzle-to-crosshair streak)
was removed per the user's own playtest note — deleted outright, not
disabled: its drawing, its ``WeaponFeel.tracer`` column, every constant
and the ``_Tracer`` class are gone, along with their tests.
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

#: TASK_game_feel_1118 P5: full-auto bloom — visual only, Heavy-mag gun
#: only. Each consecutive auto shot raises the crosshair's RESTING scale
#: (the kick spring's own ``.to``, so one spring does both jobs) by this
#: much, capped at this scale; releasing the trigger drops it straight
#: back to the normal resting scale and lets the same spring pull it in.
BLOOM_STEP = 0.04
BLOOM_CAP_SCALE = 1.3


@dataclass(frozen=True)
class WeaponFeel:
    shake_px: float
    shake_cap_px: float
    kick_scale: float


#: TASK_game_feel_1118 P4's own table. Heavy-mag is the only one whose
#: shake ADDS UP across consecutive shots (full-auto, Phase 5) rather than
#: just being capped at its own single-shot amount — Precision/Spread
#: fire slowly enough that one shot's shake has mostly decayed before the
#: next, so their cap is simply their own per-shot amount.
WEAPON_FEEL = {
    "Precision gun": WeaponFeel(shake_px=3.0, shake_cap_px=3.0, kick_scale=1.35),
    "Spread gun": WeaponFeel(shake_px=4.0, shake_cap_px=4.0, kick_scale=1.40),
    "Heavy-mag gun": WeaponFeel(shake_px=1.5, shake_cap_px=3.0, kick_scale=1.15),
}

#: Fallback for an unrecognised weapon name — keeps ``on_shot`` from
#: raising on a stale name; the three real weapons above never hit it.
_DEFAULT_FEEL = WeaponFeel(shake_px=3.0, shake_cap_px=3.0, kick_scale=1.3)


def weapon_feel(name: str) -> WeaponFeel:
    """The shake/kick spec for ``name`` (one of ``v3_flow.WEAPONS``)."""
    return WEAPON_FEEL.get(name, _DEFAULT_FEEL)


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
        self._hit_markers: list[_HitMarker] = []
        self._crit_impact_age: float | None = None
        #: ``"screen_shake": false`` in prefs, or ``HCPD_NO_SHAKE=1`` —
        #: the caller sets this once. Disables shake AND the crit impact
        #: frame; kick and hit marker are unaffected (P4's own off-switch
        #: rule).
        self.shake_enabled = True
        #: P5: current bloom SCALE (not a delta) — the kick spring's
        #: resting point while the Heavy-mag trigger is held.
        self._bloom_scale = KICK_REST_SCALE

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
        if hit:
            x, y = float(crosshair[0]), float(crosshair[1])
            self._hit_markers.append(_HitMarker(x, y, bool(crit)))
            if crit and self.shake_enabled:
                self._crit_impact_age = 0.0

    # ---- full-auto bloom (TASK_game_feel_1118 P5, Heavy-mag gun only) -----
    def add_bloom(self) -> None:
        """Call once per consecutive auto shot while the trigger is held."""
        self._bloom_scale = min(BLOOM_CAP_SCALE, self._bloom_scale + BLOOM_STEP)
        self._kick.to = self._bloom_scale

    def release_bloom(self) -> None:
        """Call once the trigger is released — drops straight back to the
        normal resting scale; the kick spring pulls the displayed scale
        in from there, same as it does after any ordinary shot."""
        self._bloom_scale = KICK_REST_SCALE
        self._kick.to = KICK_REST_SCALE

    # ---- per-frame -----------------------------------------------------
    def step(self, dt: float) -> None:
        if not dt > 0.0:
            return
        self._kick.step(dt)
        decay = math.exp(-dt / SHAKE_TAU_S)
        self._shake_x *= decay
        self._shake_y *= decay
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
    def hit_markers(self) -> list:
        """``[(x, y, age, crit), ...]`` for drawing."""
        return [(m.x, m.y, m.age, m.crit) for m in self._hit_markers]

    @property
    def crit_impact_alpha(self) -> float:
        return CRIT_IMPACT_ALPHA if self._crit_impact_age is not None else 0.0
