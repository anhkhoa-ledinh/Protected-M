"""
``FxPlayer`` — spawns, advances and lists draw commands for live Sakuga
effects. Pure Python, no GL: a "draw command" here is just a plain record
(texture key + four design-px corners + a uv rect + an alpha); turning
that into actual GL calls is ``src/ui_album/v3.py``'s/``app.py``'s job
(Phase 4).
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

from PIL import Image

from src.fx.anim import WordAnim
from src.fx.spec import FxSpec

#: Oldest effect is dropped when a spawn would exceed this many live effects.
MAX_LIVE_EFFECTS = 12


@dataclass(frozen=True)
class FxDrawCommand:
    """One textured quad to draw, in design px. ``corners`` is
    ``((x0,y0), (x1,y0), (x1,y1), (x0,y1))`` for an axis-aligned sprite
    frame, or four ROTATED points (same winding) for a word. ``uv`` is
    ``(u0, v0, u1, v1)`` within the texture named by ``texture_key``."""

    texture_key: str
    corners: tuple
    uv: tuple
    alpha: float = 1.0


class _ActiveWord:
    __slots__ = ("cue", "anim", "current")

    def __init__(self, cue: dict) -> None:
        self.cue = cue
        self.anim: WordAnim | None = None
        self.current = None  # (cx, cy, scale, angle) from the last step(), or None


class _ActiveEffect:
    __slots__ = ("name", "variant_index", "mirror", "x", "y", "words_enabled",
                 "age", "word_states")

    def __init__(self, name: str, variant_index: int, mirror: bool,
                 x: float, y: float, words_enabled: bool, cues: list) -> None:
        self.name = name
        self.variant_index = variant_index
        self.mirror = mirror
        self.x = x
        self.y = y
        self.words_enabled = words_enabled
        self.age = 0.0
        self.word_states = [_ActiveWord(cue) for cue in cues] if words_enabled else []


def _word_png_size(spec: FxSpec, word: str, style: str, cache: dict) -> tuple:
    key = (word, style)
    if key not in cache:
        with Image.open(spec.word_path(word, style)) as image:
            cache[key] = image.size
    return cache[key]


class FxPlayer:
    """One screen's worth of live effects. Construct a new one per
    screen-with-fx (or just ``clear()`` between screens) — it holds no GL
    state, only this pure-Python animation bookkeeping."""

    def __init__(self, spec: FxSpec, *, seed=None) -> None:
        self._spec = spec
        self._rng = random.Random(seed)
        self._effects: list[_ActiveEffect] = []
        self._word_png_size_cache: dict = {}

    # ---- spawning ----------------------------------------------------
    def spawn(self, name: str, x: float, y: float, *, mirror: bool | None = None,
              variant: int | None = None, words: bool = True) -> None:
        effect = self._spec.effects[name]
        variants = effect["variants"]
        if variant is None:
            variant_index = self._rng.randrange(len(variants))
        else:
            variant_index = variant
        if mirror is None:
            mirror = self._rng.random() < 0.5
        cues = effect.get("words") or [] if words else []
        active = _ActiveEffect(name, variant_index, bool(mirror), x, y,
                               words_enabled=words, cues=cues)
        self._append(active)

    def spawn_word(self, word: str, style: str, x: float, y: float, *,
                   size: float, rot: float = 0.0, life: float = 1.0) -> None:
        """Spawn a bare sound-effect word with no sprite behind it — for
        places the app triggers a word directly (the loading bar's
        "gogo"), not as part of a named effect's own baked-in cue list."""
        cue = {"word": word, "style": style, "at": 0.0, "dx": 0.0, "dy": 0.0,
              "size": size, "rot": rot, "life": life}
        active = _ActiveEffect(f"__word__:{word}:{style}", None, False, x, y,
                               words_enabled=True, cues=[cue])
        self._append(active)

    def _append(self, active: "_ActiveEffect") -> None:
        self._effects.append(active)
        if len(self._effects) > MAX_LIVE_EFFECTS:
            self._effects.pop(0)  # drop the oldest

    # ---- stepping ------------------------------------------------------
    def step(self, dt: float) -> None:
        for effect in self._effects:
            effect.age += dt
            for ws in effect.word_states:
                if ws.anim is None:
                    if effect.age >= ws.cue["at"]:
                        word, style = ws.cue["word"], ws.cue["style"]
                        traced_w = self._spec.words[word]["traced_w"]
                        ws.anim = WordAnim(
                            traced_w, effect.x + ws.cue["dx"], effect.y + ws.cue["dy"],
                            ws.cue["size"], ws.cue["rot"], ws.cue["life"])
                    else:
                        continue
                ws.current = ws.anim.step(dt)
        self._effects = [e for e in self._effects if not self._is_finished(e)]

    def _variant(self, effect: _ActiveEffect) -> dict | None:
        if effect.variant_index is None:
            return None
        return self._spec.effects[effect.name]["variants"][effect.variant_index]

    def _end_time(self, effect: _ActiveEffect) -> float:
        variant = self._variant(effect)
        frame_end = (len(variant["frames"]) / self._spec.fps) if variant is not None else 0.0
        if not effect.words_enabled:
            return frame_end
        if effect.variant_index is None:
            # Bare word: its own cue's life is the only lifetime there is.
            cues = [ws.cue for ws in effect.word_states]
        else:
            cues = self._spec.effects[effect.name].get("words") or []
        if not cues:
            return frame_end
        return max(frame_end, max(c["at"] + c["life"] for c in cues))

    def _is_finished(self, effect: _ActiveEffect) -> bool:
        return effect.age >= self._end_time(effect)

    # ---- querying --------------------------------------------------------
    def active_textures(self) -> set:
        """Every texture key a currently-live effect needs NOW or WILL
        need before it ends — the whole sheet for its whole lifetime (all
        frames share one sheet), plus every word cue's PNG, even ones that
        have not started yet (so the library keeps them resident)."""
        keys = set()
        for effect in self._effects:
            variant = self._variant(effect)
            if variant is not None:
                keys.add(variant["sheet"])
            if effect.words_enabled:
                for ws in effect.word_states:
                    keys.add(self._spec.word_rel_path(ws.cue["word"], ws.cue["style"]))
        return keys

    def draw_list(self) -> list:
        commands = []
        for effect in self._effects:
            variant = self._variant(effect)
            if variant is not None:
                frame_index = math.floor(effect.age * self._spec.fps)
                frames = variant["frames"]
                if 0 <= frame_index < len(frames):
                    frame = frames[frame_index]
                    if frame is not None:
                        commands.append(self._sprite_command(effect, variant, frame))
            if effect.words_enabled:
                for ws in effect.word_states:
                    if ws.current is not None:
                        commands.append(self._word_command(ws))
        return commands

    def _sprite_command(self, effect: _ActiveEffect, variant: dict,
                        frame: dict) -> FxDrawCommand:
        scale = self._spec.image_scale
        rx, ry, rw, rh = frame["rect"]
        ox, oy = frame["offset"]
        sheet_key = variant["sheet"]
        sheet_w, sheet_h = self._spec.sheet_sizes[sheet_key]
        if effect.mirror:
            ox = -ox - rw
            u0, v0 = (rx + rw) / sheet_w, ry / sheet_h
            u1, v1 = rx / sheet_w, (ry + rh) / sheet_h
        else:
            u0, v0 = rx / sheet_w, ry / sheet_h
            u1, v1 = (rx + rw) / sheet_w, (ry + rh) / sheet_h
        x0 = effect.x + ox / scale
        y0 = effect.y + oy / scale
        x1 = x0 + rw / scale
        y1 = y0 + rh / scale
        corners = ((x0, y0), (x1, y0), (x1, y1), (x0, y1))
        return FxDrawCommand(sheet_key, corners, (u0, v0, u1, v1), 1.0)

    def _word_command(self, ws: _ActiveWord) -> FxDrawCommand:
        cx, cy, sc, angle = ws.current
        word, style = ws.cue["word"], ws.cue["style"]
        png_w, png_h = _word_png_size(self._spec, word, style,
                                      self._word_png_size_cache)
        half_w, half_h = (png_w * sc) / 2.0, (png_h * sc) / 2.0
        local = ((-half_w, -half_h), (half_w, -half_h),
                (half_w, half_h), (-half_w, half_h))
        cos_a, sin_a = math.cos(angle), math.sin(angle)
        # Positive angle = clockwise in y-down space: the standard
        # (math-CCW) rotation matrix applied in a y-down space reads as
        # clockwise on screen.
        corners = tuple((cx + lx * cos_a - ly * sin_a,
                         cy + lx * sin_a + ly * cos_a) for lx, ly in local)
        key = self._spec.word_rel_path(word, style)
        return FxDrawCommand(key, corners, (0.0, 0.0, 1.0, 1.0), 1.0)

    # ---- clearing ----------------------------------------------------------
    def clear(self, pred) -> list:
        """Remove every live effect whose NAME satisfies ``pred(name)``.
        Returns the list of removed effects' names, so the GL layer can
        decide what textures are now safe to free."""
        keep, removed = [], []
        for effect in self._effects:
            if pred(effect.name):
                removed.append(effect.name)
            else:
                keep.append(effect)
        self._effects = keep
        return removed
