"""
GL texture residency for Sakuga VFX (TASK_ui_sakuga_1117). Decoding a PNG
with PIL happens on a background thread; uploading the decoded pixels to
the GPU can only happen on the main GL thread, so ``pump()`` drains at
most one decoded image per frame there. An effect whose sheet or word PNG
isn't resident yet is simply not spawned (``try_spawn``/``try_spawn_word``,
logged at debug) or not drawn (``draw``, silently skipped for that frame)
— this layer never blocks a frame waiting on disk or decode.

Kept apart from ``player.py``/``spec.py`` (no GL there) so those stay
importable and testable with no OpenGL context.
"""

from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from PIL import Image

from src.fx import screens as fx_screens
from src.fx.player import FxPlayer
from src.fx.spec import FxSpec
from src.ui_album.textures import upload_rgba

logger = logging.getLogger(__name__)

#: Upload at most this many decoded textures per frame (``pump()``) — one
#: ``glTexImage2D`` of even the biggest sheet is cheap next to a 25ms
#: frame budget (G4), but staggering uploads keeps any single frame's fx
#: GL work small and predictable.
MAX_UPLOADS_PER_FRAME = 1

#: The tallest real sheet (hit_kill, per assets/fx/README.md) is 3548px;
#: a GPU that can't hold that can't hold Sakuga VFX at all.
MIN_REQUIRED_TEXTURE_SIZE = 4096


@dataclass
class _Decoded:
    key: str
    array: np.ndarray | None


def _word_key(name: str) -> tuple | None:
    """``"__word__:gogo:ink"`` -> ``("gogo", "ink")``; ``None`` for a
    named-effect's own ``_ActiveEffect.name``."""
    if not name.startswith("__word__:"):
        return None
    _prefix, word, style = name.split(":", 2)
    return (word, style)


def _in_scope(name: str, effect_names: frozenset, word_pairs: frozenset) -> bool:
    pair = _word_key(name)
    if pair is not None:
        return pair in word_pairs
    return name in effect_names


def _texture_keys_for_screen(spec: FxSpec, screen: str) -> set:
    """Every sheet/word PNG a screen's effects COULD need — every
    variant's sheet (the one actually picked at spawn time is random),
    plus each of those effects' own baked-in word cues, plus the
    screen's bare words (``screens.words_for_screen``)."""
    keys: set = set()
    for name in fx_screens.effects_for_screen(screen):
        effect = spec.effects.get(name)
        if effect is None:
            continue
        for variant in effect.get("variants", ()):
            keys.add(variant["sheet"])
        for cue in effect.get("words") or ():
            keys.add(spec.word_rel_path(cue["word"], cue["style"]))
    for word, style in fx_screens.words_for_screen(screen):
        keys.add(spec.word_rel_path(word, style))
    return keys


class FxLibrary:
    """Lazily decodes/uploads the sheet and word PNGs the current screen
    needs, and frees ones it no longer does. One instance for the whole
    app; construct it once a ``FxSpec`` is loaded."""

    def __init__(self, spec: FxSpec) -> None:
        self._spec = spec
        self._textures: dict[str, tuple[int, int, int]] = {}
        self._pending: set[str] = set()
        self._queue: "queue.Queue[_Decoded]" = queue.Queue()
        self._max_resident_bytes = 0
        self.disabled_reason: str | None = None

    # ---- capability --------------------------------------------------
    def check_capability(self) -> bool:
        """Call once, right after a GL context exists. Returns ``False``
        (and disables every method below into a no-op) if the GPU can't
        hold the tallest sheet."""
        max_size = int(GL.glGetIntegerv(GL.GL_MAX_TEXTURE_SIZE))
        if max_size < MIN_REQUIRED_TEXTURE_SIZE:
            self.disabled_reason = (
                f"GL_MAX_TEXTURE_SIZE={max_size} < {MIN_REQUIRED_TEXTURE_SIZE}")
            logger.warning("FxLibrary: tat hieu ung (%s)", self.disabled_reason)
            return False
        return True

    # ---- residency -----------------------------------------------------
    def request(self, key: str) -> None:
        """Kick off a background decode for ``key`` if not already
        resident or in flight. Never blocks."""
        if self.disabled_reason or key in self._textures or key in self._pending:
            return
        self._pending.add(key)
        path = self._spec.root / key
        threading.Thread(target=self._decode, args=(key, path), daemon=True).start()

    def _decode(self, key: str, path) -> None:
        try:
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
        except (OSError, ValueError) as err:
            logger.warning("FxLibrary: khong doc duoc %s (%s)", path, err)
            array = None
        self._queue.put(_Decoded(key, array))

    def pump(self) -> None:
        """Main GL thread, once per frame: upload up to
        ``MAX_UPLOADS_PER_FRAME`` images that finished decoding."""
        uploads = 0
        while uploads < MAX_UPLOADS_PER_FRAME:
            try:
                decoded = self._queue.get_nowait()
            except queue.Empty:
                break
            self._pending.discard(decoded.key)
            if decoded.array is not None:
                height, width = decoded.array.shape[:2]
                self._textures[decoded.key] = (upload_rgba(decoded.array), width, height)
                uploads += 1
                resident = self.resident_bytes()
                if resident > self._max_resident_bytes:
                    self._max_resident_bytes = resident
                    logger.debug("FxLibrary: resident_bytes cao nhat moi = %d", resident)

    def get(self, key: str) -> tuple | None:
        """``(texture, w, h)`` if resident; else kicks off a decode and
        returns ``None`` — caller must skip THIS frame, not wait."""
        entry = self._textures.get(key)
        if entry is None:
            self.request(key)
        return entry

    def resident_bytes(self) -> int:
        return sum(w * h * 4 for _tex, w, h in self._textures.values())

    @property
    def max_resident_bytes(self) -> int:
        return self._max_resident_bytes

    def free(self, keys) -> None:
        for key in keys:
            entry = self._textures.pop(key, None)
            if entry is not None:
                GL.glDeleteTextures(1, [entry[0]])
            self._pending.discard(key)

    def release(self) -> None:
        self.free(list(self._textures))

    # ---- spawning (skips, never blocks, if not resident) ------------------
    def try_spawn(self, player: FxPlayer, name: str, x: float, y: float, *,
                  mirror: bool | None = None, variant: int | None = None,
                  words: bool = True) -> bool:
        """Spawn ``name`` on ``player`` if its chosen variant's sheet (and,
        if ``words``, its word PNGs) are resident; else request them and
        skip (debug log) — never spawns an effect that would render
        nothing because its texture is still decoding."""
        if self.disabled_reason:
            return False
        effect = self._spec.effects[name]
        variants = effect["variants"]
        ready = [i for i, v in enumerate(variants) if v["sheet"] in self._textures]
        for v in variants:
            self.request(v["sheet"])
        if words:
            for cue in effect.get("words") or ():
                self.request(self._spec.word_rel_path(cue["word"], cue["style"]))
        if variant is not None:
            if variant not in ready:
                logger.debug("FxLibrary: bo qua spawn %r bien the %d, chua san sang",
                            name, variant)
                return False
        elif not ready:
            logger.debug("FxLibrary: bo qua spawn %r, sheet chua san sang", name)
            return False
        else:
            variant = ready[0]
        player.spawn(name, x, y, mirror=mirror, variant=variant, words=words)
        return True

    def try_spawn_word(self, player: FxPlayer, word: str, style: str,
                       x: float, y: float, **kwargs) -> bool:
        """Same idea as :meth:`try_spawn`, for a bare word."""
        if self.disabled_reason:
            return False
        key = self._spec.word_rel_path(word, style)
        self.request(key)
        if key not in self._textures:
            logger.debug("FxLibrary: bo qua spawn_word %s/%s, chua san sang", word, style)
            return False
        player.spawn_word(word, style, x, y, **kwargs)
        return True

    # ---- per-screen scoping ----------------------------------------------
    def enter_screen(self, screen: str, player: FxPlayer) -> None:
        """Call once when the active screen changes. Clears any live
        effect outside the new screen's set — so entering a match screen
        clears every menu-set effect first — then frees whatever
        textures that leaves unneeded."""
        effect_names = fx_screens.effects_for_screen(screen)
        word_pairs = fx_screens.words_for_screen(screen)
        player.clear(lambda name: not _in_scope(name, effect_names, word_pairs))
        if self.disabled_reason:
            return
        scope = _texture_keys_for_screen(self._spec, screen)
        needed = scope | player.active_textures()
        stale = [key for key in list(self._textures) if key not in needed]
        self.free(stale)

    # ---- drawing -----------------------------------------------------
    def draw(self, painter, player: FxPlayer) -> None:
        """Draw every live effect's current frame/words. Silently skips
        any command whose texture isn't resident yet."""
        if self.disabled_reason:
            return
        for command in player.draw_list():
            entry = self.get(command.texture_key)
            if entry is None:
                continue
            texture, _width, _height = entry
            painter.quad_corners(command.corners, texture,
                                 tint=(1.0, 1.0, 1.0, command.alpha), uv=command.uv)
