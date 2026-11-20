"""
GL texture residency for Sakuga VFX.

TASK_game_feel_1118 P3 replaced the original per-screen LAZY loading
(TASK_ui_sakuga_1117): the 1117 report's own G4 note flagged sheets as
"cold" the first time a screen opened, and two 1117 screenshots
(``fx_button_primary.png``, ``fx_toast.png``) show no effect at all
because of it. Now every fx sheet and word is decoded on background
threads starting at launch (``start_preload``), menu set first, then the
match set, then the result set (TASK_game_feel_fix_1118 Fix 7 —
``star_five`` was stuck in the lowest-priority catch-all and could lose
the race to a fast first match reaching WIN), then everything else, and
uploaded in ROW BANDS (~128 rows per
``glTexSubImage2D`` call) capped at a total upload-time budget per frame
(``MAX_UPLOAD_MS_PER_FRAME``) so even the biggest atlas never costs one
frame more than that. Once uploaded, a texture stays resident for the
rest of the process — nothing is freed per-screen any more, only on
shutdown (``release``) — and its CPU-side decoded copy is dropped
immediately so RAM doesn't hold both copies forever.

Decoding a PNG with PIL happens on a background thread; uploading the
decoded pixels to the GPU can only happen on the main GL thread
(``pump()``). An effect whose sheet or word PNG isn't resident yet is
simply not spawned (``try_spawn``/``try_spawn_word``, counted in
``skipped_spawns`` and logged) or not drawn (``draw``, silently skipped
for that frame) — this layer never blocks a frame waiting on disk or
decode.

Kept apart from ``player.py``/``spec.py`` (no GL there) so those stay
importable and testable with no OpenGL context.
"""

from __future__ import annotations

import logging
import math
import queue
import threading
import time
from dataclasses import dataclass

import numpy as np
from OpenGL import GL
from PIL import Image

from src.fx import screens as fx_screens
from src.fx.player import FxPlayer
from src.fx.spec import FxSpec
from src.ui_album.textures import bleed_transparent
from src.ui_album.v3 import DESIGN_SIZE

logger = logging.getLogger(__name__)

#: TASK_game_feel_1118 P2: atlas khung cách nhau 2px — bậc mip cao hơn sẽ
#: loang màu SANG khung bên cạnh, nên chặn ở bậc 1 (vẫn đủ AA cho viền,
#: không đủ để lem). Chỉ áp cho ``sheets/`` — ảnh ``words/`` không phải
#: atlas, không có khung cạnh nhau để lo lem.
FX_MAX_MIP_LEVEL = 1

#: TASK_game_feel_1118 P3: ngân sách UPLOAD (glTexSubImage2D, không tính
#: lúc cấp phát glTexImage2D(NULL) — gần như miễn phí, không chuyển dữ
#: liệu) mỗi khung, và số hàng một "băng".
MAX_UPLOAD_MS_PER_FRAME = 4.0
BAND_ROWS = 128

#: Số luồng nền giải mã PNG song song — vài luồng đủ để không chờ tuần tự
#: từng ảnh, không nhiều tới mức tranh GIL vô ích.
PRELOAD_WORKERS = 5

#: The tallest real sheet (hit_kill, per assets/fx/README.md) is 3548px;
#: a GPU that can't hold that can't hold Sakuga VFX at all.
MIN_REQUIRED_TEXTURE_SIZE = 4096

#: Hàng đợi giải mã DÙNG CHUNG cho MỌI ``FxLibrary`` trong tiến trình,
#: với ĐÚNG ``PRELOAD_WORKERS`` luồng nền (daemon) cố định, khởi lên MỘT
#: LẦN khi cần tới. Trước đây mỗi ``start_preload()``/``request()`` tự
#: tạo một ``threading.Thread`` MỚI cho MỖI khoá — vô hại cho MỘT
#: ``FxLibrary`` sống suốt phiên chơi thật, nhưng dựng nhiều
#: ``V3UI``/``FxLibrary`` trong CÙNG MỘT tiến trình (mọi test có
#: ``gl_context``) chất chồng hàng TRĂM luồng đứng chờ mãi không ai dọn —
#: crash THẬT (Windows "stack overflow" — hết bộ nhớ ngăn xếp luồng, đo
#: được ở TASK_game_feel_1118 P3 khi chạy ``test_v3ui_fx_hooks.py``, ~19
#: ``V3UI`` dựng liên tiếp). Hàng đợi dùng chung + số luồng CỐ ĐỊNH tránh
#: hẳn kiểu phình này, bất kể dựng bao nhiêu ``FxLibrary``.
_decode_queue: "queue.Queue[tuple]" = queue.Queue()
_decode_workers_lock = threading.Lock()
_decode_workers_started = False


def _ensure_decode_workers() -> None:
    global _decode_workers_started
    if _decode_workers_started:
        return
    with _decode_workers_lock:
        if _decode_workers_started:
            return
        for _ in range(PRELOAD_WORKERS):
            threading.Thread(target=_decode_worker_loop, daemon=True).start()
        _decode_workers_started = True


def _decode_worker_loop() -> None:
    while True:
        func, args = _decode_queue.get()
        try:
            func(*args)
        except Exception:                              # noqa: BLE001
            logger.exception("FxLibrary: loi trong luong giai ma nen")


@dataclass
class _Decoded:
    key: str
    array: np.ndarray | None
    #: TASK_game_feel_fix_1118 Fix 2: per-file stage timing (background
    #: thread) — seconds, measured around PIL ``Image.open``/``asarray``
    #: and ``bleed_transparent`` separately, so a slow file's cause is
    #: visible without guessing.
    decode_s: float = 0.0
    bleed_s: float = 0.0


@dataclass
class _InProgress:
    """A texture that has been GPU-allocated (``glTexImage2D(NULL)``) but
    not yet fully filled — ``pump()`` fills ``BAND_ROWS`` rows of it at a
    time until ``next_row >= height``."""
    key: str
    texture: int
    width: int
    height: int
    array: np.ndarray
    max_level: int | None
    next_row: int = 0
    #: TASK_game_feel_fix_1118 Fix 2: cumulative ``glTexSubImage2D`` time
    #: (seconds) across however many ``pump()`` calls/bands this file
    #: took — the "upload" stage, per file.
    upload_s: float = 0.0


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


def _sheet_keys_for(spec: FxSpec, names) -> list:
    keys = []
    for name in names:
        effect = spec.effects.get(name)
        if effect is None:
            continue
        for variant in effect.get("variants", ()):
            keys.append(variant["sheet"])
        for cue in effect.get("words") or ():
            keys.append(spec.word_rel_path(cue["word"], cue["style"]))
    return keys


def _all_fx_keys(spec: FxSpec) -> list:
    """Every sheet (every variant of every effect) and every word PNG
    (every style of every word) that exists in ``fx.json`` — the full
    "decode everything" set for P3's eager preload, not just what one
    screen happens to need."""
    keys = []
    for effect in spec.effects.values():
        for variant in effect.get("variants", ()):
            keys.append(variant["sheet"])
    for word, info in spec.words.items():
        for style in info.get("styles", ()):
            keys.append(spec.word_rel_path(word, style))
    seen: set = set()
    ordered = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def _dedupe_priority(groups) -> list:
    seen: set = set()
    ordered = []
    for group in groups:
        for key in group:
            if key not in seen:
                seen.add(key)
                ordered.append(key)
    return ordered


class FxLibrary:
    """Decodes every fx sheet/word on background threads starting at
    launch, uploads them in row-bands within a per-frame time budget, and
    keeps every one resident for the rest of the process. One instance
    for the whole app; construct it once a ``FxSpec`` is loaded."""

    def __init__(self, spec: FxSpec) -> None:
        self._spec = spec
        self._textures: dict[str, tuple[int, int, int]] = {}
        self._pending: set[str] = set()
        self._failed: set[str] = set()
        self._queue: "queue.Queue[_Decoded]" = queue.Queue()
        self._in_progress: list[_InProgress] = []
        self._max_resident_bytes = 0
        self.disabled_reason: str | None = None
        #: Spawns skipped because a texture wasn't resident yet — see
        #: ``try_spawn``/``try_spawn_word``. P3's own gate (G4): this
        #: should settle at a fixed, logged count once preload finishes
        #: and never grow again afterward.
        self.skipped_spawns = 0
        self._preload_total = 0
        self._preload_started_t: float | None = None
        self._preload_done_t: float | None = None
        #: TASK_game_feel_fix_1118 Fix 2: per-file stage timing (seconds)
        #: — ``{"decode_s", "bleed_s", "mipmap_s", "upload_s"}``, filled
        #: in as each stage finishes (decode/bleed on the background
        #: thread in ``_decode``, mipmap/upload on the GL thread in
        #: ``pump()``). Never cleared — a diagnostic log, not live state.
        self.stage_times: dict[str, dict[str, float]] = {}
        #: This frame's GL-thread cost (seconds) — reset at the top of
        #: every ``pump()`` call, so a caller measuring a whole frame's
        #: wall time can tell how much of it was fx mipmap/upload work.
        self._frame_mipmap_s = 0.0
        self._frame_upload_s = 0.0

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

    # ---- P3: eager preload, menu set -> match set -> result set -> rest ---
    def start_preload(self) -> None:
        """Call once, after ``check_capability()`` succeeds. Queues a
        background decode for EVERY fx sheet/word, in priority order, on
        the small SHARED worker pool (see ``_ensure_decode_workers``) —
        never blocks.

        TASK_game_feel_fix_1118 Fix 7: ``RESULT_SET`` (``star_five``) gets
        its own tier between ``MATCH_SET`` and the catch-all — WIN is
        reached as fast as any menu/match screen (often the very first
        match), so leaving it to the lowest-priority tier let players get
        there before the sheet had decoded, i.e. the effect silently
        never spawned (``try_spawn`` just reports "not ready").
        """
        if self.disabled_reason or self._preload_started_t is not None:
            return
        order = _dedupe_priority((
            _sheet_keys_for(self._spec, fx_screens.MENU_SET),
            _sheet_keys_for(self._spec, fx_screens.MATCH_SET),
            _sheet_keys_for(self._spec, fx_screens.RESULT_SET),
            _all_fx_keys(self._spec),
        ))
        self._preload_total = len(order)
        self._preload_started_t = time.perf_counter()
        _ensure_decode_workers()
        for key in order:
            self._pending.add(key)
            _decode_queue.put((self._decode, (key,)))

    @property
    def preload_total(self) -> int:
        return self._preload_total

    @property
    def preload_remaining(self) -> int:
        return max(0, self._preload_total - len(self._textures) - len(self._failed))

    @property
    def preload_elapsed_s(self) -> float | None:
        """Seconds from ``start_preload()`` to every key either resident
        or failed — ``None`` until that happens (G4: "every fx resident
        <= 15s after launch")."""
        if self._preload_started_t is None or self._preload_done_t is None:
            return None
        return self._preload_done_t - self._preload_started_t

    @property
    def frame_mipmap_s(self) -> float:
        """Seconds spent in ``glGenerateMipmap`` during the LAST
        ``pump()`` call."""
        return self._frame_mipmap_s

    @property
    def frame_upload_s(self) -> float:
        """Seconds spent in ``glTexSubImage2D`` during the LAST
        ``pump()`` call."""
        return self._frame_upload_s

    def slowest_file(self, stage: str) -> tuple[str, float] | None:
        """``(key, seconds)`` of the file with the largest recorded
        ``stage`` time (one of "decode_s"/"bleed_s"/"mipmap_s"/
        "upload_s"), or ``None`` if nothing has that stage recorded yet."""
        best: tuple[str, float] | None = None
        for key, stages in self.stage_times.items():
            value = stages.get(stage)
            if value is None:
                continue
            if best is None or value > best[1]:
                best = (key, value)
        return best

    # ---- residency -----------------------------------------------------
    def request(self, key: str) -> None:
        """Queue a background decode for ``key`` if not already resident
        or in flight. Never blocks. A fallback path for a key preload
        somehow missed — in normal operation ``start_preload()`` already
        covers everything.

        TASK_perf_textures_1119 Fix 4: a decoded-but-still-uploading key
        is in NEITHER ``_pending`` (cleared the moment ``pump()`` drains
        it off the decode queue) NOR ``_textures`` (only set once ALL its
        bands + mipmap are done) — missing that window here meant ``get``
        could re-request (and eventually re-decode+re-upload) a key
        that's mid-upload, previously masked by decode (with the old
        full-image bleed) almost always outlasting the upload entirely."""
        if (self.disabled_reason or key in self._textures or key in self._pending
                or key in self._failed
                or any(item.key == key for item in self._in_progress)):
            return
        self._pending.add(key)
        _ensure_decode_workers()
        _decode_queue.put((self._decode, (key,)))

    def _decode(self, key: str) -> None:
        path = self._spec.root / key
        decode_s = bleed_s = 0.0
        try:
            t0 = time.perf_counter()
            with Image.open(path) as image:
                array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            t1 = time.perf_counter()
            # Loang màu ở LUỒNG NỀN này (CPU, có thể chậm), không phải ở
            # ``pump()`` (luồng GL chính). TASK_perf_textures_1119 Fix 4:
            # ``tools/sakuga_bake/pack.py`` đã loang màu OFFLINE cho MỌI
            # file của bộ fx đã chuẩn bị (``fx.json``'s ``rgb_bled: true``
            # — PNG riêng lẻ không mang chunk, pack.py không stamp) — làm
            # lại ở đây vừa tốn vừa vô ích.
            if not self._spec.rgb_bled and np.any(array[..., 3] == 0):
                array = bleed_transparent(array)
            t2 = time.perf_counter()
            decode_s, bleed_s = t1 - t0, t2 - t1
        except (OSError, ValueError) as err:
            logger.warning("FxLibrary: khong doc duoc %s (%s)", path, err)
            array = None
        self._queue.put(_Decoded(key, array, decode_s=decode_s, bleed_s=bleed_s))

    def pump(self) -> None:
        """Main GL thread, once per frame:

        1. Move every newly-decoded image into an in-progress upload —
           allocates empty GPU storage (``glTexImage2D(NULL)``, no pixel
           transfer, effectively free).
        2. Fill ``BAND_ROWS``-row bands of in-progress textures via
           ``glTexSubImage2D`` until ``MAX_UPLOAD_MS_PER_FRAME`` is spent
           (measuring only this step) or there is no more work this
           frame. A texture that finishes gets its mipmap chain
           generated, becomes resident, and its CPU-side array is
           dropped.

        TASK_game_feel_fix_1118 Fix 2: ``frame_mipmap_s``/``frame_upload_s``
        are reset here and accumulated below — THIS call's GL-thread cost
        attributable to fx, so a caller measuring a whole frame's wall
        time can tell whether fx (not something else) was the slow part.
        """
        self._frame_mipmap_s = 0.0
        self._frame_upload_s = 0.0
        if self.disabled_reason:
            return
        while True:
            try:
                decoded = self._queue.get_nowait()
            except queue.Empty:
                break
            self._pending.discard(decoded.key)
            if decoded.array is None:
                self._failed.add(decoded.key)
                self._note_if_preload_done()
                continue
            self.stage_times[decoded.key] = {
                "decode_s": decoded.decode_s, "bleed_s": decoded.bleed_s}
            self._in_progress.append(self._allocate(decoded.key, decoded.array))
        if not self._in_progress:
            return
        deadline = time.perf_counter() + (MAX_UPLOAD_MS_PER_FRAME / 1000.0)
        index = 0
        while index < len(self._in_progress):
            if time.perf_counter() >= deadline:
                break
            item = self._in_progress[index]
            self._upload_band(item)
            if item.next_row >= item.height:
                self._finish(item)
                del self._in_progress[index]
                continue
            index += 1

    def _allocate(self, key: str, array: np.ndarray) -> _InProgress:
        height, width = array.shape[:2]
        texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        max_level = FX_MAX_MIP_LEVEL if key.startswith("sheets/") else None
        if max_level is not None:
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, max_level)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        return _InProgress(key, texture, width, height, array, max_level)

    def _upload_band(self, item: _InProgress) -> None:
        t0 = time.perf_counter()
        band = min(BAND_ROWS, item.height - item.next_row)
        rows = np.ascontiguousarray(item.array[item.next_row:item.next_row + band])
        GL.glBindTexture(GL.GL_TEXTURE_2D, item.texture)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, item.next_row,
                           item.width, band, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, rows)
        item.next_row += band
        elapsed = time.perf_counter() - t0
        item.upload_s += elapsed
        self._frame_upload_s += elapsed

    def _finish(self, item: _InProgress) -> None:
        t0 = time.perf_counter()
        GL.glBindTexture(GL.GL_TEXTURE_2D, item.texture)
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        mipmap_s = time.perf_counter() - t0
        self._frame_mipmap_s += mipmap_s
        self._textures[item.key] = (item.texture, item.width, item.height)
        stage = self.stage_times.setdefault(item.key, {})
        stage["mipmap_s"] = mipmap_s
        stage["upload_s"] = item.upload_s
        item.array = None          # giải phóng bản CPU — đã lên GPU xong
        resident = self.resident_bytes()
        if resident > self._max_resident_bytes:
            self._max_resident_bytes = resident
        self._note_if_preload_done()

    def _note_if_preload_done(self) -> None:
        if self._preload_done_t is not None or self._preload_started_t is None:
            return
        if self.preload_remaining > 0:
            return
        self._preload_done_t = time.perf_counter()
        elapsed = self._preload_done_t - self._preload_started_t
        logger.info(
            "FxLibrary: nap truoc xong sau %.2fs (%d anh, %d loi, %d lan bo "
            "qua spawn vi chua san sang)", elapsed, self._preload_total,
            len(self._failed), self.skipped_spawns)

    def get(self, key: str) -> tuple | None:
        """``(texture, w, h)`` if resident; else kicks off a decode (as a
        fallback — ``start_preload()`` should already cover it) and
        returns ``None``, caller must skip THIS frame, not wait."""
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
        """Spawn ``name`` on ``player`` ONLY when its chosen variant's
        sheet (every band uploaded — ``in self._textures`` only becomes
        true in ``_finish()``, never for a half-uploaded ``_InProgress``)
        AND every word PNG it uses (if ``words``) are ALL resident;
        else request whatever's missing, count it in ``skipped_spawns``
        and skip the WHOLE effect (debug log).

        TASK_game_feel_fix_1118 Fix 1: this used to gate on the sheet
        alone — the word PNGs were requested but never checked, so an
        effect could spawn with a resident sheet and a still-decoding
        word, drawing the sheet's frames with no word (or, since a
        word's own PNG is tiny and often finishes first while a big
        sheet is still mid-upload, a word with no sheet at all —
        ``aa_burst.png``'s "after" shot in the P7 report showed exactly
        that: DON with no burst). An effect must appear all at once or
        not at all.
        """
        if self.disabled_reason:
            return False
        effect = self._spec.effects[name]
        variants = effect["variants"]
        ready = [i for i, v in enumerate(variants) if v["sheet"] in self._textures]
        for v in variants:
            self.request(v["sheet"])
        word_keys = []
        if words:
            for cue in effect.get("words") or ():
                key = self._spec.word_rel_path(cue["word"], cue["style"])
                word_keys.append(key)
                self.request(key)
        words_ready = all(key in self._textures for key in word_keys)
        sheet_ready = (variant in ready) if variant is not None else bool(ready)
        if not sheet_ready or not words_ready:
            self.skipped_spawns += 1
            logger.debug(
                "FxLibrary: bo qua spawn %r (sheet_ready=%s, words_ready=%s), "
                "chua san sang het", name, sheet_ready, words_ready)
            return False
        if variant is None:
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
            self.skipped_spawns += 1
            logger.debug("FxLibrary: bo qua spawn_word %s/%s, chua san sang", word, style)
            return False
        player.spawn_word(word, style, x, y, **kwargs)
        return True

    # ---- per-screen scoping ----------------------------------------------
    def enter_screen(self, screen: str, player: FxPlayer) -> None:
        """Call once when the active screen changes. Clears any live
        effect outside the new screen's set — so entering a match screen
        clears every menu-set effect first.

        TASK_game_feel_1118 P3: no longer frees textures here — "keep
        everything resident" (see module docstring); freeing only
        happens on ``release()`` at shutdown.
        """
        effect_names = fx_screens.effects_for_screen(screen)
        word_pairs = fx_screens.words_for_screen(screen)
        player.clear(lambda name: not _in_scope(name, effect_names, word_pairs))

    # ---- drawing -----------------------------------------------------
    def draw(self, painter, player: FxPlayer, offset: tuple = (0.0, 0.0), *,
            transform=None, hide_beyond: tuple | None = None) -> None:
        """Draw every live effect's current frame/words. Silently skips
        any command whose texture isn't resident yet.

        Clipped to the design box with a scissor — a word/sprite quad
        can straddle the edge of the 1280x720 canvas (e.g. a word cue's
        ``dx``/``dy`` pushed past it), and without a clip it draws into
        the black letterbox bars either side of the canvas at non-16:9
        window sizes.

        ``offset``: TASK_game_feel_1118 P4's screen shake moves the fx
        layer (along with the 3D scene viewport, elsewhere) but never the
        HUD — every corner is shifted by this design-px ``(dx, dy)``
        before drawing; ``(0.0, 0.0)`` (the default) is a no-op.

        ``transform``/``hide_beyond`` (TASK_game_feel_fix_1118 Fix 5):
        while scoped, a scene-anchored effect (``hit_kill``/``hit_crit``/
        ``hit_block``, spawned at the crosshair's position AT THE TIME)
        must track the GLASS POINT it was spawned over, not stay frozen
        at that old design-px spot — ``transform`` is
        ``scope.screen_scope_transform`` bound to the CURRENT crosshair/
        zoom, passed straight to ``player.draw_list()`` (applied per
        CORNER there, so the effect scales with the magnified scene, not
        just repositions). ``hide_beyond``, if given, is
        ``(cx, cy, radius)`` — a command whose (transformed) corner
        centroid is further than ``radius`` from ``(cx, cy)`` is skipped
        ("hide them outside the scope circle").
        """
        if self.disabled_reason:
            return
        commands = player.draw_list(transform=transform)
        if not commands:
            return
        dx, dy = offset
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(*painter.viewport_px((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1])))
        try:
            for command in commands:
                entry = self.get(command.texture_key)
                if entry is None:
                    continue
                texture, _width, _height = entry
                corners = command.corners
                if dx or dy:
                    corners = tuple((px + dx, py + dy) for px, py in corners)
                if hide_beyond is not None:
                    hx, hy, radius = hide_beyond
                    cx = sum(p[0] for p in corners) / len(corners)
                    cy = sum(p[1] for p in corners) / len(corners)
                    if math.hypot(cx - hx, cy - hy) > radius:
                        continue
                painter.quad_corners(corners, texture,
                                     tint=(1.0, 1.0, 1.0, command.alpha), uv=command.uv)
        finally:
            GL.glDisable(GL.GL_SCISSOR_TEST)
