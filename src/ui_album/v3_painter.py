"""
The v3 UI's GL painter (TASK_arch_refactor_1120 R4: split out of
``src.ui_album.v3``, which had grown to 2479 lines).

``V3Painter`` draws quads in **design space** (1280x720); scaling/blending
is its own concern, same contract as before — see ``src/ui_album/v3.py``'s
own module docstring for the draw-order contract this whole UI follows.
"""

from __future__ import annotations

import math
import re
import time
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw

from src.renderer.shader import ShaderProgram
from src.ui_album.fonts import FontBook
from src.ui_album.manifest import letterbox
from src.ui_album.texture_loader import PRIORITY_CURRENT, PRIORITY_PREFETCH, AsyncTextureLoader
from src.ui_album.textures import TextureCache, upload_rgba

if TYPE_CHECKING:
    from src.ui_album.v3 import V3Manifest

#: Design-space canvas size + default ink colour (``src.ui_album.v3``'s own
#: values, duplicated here — same pattern as ``v3_hit.py``/R2's
#: ``fx/library.py`` DESIGN_SIZE, so this module needs nothing back from
#: ``v3.py``).
DESIGN_SIZE = (1280.0, 720.0)
INK = "#24415F"

#: Lề quanh ảnh chữ. Glyph có thể tràn ra ngoài bút (bearing âm, dấu cao hơn
#: ascent); không chừa lề thì chữ bị cắt mép và điểm compare.py nhảy vọt.
TEXT_PAD = 8

#: TASK_perf_textures_1119 Fix 3: số texture tối đa giữ trong
#: ``V3Painter._screen_tex`` (màn đang mở + các màn prefetch) trước khi
#: LRU bắt đầu xoá — đủ rộng cho màn nặng sprite nhất (LOADOUT, ~25) cộng
#: vài màn hàng xóm, không rộng tới mức giữ cả app trong RAM GPU.
SCREEN_CACHE_MAX_TEXTURES = 120


def rgba(colour: str | None, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``"#RRGGBB"`` → ``(r, g, b, a)`` 0–1. ``None`` trả màu chữ mặc định."""
    text = str(colour or INK).lstrip("#")
    if len(text) == 6:
        text += "FF"
    try:
        value = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6)]
    except ValueError:
        return (1.0, 0.0, 1.0, 1.0)             # màu báo lỗi, thấy ngay trên màn
    return (value[0], value[1], value[2], value[3] * float(alpha))


def _byte(colour: str | None, alpha: float = 1.0) -> tuple[int, int, int, int]:
    """Màu cho PIL: ``(r, g, b, a)`` 0–255."""
    return tuple(int(round(c * 255)) for c in rgba(colour, alpha))


#: TASK_game_feel_fix_1118 Fix 11: every PIL ``ImageDraw``-built shape in
#: this module (``rounded_frame``, the rounded cap, ``disc``, ``pip``,
#: ``pill``) draws at this many times the final size, then downsamples
#: with a box filter — ``ImageDraw`` itself has no antialiasing, so a
#: curved or diagonal edge drawn straight at the final (small) size comes
#: out a hard 1px staircase (the user's own 18:24 report: the focus ring
#: around the Password field).
_AA_FACTOR = 4


def _aa_build(width_px: int, height_px: int, draw):
    """Shared AA path: a fresh ``(width_px*_AA_FACTOR, height_px*_AA_FACTOR)``
    RGBA canvas, ``draw(pen, big_w, big_h)`` draws the shape (and, for
    ``pill``, its text) onto it at that bigger size, then
    ``Image.reduce(_AA_FACTOR)`` box-filters it back down to EXACTLY
    ``width_px x height_px`` — the cache key and every downstream UV/size
    calculation in the caller are computed at the final size and never
    need to know AA happened at all. Callers multiply their own
    scale/radius/stroke-width numbers by ``_AA_FACTOR`` inside ``draw``;
    using the ``big_w``/``big_h`` passed in (not ``width_px*_AA_FACTOR``
    re-derived) keeps every edge coordinate exact."""
    big_w, big_h = width_px * _AA_FACTOR, height_px * _AA_FACTOR
    image = Image.new("RGBA", (big_w, big_h), (0, 0, 0, 0))
    draw(ImageDraw.Draw(image), big_w, big_h)
    return np.asarray(image.reduce(_AA_FACTOR), dtype=np.uint8)


class _PilCache:
    """
    LRU "dựng một lần bằng PIL, dùng lại mãi": khoá → (texture, số đo).

    Vì sao không dùng thẳng ``TextureCache.small``: chữ và viên thuốc cần
    **số đo** (bề rộng chuỗi, ascent) để đặt quad đúng chỗ, mà số đo ấy chỉ
    tính được lúc dựng. Trả kèm số đo nghĩa là đường chạy mỗi khung không gọi
    ``font.getlength`` lần nào nữa.
    """

    def __init__(self, limit: int = 768) -> None:
        self._store: OrderedDict[tuple, tuple] = OrderedDict()
        self._limit = int(limit)
        self.builds = 0

    def get(self, key: tuple, build):
        """``(texture, w_px, h_px, meta)``; gọi ``build()`` khi chưa có."""
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        array, meta = build()
        if array is None:
            return None
        entry = (upload_rgba(array), array.shape[1], array.shape[0], meta)
        self._store[key] = entry
        self.builds += 1
        while len(self._store) > self._limit:
            _key, old = self._store.popitem(last=False)
            GL.glDeleteTextures(1, [old[0]])
        return entry

    def release(self) -> None:
        for entry in self._store.values():
            GL.glDeleteTextures(1, [entry[0]])
        self._store.clear()


class V3Painter:
    """
    Vẽ quad trong **không gian thiết kế** 1280×720; tự lo tỉ lệ và blend.

    ``set_frame`` nhận cỡ framebuffer THẬT mỗi khung, nên đổi cỡ cửa sổ (D1)
    không cần dựng lại gì ngoài những texture phụ thuộc tỉ lệ.
    """

    def __init__(self, manifest: V3Manifest, shader_dir: Path) -> None:
        self.manifest = manifest
        self.cache = TextureCache()
        self.pil = _PilCache()
        self.fonts = FontBook(manifest)
        self._shader = ShaderProgram(shader_dir / "album.vert",
                                     shader_dir / "album.frag")
        self._vao = self._vbo = 0
        self._fb = (1280, 720)
        self._box = letterbox(DESIGN_SIZE, self._fb)
        #: Texture của màn đang mở VÀ các màn đã prefetch: ``(path,
        #: max_width)`` → ``(tex, w, h)``. ``OrderedDict`` cho LRU —
        #: TASK_perf_textures_1119 Fix 3: không xoá sạch mỗi lần đổi màn
        #: nữa (đó mới là nguồn đơ thật, nặng hơn cả README's
        #: ``IMAGE_CACHE_MAX=6`` của ``TextureCache``), chỉ xoá khi vượt
        #: ``SCREEN_CACHE_MAX_TEXTURES``.
        self._screen_tex: OrderedDict[tuple, tuple[int, int, int]] = OrderedDict()
        self._screen_key: tuple | None = None
        #: TASK_perf_textures_1119 Fix 3: giải mã (decode/bleed/resize)
        #: chạy trên luồng riêng — luồng vẽ chỉ queue (``request``) và
        #: rút kết quả đã xong (``poll``) trong ``pump_uploads``.
        self._loader = AsyncTextureLoader()
        #: Kết quả đã giải mã (CPU), chờ tới lượt lên GPU trong ngân sách
        #: ``pump_uploads`` — khung này chưa hết ngân sách thì phần còn lại
        #: đợi khung sau, KHÔNG rơi mất (``poll()`` không gọi lại được).
        self._upload_backlog: list[tuple[tuple, object]] = []

    # ---- hạ tầng ---------------------------------------------------------
    def initialize(self) -> None:
        """Tạo VAO/VBO một lần, sau khi đã có context GL."""
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 96, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                 GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def set_frame(self, framebuffer) -> None:
        self._fb = (max(1, int(framebuffer[0])), max(1, int(framebuffer[1])))
        self._box = letterbox(DESIGN_SIZE, self._fb)

    @property
    def scale(self) -> float:
        """Pixel framebuffer trên mỗi pixel thiết kế."""
        return self._box.scale

    @property
    def k(self) -> float:
        """
        Tỉ lệ **của bộ asset** (``image_scale`` = 1,5), dùng cho chữ và viên.

        Vì sao không dựng chữ ở tỉ lệ framebuffer, là thứ trông có vẻ đúng
        hơn: FreeType làm tròn bước tiến của **từng** glyph về pixel nguyên,
        nên bề rộng một chuỗi KHÔNG tỉ lệ thuận với cỡ chữ. Dựng ở 1366/1280
        = 1,0667 thay vì 1,5 làm "anhkhoa" rộng thêm 5 %, dãy tám dấu chấm
        hẹp đi 11 %, và điểm ``compare.py`` nhảy từ ~3 lên 42. Dựng đúng tỉ lệ
        mà ``tools/preview.py`` đã dùng thì số đo khớp từng pixel, còn việc
        co giãn để GPU làm — y như nó đang làm với ``base`` và sprite, vốn
        cũng xuất ở 1,5×.
        """
        return float(self.manifest.image_scale)

    @property
    def box(self):
        return self._box

    def unmap(self, x: float, y: float) -> tuple[float, float]:
        """Pixel cửa sổ → toạ độ thiết kế (bắt chuột, U1.2)."""
        return self._box.unmap(x, y)

    def map(self, x: float, y: float) -> tuple[float, float]:
        """Toạ độ thiết kế → pixel cửa sổ — nghịch đảo của ``unmap``."""
        return self._box.map(x, y)

    def begin(self) -> None:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def end(self) -> None:
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def viewport_px(self, rect) -> tuple[int, int, int, int]:
        """``rect`` thiết kế → ``(x, y, w, h)`` pixel GL (gốc dưới-trái)."""
        x, y, w, h = self._box.rect(rect)
        return (int(round(x)), int(round(self._fb[1] - y - h)),
                max(1, int(round(w))), max(1, int(round(h))))

    def quad(self, rect, texture, tint=(1.0, 1.0, 1.0, 1.0),
             uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        """Một quad ở ``rect`` thiết kế, lấy vùng ``uv`` của texture."""
        if texture is None or rect[2] <= 0.0 or rect[3] <= 0.0:
            return
        x, y, w, h = self._box.rect(rect)
        fb_w, fb_h = self._fb
        x0, x1 = 2.0 * x / fb_w - 1.0, 2.0 * (x + w) / fb_w - 1.0
        y0, y1 = 1.0 - 2.0 * y / fb_h, 1.0 - 2.0 * (y + h) / fb_h
        u0, v0, u1, v1 = uv
        data = np.array([x0, y1, u0, v1, x1, y1, u1, v1,
                         x1, y0, u1, v0, x0, y1, u0, v1,
                         x1, y0, u1, v0, x0, y0, u0, v0], dtype=np.float32)
        self._draw_quad(data, texture, tint)

    def quad_corners(self, corners, texture, tint=(1.0, 1.0, 1.0, 1.0),
                     uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        """
        Một quad từ 4 góc ``(trên-trái, trên-phải, dưới-phải, dưới-trái)``
        toạ độ THIẾT KẾ bất kỳ — dùng cho chữ hiệu ứng đã QUAY, thứ
        ``quad()`` không vẽ được vì nó chỉ nhận ``rect`` thẳng trục.
        """
        if texture is None:
            return
        box, fb_w, fb_h = self._box, *self._fb
        def _ndc(point):
            px, py = box.ox + point[0] * box.scale, box.oy + point[1] * box.scale
            return (2.0 * px / fb_w - 1.0, 1.0 - 2.0 * py / fb_h)
        tl, tr, br, bl = (_ndc(p) for p in corners)
        u0, v0, u1, v1 = uv
        data = np.array([bl[0], bl[1], u0, v1, br[0], br[1], u1, v1,
                         tr[0], tr[1], u1, v0, bl[0], bl[1], u0, v1,
                         tr[0], tr[1], u1, v0, tl[0], tl[1], u0, v0],
                        dtype=np.float32)
        self._draw_quad(data, texture, tint)

    def _draw_quad(self, data, texture, tint) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        self._shader.use()
        GL.glUniform1i(GL.glGetUniformLocation(self._shader.program_id,
                                               "uTexture"), 0)
        GL.glUniform4f(GL.glGetUniformLocation(self._shader.program_id,
                                               "uTint"), *tint)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def fill(self, rect, colour) -> None:
        """Hình chữ nhật tô một màu (texture 1×1 trắng × tint)."""
        entry = self.cache.white()
        if entry:
            self.quad(rect, entry[0], colour)

    def frame(self, rect, colour, width: float) -> None:
        """Khung rỗng bốn thanh — vòng hover và vòng focus."""
        x, y, w, h = rect
        self.fill((x, y, w, width), colour)
        self.fill((x, y + h - width, w, width), colour)
        self.fill((x, y, width, h), colour)
        self.fill((x + w - width, y, width, h), colour)

    def rounded_frame(self, rect, colour, width: float, radius: float) -> None:
        """
        Khung rỗng bo góc — vòng hover/focus/selection quanh một ô BO GÓC.

        ``radius`` 0 (phần tử không bo, xem ``_ring_radius``) đi thẳng về
        ``frame()`` vuông góc: khỏi tốn một texture đệm cho trường hợp phổ
        biến nhất. ``radius`` > 0 dựng bằng PIL rồi đưa lên GPU, đệm theo
        khoá — cùng khuôn với ``cap``/``disc``/``pip`` ở trên.
        """
        if radius < 0.5:
            self.frame(rect, colour, width)
            return
        scale = self.scale
        w_px = max(8, int(round(rect[2] * scale)))
        h_px = max(8, int(round(rect[3] * scale)))
        key = ("rframe", w_px, h_px, round(width * scale, 2),
               round(radius * scale, 2))

        def build():
            def draw(pen, big_w, big_h):
                sw = max(1, int(round(width * scale * _AA_FACTOR)))
                half = sw / 2.0
                pen.rounded_rectangle(
                    [half, half, big_w - 1 - half, big_h - 1 - half],
                    radius=radius * scale * _AA_FACTOR,
                    outline=(255, 255, 255, 255), width=sw)

            return _aa_build(w_px, h_px, draw), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad(rect, entry[0], colour)

    # ---- texture của một màn --------------------------------------------
    def _wanted_textures(self, screen: str, extra_files=()) -> dict[tuple, int | None]:
        """``{(path, max_width): max_width}`` của ``base``/``top``/sprite
        một màn cần — tách khỏi ``bind_screen`` để ``screen_ready`` dùng
        lại đúng tập này, không tính hai lần hai cách khác nhau."""
        manifest = self.manifest
        record = manifest.screens[screen]
        big = max(1, int(self._fb[0]))
        wanted: dict[tuple, int | None] = {}
        for layer in ("base", "top"):
            path = manifest.path(record.get(layer))
            if path is not None:
                wanted[(str(path), big)] = big
        for sprite in record.get("sprites", ()):
            path = manifest.path(sprite.get("file"))
            if path is not None:
                wanted[(str(path), 0)] = None
        for relative in extra_files:
            path = manifest.path(relative)
            if path is not None:
                wanted[(str(path), 0)] = None
        return wanted

    def bind_screen(self, screen: str, extra_files=(),
                    *, priority: int = PRIORITY_CURRENT) -> None:
        """
        Queue ``base``/``top``/sprite của màn lên loader nền — **không
        decode gì trên luồng này** (TASK_perf_textures_1119 Fix 3).

        ``base``/``top`` thu về đúng bề rộng framebuffer: ảnh xuất ở 1920 px
        mà cửa sổ 1280 thì giữ nguyên cỡ chỉ tốn RAM GPU và fillrate. Sprite
        giữ cỡ gốc (``item_tile``/``weapon_tile`` xuất ở 4× để còn nét).

        ``priority=PRIORITY_PREFETCH`` (màn hàng xóm, chưa phải màn đang
        mở) không động tới ``_screen_key`` — đó chỉ theo dõi màn đang mở
        THẬT để tránh queue lại mỗi khung.
        """
        wanted = self._wanted_textures(screen, extra_files)
        if priority == PRIORITY_CURRENT:
            big = max(1, int(self._fb[0]))
            key = (screen, big, tuple(sorted(extra_files)))
            if key == self._screen_key:
                return
            self._screen_key = key
        for tex_key, max_width in wanted.items():
            if tex_key in self._screen_tex:
                self._screen_tex.move_to_end(tex_key)
                continue
            if self._loader.is_pending(tex_key):
                continue
            self._loader.request(tex_key, tex_key[0], max_width, priority)

    def screen_ready(self, screen: str, extra_files=()) -> bool:
        """True nếu MỌI texture màn ``screen`` cần đã có trên GPU — dùng
        để quyết "đổi màn hay giữ khung cũ" (Fix 3) và để lượt chụp
        (``_album_shots_step``) không đọc pixel trước khi màn thật xong."""
        wanted = self._wanted_textures(screen, extra_files)
        return all(tex_key in self._screen_tex for tex_key in wanted)

    def pump_uploads(self, budget_s: float = 0.004) -> None:
        """Rút kết quả CPU đã giải mã xong (``AsyncTextureLoader.poll``)
        và lên GPU (``upload_rgba`` — ``glTexImage2D`` + mipmap) trong
        ngân sách ``budget_s`` mỗi khung; phần chưa tới lượt đợi khung
        sau (``_upload_backlog``, không rơi mất). Gọi mỗi khung từ
        ``V3UI.draw``/``prepare_screen``."""
        self._upload_backlog.extend(self._loader.poll())
        t0 = time.perf_counter()
        while self._upload_backlog:
            if time.perf_counter() - t0 >= budget_s:
                break
            tex_key, array = self._upload_backlog.pop(0)
            if array is None:
                continue
            self._screen_tex[tex_key] = (upload_rgba(array), array.shape[1],
                                         array.shape[0])
            self._screen_tex.move_to_end(tex_key)
        self._evict_screen_cache()

    def _evict_screen_cache(self) -> None:
        while len(self._screen_tex) > SCREEN_CACHE_MAX_TEXTURES:
            _old_key, entry = self._screen_tex.popitem(last=False)
            GL.glDeleteTextures(1, [entry[0]])

    def pause_loading(self) -> None:
        self._loader.pause()

    def resume_loading(self) -> None:
        self._loader.resume()

    def screen_cache_resident_mib(self) -> float:
        """TASK_perf_textures_1119 G7: RAM GPU hiện đang giữ cho
        ``_screen_tex`` (RGBA 8-bit, không tính mipmap)."""
        total_bytes = sum(w * h * 4 for _tex, w, h in self._screen_tex.values())
        return total_bytes / (1024.0 * 1024.0)

    def screen_image(self, relative: str | None, *, big: bool = False):
        """``(tex, w, h)`` của một file đã nạp cho màn đang mở."""
        path = self.manifest.path(relative)
        if path is None:
            return None
        return self._screen_tex.get((str(path), int(self._fb[0]) if big else 0))

    def layer(self, rect, relative: str | None) -> None:
        """Vẽ ``base``/``top`` phủ kín ``rect``."""
        entry = self.screen_image(relative, big=True)
        if entry:
            self.quad(rect, entry[0])

    def layer_region(self, rect, relative: str | None, alpha: float = 1.0) -> None:
        """TASK_game_feel_fix_1118 Fix 10.1: a SUB-RECT of a full-canvas
        ``base``/``top`` layer image, own alpha — ``rect`` is one HUD
        card's design-space rect (from ``hud_cards``). The image still
        covers the WHOLE design canvas (same file as ``layer()`` draws),
        so the UV sub-rect is just ``rect`` divided by ``DESIGN_SIZE`` —
        this is what lets one card fade independently without the art
        being baked into separate files per card."""
        entry = self.screen_image(relative, big=True)
        if not entry:
            return
        width, height = DESIGN_SIZE
        x, y, w, h = rect
        uv = (x / width, y / height, (x + w) / width, (y + h) / height)
        self.quad(rect, entry[0], (1.0, 1.0, 1.0, float(alpha)), uv)

    def sprite(self, rect, relative: str | None, alpha: float = 1.0,
               mirror: bool = False) -> None:
        """``mirror`` lật ngang bằng UV — không cần file sprite thứ hai."""
        entry = self.screen_image(relative)
        if entry:
            uv = (1.0, 0.0, 0.0, 1.0) if mirror else (0.0, 0.0, 1.0, 1.0)
            self.quad(rect, entry[0], (1.0, 1.0, 1.0, float(alpha)), uv)

    # ---- chữ -------------------------------------------------------------
    def _text_entry(self, runs, texts):
        """
        Texture + số đo của một dòng chữ (một baseline, nhiều run).

        Hai tỉ lệ, và phải tách bạch:

        - **Số đo** lấy ở ``k`` = 1,5, đúng tỉ lệ ``tools/preview.py`` dùng.
          Bước tiến từng glyph bị FreeType làm tròn về pixel nguyên, nên bề
          rộng chuỗi không tỉ lệ thuận với cỡ chữ; đo ở tỉ lệ khác là lệch
          tới 11 % (xem ``V3Painter.k``).
        - **Nét chữ** rasterise ở tỉ lệ framebuffer thật. Dựng ở 1,5× rồi để
          GL thu nhỏ bằng bilinear (không mipmap) làm chữ rỗ và mảnh đi:
          hình học đúng tới ±1 px mà ``compare.py`` vẫn 30–50.

        Nên: vị trí bút tính ở ``k`` rồi quy về pixel thiết kế, còn glyph vẽ
        bằng font cỡ ``k``. Không có kerning (PIL layout cơ bản — đã kiểm:
        tổng bước tiến từng glyph khớp đúng bề rộng cả chuỗi), nên vẽ rời
        từng glyph cho ra đúng chữ mà ``preview.py`` vẽ liền một hơi.

        Đã thử rasterise ở tỉ lệ framebuffer cho chữ nét hơn: nó **xa**
        ``expected/`` hơn chứ không gần hơn (ô tệ nhất 49,6 → 59,1). Lý do:
        ``expected/`` cũng là bản 1,5× bị thu nhỏ, nên muốn giống nó thì
        phải đi qua đúng cái phễu thu nhỏ ấy. Mà hệ số thu ở đây là 0,711 —
        vẫn nằm trong mip 0, nên GL chỉ lọc bilinear một nhịp, không nhoè.
        """
        k = self.k
        scale = k
        style = tuple((r.get("ff"), r.get("fw"), round(float(r.get("fs", 16)), 3),
                       r.get("col"), round(float(r.get("ls") or 0.0), 3),
                       bool(r.get("upper")))
                      for r in runs)
        key = ("text", style, tuple(texts), round(scale, 5))

        def build():
            items, total = [], 0.0
            for record, raw in zip(runs, texts):
                text = raw.upper() if record.get("upper") else raw
                size = float(record.get("fs", 16))
                metric = self.fonts.get(record.get("ff"), record.get("fw"),
                                        round(size * k))
                glyph = self.fonts.get(record.get("ff"), record.get("fw"),
                                       round(size * scale))
                spacing = float(record.get("ls") or 0.0)
                items.append((metric, glyph, text, _byte(record.get("col")),
                              spacing))
                total += _run_width(metric, text, spacing * k) / k
            biggest = max(runs, key=lambda r: float(r.get("fs", 16)))
            ascent_k, descent_k = self.fonts.get(
                biggest.get("ff"), biggest.get("fw"),
                round(float(biggest.get("fs", 16)) * k)).getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            width = max(1, int(math.ceil(total * scale)) + 2 * TEXT_PAD)
            height = max(1, int(math.ceil((ascent + descent) * scale))
                         + 2 * TEXT_PAD)
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            cursor = 0.0
            baseline = float(TEXT_PAD) + ascent * scale
            for metric, glyph, text, colour, spacing in items:
                for char in text:
                    pen.text((TEXT_PAD + cursor * scale, baseline), char,
                             font=glyph, fill=colour, anchor="ls")
                    cursor += metric.getlength(char) / k + spacing
            return (np.asarray(image, dtype=np.uint8),
                    (float(total), float(ascent), float(descent)))

        return self.pil.get(key, build)

    def line(self, rect, runs, texts, align: str = "left",
             alpha: float = 1.0) -> float:
        """
        Vẽ một dòng chữ theo đúng phép canh của preview.py.

        Trả mép phải của chuỗi, tính bằng **pixel thiết kế** (con trỏ nhập
        bám vào đó). Canh dọc: hộp dòng của run **lớn nhất** nằm giữa
        ``rect``. Canh ngang theo ``align`` trên tổng bề rộng mọi run.
        """
        entry = self._text_entry(runs, texts)
        if not entry:
            return 0.0
        texture, width_px, height_px, (total, ascent, descent) = entry
        scale = self.k
        x, y, w, h = (float(v) for v in rect)
        if align == "right":
            start = x + w - total
        elif align == "center":
            start = x + (w - total) * 0.5
        else:
            start = x
        baseline = y + (h - (ascent + descent)) * 0.5 + ascent
        self.quad((start - TEXT_PAD / scale,
                   baseline - ascent - TEXT_PAD / scale,
                   width_px / scale, height_px / scale),
                  texture, (1.0, 1.0, 1.0, float(alpha)))
        return start + total

    def text_width(self, runs, texts) -> float:
        """Bề rộng một dòng chữ, tính bằng **pixel thiết kế**."""
        entry = self._text_entry(runs, texts)
        return 0.0 if not entry else entry[3][0]

    def caret_offsets(self, run: dict, text: str) -> list:
        """TASK_small_fixes_1120: cumulative x advance (design px) BEFORE
        each character of ``text`` under ``run``'s font/size/scale/
        letter-spacing — ``offsets[0] == 0.0``, ``offsets[len(text)]`` ==
        this run's total width (matches ``text_width``/``line``'s own
        ``total``, since no two glyphs kern — see ``_text_entry``'s own
        comment). ``offsets[i]`` is where a caret sitting BEFORE
        character ``i`` belongs; this is the one measurement both the
        field's drawing AND its click-to-caret hit test use, so they can
        never land on different pixels for the same ``(text, i)``."""
        k = self.k
        size = float(run.get("fs", 16))
        metric = self.fonts.get(run.get("ff"), run.get("fw"), round(size * k))
        spacing = float(run.get("ls") or 0.0)
        text = text.upper() if run.get("upper") else text
        offsets = [0.0]
        for index, char in enumerate(text):
            extra = metric.getlength(char) / k
            if index >= 1:
                extra += spacing
            offsets.append(offsets[-1] + extra)
        return offsets

    # ---- hình khối bo góc ------------------------------------------------
    def rounded(self, rect, radius: float, colour) -> None:
        """
        Hình chữ nhật bo góc, **không** gọi PIL mỗi khung.

        Chia ba: hai đầu lấy từ một texture đệm cao đúng ``h`` và rộng
        ``2r+2``, khúc giữa là một ô tô màu phẳng. Thanh tiến độ đổi bề rộng
        mỗi khung nên dựng lại ảnh PIL theo từng bề rộng là đường chắc chắn
        vỡ ngưỡng FPS (luật F4).
        """
        x, y, w, h = (float(v) for v in rect)
        if w <= 0.0 or h <= 0.0:
            return
        radius = max(0.0, min(float(radius), h * 0.5, w * 0.5))
        if radius < 0.5:
            self.fill(rect, colour)
            return
        scale = self.scale
        cap_w = 2.0 * radius + 2.0
        key = ("cap", round(cap_w * scale), round(h * scale),
               round(radius * scale, 2))

        def build():
            width = max(2, int(round(cap_w * scale)))
            height = max(2, int(round(h * scale)))

            def draw(pen, big_w, big_h):
                pen.rounded_rectangle([0, 0, big_w - 1, big_h - 1],
                                      radius=radius * scale * _AA_FACTOR,
                                      fill=(255, 255, 255, 255))

            return _aa_build(width, height, draw), None

        entry = self.pil.get(key, build)
        if not entry:
            return
        texture = entry[0]
        half = radius + 1.0
        if w <= cap_w:
            # Hẹp hơn cả hai đầu: kéo nguyên ảnh đệm. README bảo thanh không
            # bao giờ hẹp hơn 2×radius, nên đây chỉ là lưới an toàn.
            self.quad((x, y, w, h), texture, colour)
            return
        middle = w - 2.0 * half
        self.quad((x, y, half, h), texture, colour, (0.0, 0.0, 0.5, 1.0))
        self.quad((x + half, y, middle, h), texture, colour,
                  (0.5, 0.0, 0.5, 1.0))
        self.quad((x + w - half, y, half, h), texture, colour,
                  (0.5, 0.0, 1.0, 1.0))

    def disc(self, rect, colour) -> None:
        """
        Hình tròn đặc, mềm biên. Texture TRẮNG đệm theo cỡ, màu đi qua tint.

        Vì sao không dùng ``pip``: khoá đệm của nó gồm cả màu, mà lỗ đạn thì
        mờ dần từng khung — mỗi khung một màu là mỗi khung một texture mới.
        Tách màu ra khỏi khoá thì cả vệt mờ dần chỉ tốn đúng một texture.
        """
        scale = self.scale
        size = max(1, int(round(float(rect[2]) * scale)))
        key = ("disc", size, max(1, int(round(float(rect[3]) * scale))))

        def build():
            width = size + 1
            height = max(1, int(round(float(rect[3]) * scale))) + 1

            def draw(pen, big_w, big_h):
                pen.ellipse([0, 0, big_w - 1, big_h - 1],
                           fill=(255, 255, 255, 255))

            return _aa_build(width, height, draw), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad((rect[0], rect[1], entry[1] / scale, entry[2] / scale),
                      entry[0], colour)

    def pip(self, rect, shape: str, radius: float, style: dict,
           alpha: float = 1.0) -> None:
        """
        Một pip: bầu dục hoặc chữ nhật bo góc, tô và/hoặc viền.

        ``fill_op`` được tôn trọng thật. ``preview.py`` vẽ pip mờ bằng PIL
        lên canvas RGBA rồi ``convert('RGB')`` ở cuối — thao tác đó **vứt**
        kênh alpha chứ không trộn, nên trong ``expected/`` pip "off" hiện ra
        đặc kịt. ``design/*.jpg`` (README: "the visual truth") cho thấy nó
        phải mờ, và đó là thứ dãy pip dùng để nói "lớp này chưa tới".
        """
        scale = self.k
        key = ("pip", shape, round(rect[2] * scale), round(rect[3] * scale),
               round(radius * scale, 2), style.get("fill"),
               round(float(style.get("fill_op", 1.0) or 1.0), 3),
               style.get("stroke"), round(float(style.get("sw", 1) or 1) * scale, 2))

        def build():
            width = max(1, int(round(rect[2] * scale))) + 1
            height = max(1, int(round(rect[3] * scale))) + 1
            fill = (_byte(style["fill"], float(style.get("fill_op", 1.0) or 1.0))
                    if style.get("fill") else None)
            outline = _byte(style["stroke"]) if style.get("stroke") else None

            def draw(pen, big_w, big_h):
                stroke_px = max(1, int(round(float(style.get("sw", 1) or 1)
                                             * scale * _AA_FACTOR)))
                box = [0, 0, big_w - 1, big_h - 1]
                if shape == "ellipse":
                    pen.ellipse(box, fill=fill, outline=outline, width=stroke_px)
                else:
                    pen.rounded_rectangle(box, radius=radius * scale * _AA_FACTOR,
                                          fill=fill, outline=outline, width=stroke_px)

            return _aa_build(width, height, draw), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad((rect[0], rect[1], entry[1] / scale, entry[2] / scale),
                      entry[0], (1.0, 1.0, 1.0, float(alpha)))

    def pill(self, record: dict, text: str, style: dict, alpha: float = 1.0) -> None:
        """
        Viên thuốc: nền bo góc + chữ canh giữa, dựng thành **một** texture.

        Bề rộng = bề rộng chữ + 2×``pad``, đúng công thức của preview.py (kể
        cả chỗ nó nhân ``ls`` với ``len`` chứ không phải ``len - 1``).
        """
        k = self.k
        scale = k
        font_spec = record["font"]
        x, y, _w, h = (float(v) for v in record["rect"])
        radius = float(record.get("radius", 0.0))
        pad = float(record.get("pad", 0.0))
        key = ("pill", text, font_spec["ff"], font_spec["fw"],
               round(float(font_spec["fs"]), 3),
               round(float(font_spec.get("ls") or 0.0), 3),
               style.get("bg"), style.get("fg"), round(h, 2), round(radius, 2),
               round(pad, 2), round(scale, 5))

        def build():
            size = float(font_spec["fs"])
            metric = self.fonts.get(font_spec["ff"], font_spec["fw"],
                                    round(size * k))
            spacing = float(font_spec.get("ls") or 0.0)
            # Công thức bề rộng của preview.py, kể cả chỗ nó nhân ``ls`` với
            # ``len`` chứ không phải ``len - 1``.
            design_w = metric.getlength(text) / k + spacing * len(text) + 2.0 * pad
            width = max(2, int(round(design_w * scale))) + 1
            height = max(2, int(round(h * scale))) + 1
            total = _run_width(metric, text, spacing * k) / k
            ascent_k, descent_k = metric.getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            colour = _byte(style.get("fg"))
            bg = _byte(style.get("bg"))

            def draw(pen, big_w, big_h):
                # Fix 11: the pill's TEXT gets supersampled right along
                # with its background — same canvas, same reduce() — a
                # sharper glyph at ``scale*_AA_FACTOR`` is a free upgrade
                # over FreeType's own hinting at the final (small) size.
                big_scale = scale * _AA_FACTOR
                glyph = self.fonts.get(font_spec["ff"], font_spec["fw"],
                                       round(size * big_scale))
                pen.rounded_rectangle([0, 0, big_w - 1, big_h - 1],
                                      radius=radius * big_scale, fill=bg)
                cursor = (big_w - 1 - total * big_scale) * 0.5
                baseline = ((big_h - 1 - (ascent + descent) * big_scale) * 0.5
                           + ascent * big_scale)
                for char in text:
                    pen.text((cursor, baseline), char, font=glyph, fill=colour,
                             anchor="ls")
                    cursor += (metric.getlength(char) / k + spacing) * big_scale

            return _aa_build(width, height, draw), design_w

        entry = self.pil.get(key, build)
        if entry:
            self.quad((x, y, entry[1] / scale, entry[2] / scale), entry[0],
                      (1.0, 1.0, 1.0, float(alpha)))

    def outlined_label(self, centre, text: str, font_spec: dict, fg: str,
                       outline_col: str, outline_px: float,
                       alpha: float = 1.0) -> None:
        """TASK_game_feel_fix_1118 Fix 10.3: free-floating outlined text,
        centred at ``centre`` — the scope's mini readout (ammo/timer/kills
        just outside the ring), no background chip (unlike ``pill``).
        PIL's own ``stroke_width``/``stroke_fill`` draw the outline —
        cached the same way every other PIL-built label here is
        (``_PilCache.get`` wants ``build()`` -> ``(array, meta)``; this one
        has no use for ``meta``, pixel dims come back via ``entry[1:3]``
        same as ``pill``'s own quad placement below)."""
        k = self.k
        scale = self.scale
        size = float(font_spec["fs"])
        spacing = float(font_spec.get("ls") or 0.0)
        stroke_px = max(1, int(round(outline_px * scale)))
        key = ("outlined_label", text, font_spec["ff"], font_spec["fw"],
              round(size, 3), round(spacing, 3), fg, outline_col, stroke_px,
              round(scale, 5))

        def build():
            metric = self.fonts.get(font_spec["ff"], font_spec["fw"], round(size * k))
            glyph = self.fonts.get(font_spec["ff"], font_spec["fw"], round(size * scale))
            total = _run_width(metric, text, spacing * k) / k
            ascent_k, descent_k = metric.getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            pad = stroke_px + 2
            design_w = total + 2.0 * pad / scale
            width = max(2, int(round(design_w * scale)))
            height = max(2, int(round((ascent + descent) * scale))) + 2 * pad
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            cursor = float(pad)
            baseline = float(pad) + ascent * scale
            fill, stroke_fill = _byte(fg), _byte(outline_col)
            for char in text:
                pen.text((cursor, baseline), char, font=glyph, fill=fill,
                         stroke_width=stroke_px, stroke_fill=stroke_fill,
                         anchor="ls")
                cursor += (metric.getlength(char) / k + spacing) * scale
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            cx, cy = float(centre[0]), float(centre[1])
            design_w, design_h = entry[1] / scale, entry[2] / scale
            self.quad((cx - design_w * 0.5, cy - design_h * 0.5, design_w, design_h),
                      entry[0], (1.0, 1.0, 1.0, float(alpha)))

    # ---- ảnh người dùng --------------------------------------------------
    def cover(self, rect, source, clip=None) -> bool:
        """Ảnh cover-fit (phủ kín, cắt giữa) vào ``rect``; ``clip`` thu hẹp."""
        entry = self.cache.image(source)
        if not entry:
            return False
        _tex, tex_w, tex_h = entry
        factor = max(rect[2] / tex_w, rect[3] / tex_h)
        width, height = tex_w * factor, tex_h * factor
        placed = (rect[0] + (rect[2] - width) * 0.5,
                  rect[1] + (rect[3] - height) * 0.5, width, height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        for area in (clip or [rect]):
            GL.glScissor(*self.viewport_px(area))
            self.quad(placed, entry[0])
        GL.glDisable(GL.GL_SCISSOR_TEST)
        return True

    def release(self) -> None:
        self._loader.stop()
        self.cache.release()
        self.pil.release()
        for entry in self._screen_tex.values():
            GL.glDeleteTextures(1, [entry[0]])
        self._screen_tex.clear()
        self._upload_backlog.clear()
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        self._shader.release()


def _run_width(font, text: str, spacing: float) -> float:
    """Bề rộng một run, kể cả letter-spacing (y như ``preview.run_width``)."""
    if spacing:
        return font.getlength(text) + spacing * max(0, len(text) - 1)
    return font.getlength(text)


