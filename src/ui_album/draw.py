"""
Lớp vẽ của giao diện album: mọi thứ là một quad có texture, nhân với một màu.

Thứ tự vẽ một màn (README của bộ asset):
1. ``base_<theme>.png``
2. ``photos[]`` — ảnh người chơi (cover-fit) hoặc vùng cảnh 3D
3. ``top_<theme>.png``
4. ``dyn[]`` — ô động, vẽ theo đúng kiểu chữ/màu đã ghi
5. khung chọn/hover ``primary`` dày 3 px
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw

from src.renderer.shader import ShaderProgram
from src.ui_album.fonts import FontBook, render_text
from src.ui_album.manifest import letterbox
from src.ui_album.textures import TextureCache

logger = logging.getLogger(__name__)

SELECT_PX = 3.0


class AlbumPainter:
    """Vẽ quad trong **không gian thiết kế**; tự lo letterbox và blend."""

    def __init__(self, manifest, shader_dir: Path) -> None:
        self.manifest = manifest
        self.cache = TextureCache()
        self.fonts = FontBook(manifest)
        self._shader = ShaderProgram(shader_dir / "album.vert",
                                     shader_dir / "album.frag")
        self._vao = self._vbo = 0
        self._box = letterbox(manifest.design_size, (1280, 720))
        self._window = (1280, 720)
        self._design = manifest.design_size

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

    def set_frame(self, window, design) -> None:
        """Cập nhật cỡ cửa sổ và cỡ màn thiết kế cho khung hình này."""
        self._window, self._design = window, design
        self._box = letterbox(design, window)

    @property
    def box(self):
        return self._box

    def viewport_px(self, rect) -> tuple[int, int, int, int]:
        """``rect`` thiết kế → ``(x, y, w, h)`` pixel GL (gốc dưới-trái)."""
        x, y, w, h = self._box.rect(rect)
        return (int(round(x)), int(round(self._window[1] - y - h)),
                max(1, int(round(w))), max(1, int(round(h))))

    def quad(self, rect, texture, tint=(1.0, 1.0, 1.0, 1.0)) -> None:
        """Vẽ một quad ở ``rect`` (toạ độ thiết kế) với texture và màu."""
        if texture is None:
            return
        x, y, w, h = self._box.rect(rect)
        win_w, win_h = self._window
        x0, x1 = 2.0 * x / win_w - 1.0, 2.0 * (x + w) / win_w - 1.0
        y0, y1 = 1.0 - 2.0 * y / win_h, 1.0 - 2.0 * (y + h) / win_h
        data = np.array([x0, y1, 0.0, 1.0, x1, y1, 1.0, 1.0,
                         x1, y0, 1.0, 0.0, x0, y1, 0.0, 1.0,
                         x1, y0, 1.0, 0.0, x0, y0, 0.0, 0.0], dtype=np.float32)
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

    def begin(self) -> None:
        """Chuẩn bị trạng thái GL cho lớp UI (2D, có alpha, không depth)."""
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def end(self) -> None:
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def fill(self, rect, colour) -> None:
        """Hình chữ nhật tô một màu."""
        entry = self.cache.white()
        if entry:
            self.quad(rect, entry[0], colour)

    def frame(self, rect, colour, width=SELECT_PX) -> None:
        """Khung rỗng (bốn thanh) — dùng cho ô đang chọn."""
        x, y, w, h = rect
        self.fill((x, y, w, width), colour)
        self.fill((x, y + h - width, w, width), colour)
        self.fill((x, y, width, h), colour)
        self.fill((x + w - width, y, width, h), colour)

    #: TASK_bug_art_0530: khung bo góc nhỏ hơn ngần này px thì bỏ (cùng lý do
    #: ``MIN_ELLIPSE_PX``).
    MIN_ROUNDED_FRAME_PX = 4.0

    def rounded_frame(self, rect, colour, width=SELECT_PX, radius=15.0) -> None:
        """
        Khung bo góc (``PIL.ImageDraw.rounded_rectangle``) — vòng focus của ô
        nhập, cùng bán kính với bo góc của chính ô (``01:18``: vòng focus cũ
        là bốn thanh vuông góc quanh một ô BO GÓC, nhìn lệch hẳn).

        Dựng bằng PIL rồi đưa lên GPU, đệm theo khoá — cùng khuôn với
        ``ellipse()``: siêu lấy mẫu ở độ phân giải ×2 cho nét mượt.
        """
        if rect[2] < self.MIN_ROUNDED_FRAME_PX or rect[3] < self.MIN_ROUNDED_FRAME_PX:
            return
        key = ("rframe", round(rect[2]), round(rect[3]), round(width, 2),
               round(radius, 2))

        def build():
            w_px = max(8, int(round(rect[2] * 2)))
            h_px = max(8, int(round(rect[3] * 2)))
            image = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            sw = max(1, int(round(width * 2)))
            half = sw / 2.0
            pen.rounded_rectangle(
                [half, half, w_px - 1 - half, h_px - 1 - half],
                radius=radius * 2.0, outline=(255, 255, 255, 255), width=sw)
            return np.asarray(image, dtype=np.uint8)

        entry = self.cache.small(key, build)
        if entry:
            self.quad(rect, entry[0], colour)

    #: Hình bầu dục nhỏ hơn ngần này pixel thì bỏ qua: PIL ném
    #: ``x1 must be greater than or equal to x0`` khi khung co về 0, và cú đó
    #: đã làm app văng giữa trận (06/11 23:51, chớp đầu nòng lúc tàn).
    MIN_ELLIPSE_PX = 2.0

    def ellipse(self, rect, fill=None, stroke=None, stroke_px=2.0) -> None:
        """Hình bầu dục vẽ bằng PIL rồi đưa lên GPU (có đệm theo khoá)."""
        if rect[2] < self.MIN_ELLIPSE_PX or rect[3] < self.MIN_ELLIPSE_PX:
            return
        key = ("ellipse", round(rect[2]), round(rect[3]),
               fill, stroke, round(stroke_px))

        def build():
            width = max(8, int(round(rect[2] * 2)))
            height = max(8, int(round(rect[3] * 2)))
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            pen.ellipse([1, 1, width - 2, height - 2],
                        fill=_byte(fill), outline=_byte(stroke),
                        width=max(1, int(round(stroke_px * 2))))
            return np.asarray(image, dtype=np.uint8)

        entry = self.cache.small(key, build)
        if entry:
            self.quad(rect, entry[0])

    def text(self, rect, value, style, colour, align="left") -> None:
        """Chữ; ``rect`` là khung bao trong toạ độ thiết kế."""
        key = ("text", str(value), style, tuple(round(c, 3) for c in colour))
        entry = self.cache.small(
            key, lambda: render_text(self.fonts, value, style, colour))
        if not entry:
            return
        _tex, tex_w, tex_h = entry
        scale = self._design[1] / 720.0        # bảng thành phần cao 960
        width, height = tex_w * scale, tex_h * scale
        x = rect[0]
        if align == "center":
            x = rect[0] + (rect[2] - width) * 0.5
        elif align == "right":
            x = rect[0] + rect[2] - width
        self.quad((x, rect[1], width, height), entry[0])

    def image(self, rect, path, tint=(1.0, 1.0, 1.0, 1.0)) -> bool:
        """Vẽ một file PNG phủ kín ``rect``. ``False`` nếu thiếu file."""
        entry = self.cache.image(path)
        if not entry:
            return False
        self.quad(rect, entry[0], tint)
        return True

    def cover(self, rect, path) -> bool:
        """Ảnh người chơi: cover-fit, cắt giữa — vẽ bằng cách cắt rect nguồn."""
        entry = self.cache.image(path)
        if not entry:
            return False
        _tex, tex_w, tex_h = entry
        scale = max(rect[2] / tex_w, rect[3] / tex_h)
        width, height = tex_w * scale, tex_h * scale
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(*self.viewport_px(rect))
        self.quad((rect[0] + (rect[2] - width) * 0.5,
                   rect[1] + (rect[3] - height) * 0.5, width, height),
                  entry[0])
        GL.glDisable(GL.GL_SCISSOR_TEST)
        return True

    def release(self) -> None:
        self.cache.release()
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        self._shader.release()


def _byte(colour):
    """``(r, g, b, a)`` 0–1 → bộ số 0–255 cho PIL; ``None`` giữ nguyên."""
    if colour is None:
        return None
    return tuple(int(round(c * 255)) for c in colour)
