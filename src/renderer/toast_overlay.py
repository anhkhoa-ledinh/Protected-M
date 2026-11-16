"""Brief on-screen status messages (load success / errors)."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
from OpenGL import GL

from src.renderer.shader import ShaderProgram


class ToastOverlay:
    """Top-center HUD toast shown for a few seconds."""

    def __init__(self, shader_dir: Path, duration_s: float = 2.0) -> None:
        self._shader = ShaderProgram(shader_dir / "hud.vert", shader_dir / "hud.frag")
        self._texture = 0
        self._vao = 0
        self._vbo = 0
        self._ebo = 0
        self._duration_s = duration_s
        self._message = ""
        self._expires_at = 0.0
        self._ready = False
        self._tex_width = 0
        self._tex_height = 0

    def show(self, message: str, *, is_error: bool = False) -> None:
        prefix = "Error: " if is_error else ""
        self._message = f"{prefix}{message}"
        self._expires_at = time.perf_counter() + self._duration_s
        self._render_texture(is_error)

    def initialize(self) -> None:
        self._texture = GL.glGenTextures(1)
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._ebo = GL.glGenBuffers(1)
        indices = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 64, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, self._ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)
        self._ready = True

    def draw(self, window_width: int, window_height: int) -> None:
        if not self._ready or time.perf_counter() > self._expires_at:
            return
        if self._tex_width == 0:
            return

        pad = 16.0
        tw = min(window_width * 0.55, self._tex_width + pad * 2)
        th = self._tex_height + pad
        x0 = 1.0 - (tw / window_width)
        x1 = 1.0 - 8.0 / window_width
        y1 = 1.0 - 12.0 / window_height
        y0 = y1 - th / window_height

        # V is intentionally inverted: glTexImage2D treats array row 0 (the
        # text canvas TOP) as v=0, while y0 here is the BOTTOM edge of the
        # quad.  Mapping v=1 onto y0 keeps the message upright.
        vertices = np.array(
            [
                x0, y0, 0.0, 1.0,
                x1, y0, 1.0, 1.0,
                x1, y1, 1.0, 0.0,
                x0, y1, 0.0, 0.0,
            ],
            dtype=np.float32,
        )
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        self._shader.use()
        tex_loc = GL.glGetUniformLocation(self._shader.program_id, "uTexture")
        GL.glUniform1i(tex_loc, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def release(self) -> None:
        if self._texture:
            GL.glDeleteTextures(1, [self._texture])
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        if self._ebo:
            GL.glDeleteBuffers(1, [self._ebo])
        self._shader.release()

    def _render_texture(self, is_error: bool) -> None:
        if not self._ready:
            return
        canvas_h = 56
        font_scale = 0.55
        thickness = 1
        (tw, th), _ = cv2.getTextSize(
            self._message, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        canvas_w = max(tw + 24, 180)
        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
        bg = (40, 28, 28) if is_error else (24, 32, 28)
        fg = (120, 160, 255) if is_error else (120, 255, 180)
        canvas[:] = bg
        y = (canvas_h + th) // 2 - 2
        cv2.putText(
            canvas, self._message, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness + 1, cv2.LINE_AA,
        )
        cv2.putText(
            canvas, self._message, (12, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, fg, thickness, cv2.LINE_AA,
        )
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        h, w = rgb.shape[:2]
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb,
        )
        self._tex_width = w
        self._tex_height = h
