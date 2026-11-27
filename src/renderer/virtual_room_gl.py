"""GL drawing of the virtual room (geometry and colours live in virtual_room)."""

from __future__ import annotations

import ctypes
from pathlib import Path

import numpy as np
from OpenGL import GL

from src.renderer import virtual_room as vr
from src.renderer.shader import ShaderProgram

#: Number of strips a side wall is cut into so its fill blends bottom to top.
_SIDE_STRIPS = 8


class VirtualRoomRenderer:
    """Five dark walls with grid lines, drawn first and without depth writes
    (the layers are then painted far to near over them). The walls flare with
    the photo scale ``s`` (see ``virtual_room``)."""

    def __init__(self, shader_dir: Path) -> None:
        self._shader = ShaderProgram(shader_dir / "room.vert", shader_dir / "room.frag")
        self._vao = GL.glGenVertexArrays(2)
        self._vbo = GL.glGenBuffers(2)
        self._counts = [0, 0]            # fill triangles, grid lines (vertices)
        self._key = None
        for i in range(2):
            GL.glBindVertexArray(self._vao[i])
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo[i])
            GL.glEnableVertexAttribArray(0)
            GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 24, None)
            GL.glEnableVertexAttribArray(1)
            GL.glVertexAttribPointer(1, 3, GL.GL_FLOAT, GL.GL_FALSE, 24,
                                     ctypes.c_void_p(12))
        GL.glBindVertexArray(0)

    # -- geometry ------------------------------------------------------------

    @staticmethod
    def build(colors: vr.RoomColors, hx: float, hy: float, depth_mm: float,
              centre_mm, s: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Interleaved (x, y, z, r, g, b) vertices in metres / 0..1:
        wall fills (triangles) and grid lines."""
        cx, cy = centre_mm
        k = 1.0 / 1000.0

        def v(p, rgb):
            return [(p[0] + cx) * k, (p[1] + cy) * k, p[2] * k,
                    rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0]

        fill: list[float] = []

        def quad(p00, p10, p11, p01, c0, c1):
            """Quad whose edge p00-p10 has colour c0 and p01-p11 colour c1."""
            for p, c in ((p00, c0), (p10, c0), (p11, c1),
                         (p00, c0), (p11, c1), (p01, c1)):
                fill.extend(v(p, c))

        d = -depth_mm
        corners = vr.room_corners(hx, hy, depth_mm, s)
        (fbl, fbr, ftr, ftl), (bbl, bbr, btr, btl) = corners["front"], corners["back"]
        cf, cc = colors.fill("floor"), colors.fill("ceiling")
        quad(fbl, fbr, bbr, bbl, cf, cf)
        quad(ftl, btl, btr, ftr, cc, cc)
        cb = colors.fill("back")
        quad(bbl, bbr, btr, btl, cb, cb)
        f = vr.flare(s)
        for x in (-hx, hx):
            for i in range(_SIDE_STRIPS):
                t0, t1 = i / _SIDE_STRIPS, (i + 1) / _SIDE_STRIPS
                y0, y1 = -hy + 2 * hy * t0, -hy + 2 * hy * t1
                c0, c1 = colors.fill("side", t0), colors.fill("side", t1)
                quad((x, y0, 0), (x * f, y0 * f, d), (x * f, y1 * f, d), (x, y1, 0),
                     c0, c1)

        lines: list[float] = []
        for name, segs in vr.grid_lines(hx, hy, depth_mm, s).items():
            for a, b in segs:
                for p in (a, b):
                    if name in ("left", "right"):
                        hh = hy * (1.0 + (f - 1.0) * abs(p[2]) / depth_mm)
                        t = (p[1] + hh) / (2 * hh)
                        rgb = colors.line("side", t)
                    else:
                        rgb = colors.line(name)
                    lines.extend(v(p, rgb))
        return (np.asarray(fill, dtype=np.float32),
                np.asarray(lines, dtype=np.float32))

    def update(self, colors: vr.RoomColors, hx: float, hy: float,
               depth_mm: float, centre_mm, s: float = 1.0) -> None:
        """Rebuild the buffers only when the photo, the room size or the
        flare (``max(1, s)``) changed."""
        key = (colors, round(hx, 3), round(hy, 3), round(depth_mm, 3),
               round(vr.flare(s), 4), tuple(round(c, 3) for c in centre_mm))
        if key == self._key:
            return
        self._key = key
        for i, buf in enumerate(self.build(colors, hx, hy, depth_mm, centre_mm, s)):
            GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo[i])
            GL.glBufferData(GL.GL_ARRAY_BUFFER, buf.nbytes, buf, GL.GL_STATIC_DRAW)
            self._counts[i] = buf.size // 6

    # -- draw ----------------------------------------------------------------

    def draw(self, view_flat, proj_flat) -> None:
        """Draw the walls. Restores the GL state it touches."""
        if self._counts[0] == 0:
            return
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDepthMask(GL.GL_FALSE)
        self._shader.use()
        self._shader.set_mat4("uView", view_flat)
        self._shader.set_mat4("uProjection", proj_flat)
        GL.glBindVertexArray(self._vao[0])
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, self._counts[0])
        GL.glBindVertexArray(self._vao[1])
        GL.glDrawArrays(GL.GL_LINES, 0, self._counts[1])
        GL.glBindVertexArray(0)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    def release(self) -> None:
        GL.glDeleteVertexArrays(2, self._vao)
        GL.glDeleteBuffers(2, self._vbo)
        self._shader.release()
