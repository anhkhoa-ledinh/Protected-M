"""Debug line rendering for screen plane and frustum."""

from __future__ import annotations

import numpy as np
from OpenGL import GL

from src.renderer.shader import ShaderProgram


class DebugGeometryRenderer:
    """Dynamic GL_LINES renderer for debug visualization."""

    def __init__(self, shader: ShaderProgram) -> None:
        self._shader = shader
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        self._count = 0

        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 0, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, 12, None)
        GL.glBindVertexArray(0)

    def upload_segments(self, segments: list[tuple[np.ndarray, np.ndarray]]) -> None:
        """Upload world-space line segments."""
        if not segments:
            self._count = 0
            return

        vertices: list[float] = []
        for start, end in segments:
            vertices.extend([float(start[0]), float(start[1]), float(start[2])])
            vertices.extend([float(end[0]), float(end[1]), float(end[2])])
        buffer = np.asarray(vertices, dtype=np.float32)

        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, buffer.nbytes, buffer, GL.GL_DYNAMIC_DRAW)
        self._count = buffer.size // 3

    def draw(self, view_flat: np.ndarray, proj_flat: np.ndarray) -> None:
        """Draw uploaded line segments."""
        if self._count <= 0:
            return

        GL.glDisable(GL.GL_CULL_FACE)
        GL.glDisable(GL.GL_DEPTH_TEST)
        self._shader.use()
        self._shader.set_mat4("uView", view_flat)
        self._shader.set_mat4("uProjection", proj_flat)
        self._shader.set_vec3("uColor", 0.1, 1.0, 0.8)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_LINES, 0, self._count)
        GL.glBindVertexArray(0)
        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)

    def release(self) -> None:
        """Free GL buffers."""
        GL.glDeleteVertexArrays(1, [self._vao])
        GL.glDeleteBuffers(1, [self._vbo])
