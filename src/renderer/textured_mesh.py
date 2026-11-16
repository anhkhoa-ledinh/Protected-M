"""GPU mesh with position and UV attributes for textured quads."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from OpenGL import GL


@dataclass(frozen=True)
class TexturedMeshData:
    """CPU-side interleaved pos(3) + uv(2) vertex data."""

    vertices: npt.NDArray[np.float32]
    indices: npt.NDArray[np.uint32]
    vertex_count: int


class TexturedMesh:
    """GPU mesh for textured surfaces (video plane)."""

    STRIDE = 5

    def __init__(self, data: TexturedMeshData) -> None:
        self._index_count = int(data.indices.shape[0])
        self._vao = GL.glGenVertexArrays(1)
        vbo = GL.glGenBuffers(1)
        ebo = GL.glGenBuffers(1)

        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, vbo)
        GL.glBufferData(
            GL.GL_ARRAY_BUFFER,
            data.vertices.nbytes,
            data.vertices,
            GL.GL_STATIC_DRAW,
        )
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(
            GL.GL_ELEMENT_ARRAY_BUFFER,
            data.indices.nbytes,
            data.indices,
            GL.GL_STATIC_DRAW,
        )
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 3, GL.GL_FLOAT, GL.GL_FALSE, self.STRIDE * 4, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(
            1, 2, GL.GL_FLOAT, GL.GL_FALSE, self.STRIDE * 4, GL.ctypes.c_void_p(12)
        )
        GL.glBindVertexArray(0)

        self._vbo = int(vbo)
        self._ebo = int(ebo)

    def draw(self) -> None:
        GL.glBindVertexArray(self._vao)
        GL.glDrawElements(GL.GL_TRIANGLES, self._index_count, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)

    def release(self) -> None:
        GL.glDeleteVertexArrays(1, [self._vao])
        GL.glDeleteBuffers(1, [self._vbo])
        GL.glDeleteBuffers(1, [self._ebo])
