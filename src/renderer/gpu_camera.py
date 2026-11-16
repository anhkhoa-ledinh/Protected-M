"""GPU-side camera uniform buffer for shader pipeline."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from src.math.homogeneous import Mat4, multiply, to_column_major_flat


@dataclass(frozen=True)
class GpuCameraState:
    """Camera matrices and screen metadata uploaded to GPU."""

    view: Mat4
    projection: Mat4
    eye_position_mm: npt.NDArray[np.float64]
    near: float
    far: float
    screen_width_mm: float
    screen_height_mm: float

    @property
    def view_projection(self) -> Mat4:
        return multiply(self.projection, self.view)


class GpuCameraBuffer:
    """
    Manages std140 uniform buffer for view/projection/eye data.

    Keeps camera updates on the GPU path so the render thread only
    uploads one UBO per frame instead of many loose uniforms.
    """

    UBO_BINDING_POINT = 0
    FLOATS_PER_MAT4 = 16
    BLOCK_FLOAT_COUNT = FLOATS_PER_MAT4 * 3 + 8

    def __init__(self) -> None:
        self._buffer_id: int | None = None
        self._initialized = False
        self._cpu_cache = np.zeros(self.BLOCK_FLOAT_COUNT, dtype=np.float32)

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self) -> None:
        """Create OpenGL uniform buffer object."""
        from OpenGL import GL

        if self._initialized:
            return

        buffer_id = GL.glGenBuffers(1)
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, buffer_id)
        GL.glBufferData(
            GL.GL_UNIFORM_BUFFER,
            self._cpu_cache.nbytes,
            self._cpu_cache,
            GL.GL_DYNAMIC_DRAW,
        )
        GL.glBindBufferBase(
            GL.GL_UNIFORM_BUFFER,
            self.UBO_BINDING_POINT,
            buffer_id,
        )
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

        self._buffer_id = int(buffer_id)
        self._initialized = True

    def update(self, state: GpuCameraState) -> None:
        """Upload camera state to GPU uniform buffer."""
        self._write_cpu_cache(state)
        if not self._initialized:
            return

        from OpenGL import GL

        assert self._buffer_id is not None
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, self._buffer_id)
        GL.glBufferSubData(
            GL.GL_UNIFORM_BUFFER,
            0,
            self._cpu_cache.nbytes,
            self._cpu_cache,
        )
        GL.glBindBuffer(GL.GL_UNIFORM_BUFFER, 0)

    def bind_block(self, program: int, block_name: str = "CameraBlock") -> None:
        """Associate UBO with shader program uniform block."""
        from OpenGL import GL

        block_index = GL.glGetUniformBlockIndex(program, block_name)
        if block_index == GL.GL_INVALID_INDEX:
            raise RuntimeError(f"Uniform block '{block_name}' not found in shader")
        GL.glUniformBlockBinding(program, block_index, self.UBO_BINDING_POINT)

    def _write_cpu_cache(self, state: GpuCameraState) -> None:
        """Pack std140 layout into CPU-side staging buffer."""
        offset = 0
        offset = self._pack_matrix(state.view, offset)
        offset = self._pack_matrix(state.projection, offset)
        offset = self._pack_matrix(state.view_projection, offset)

        eye = state.eye_position_mm.astype(np.float32)
        self._cpu_cache[offset : offset + 3] = eye
        self._cpu_cache[offset + 3] = np.float32(state.near)

        self._cpu_cache[offset + 4 : offset + 7] = np.float32(
            [state.screen_width_mm, state.screen_height_mm, 0.0]
        )
        self._cpu_cache[offset + 7] = np.float32(state.far)

    def _pack_matrix(self, matrix: Mat4, offset: int) -> int:
        flat = to_column_major_flat(matrix)
        end = offset + self.FLOATS_PER_MAT4
        self._cpu_cache[offset:end] = flat
        return end

    def release(self) -> None:
        """Delete GPU resources."""
        if not self._initialized or self._buffer_id is None:
            return

        from OpenGL import GL

        GL.glDeleteBuffers(1, [self._buffer_id])
        self._buffer_id = None
        self._initialized = False
