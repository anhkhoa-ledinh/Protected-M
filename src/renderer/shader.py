"""GLSL shader compilation and linking."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import numpy.typing as npt
from OpenGL import GL


class ShaderProgram:
    """Compiled and linked OpenGL shader program with cached uniform locations."""

    def __init__(self, vertex_path: Path, fragment_path: Path) -> None:
        self._program_id = self._build(vertex_path, fragment_path)
        self._uniform_cache: dict[str, int] = {}

    @property
    def program_id(self) -> int:
        return self._program_id

    def use(self) -> None:
        """Activate this shader program."""
        GL.glUseProgram(self._program_id)

    def set_mat4(self, name: str, matrix: npt.NDArray[np.float32]) -> None:
        """Upload a 4x4 matrix uniform."""
        location = self._location(name)
        GL.glUniformMatrix4fv(location, 1, GL.GL_TRUE, matrix)

    def set_vec3(self, name: str, x: float, y: float, z: float) -> None:
        """Upload a vec3 uniform."""
        location = self._location(name)
        GL.glUniform3f(location, x, y, z)

    def set_float(self, name: str, value: float) -> None:
        """Upload a float uniform."""
        location = self._location(name)
        GL.glUniform1f(location, value)

    def release(self) -> None:
        """Delete GPU program object."""
        if self._program_id:
            GL.glDeleteProgram(self._program_id)
            self._program_id = 0
            self._uniform_cache.clear()

    def _location(self, name: str) -> int:
        if name not in self._uniform_cache:
            self._uniform_cache[name] = GL.glGetUniformLocation(self._program_id, name)
        return self._uniform_cache[name]

    @staticmethod
    def _build(vertex_path: Path, fragment_path: Path) -> int:
        vertex_source = vertex_path.read_text(encoding="utf-8")
        fragment_source = fragment_path.read_text(encoding="utf-8")

        vertex_shader = ShaderProgram._compile(
            vertex_source,
            GL.GL_VERTEX_SHADER,
            vertex_path.name,
        )
        fragment_shader = ShaderProgram._compile(
            fragment_source,
            GL.GL_FRAGMENT_SHADER,
            fragment_path.name,
        )

        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex_shader)
        GL.glAttachShader(program, fragment_shader)
        GL.glLinkProgram(program)

        if GL.glGetProgramiv(program, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            info = GL.glGetProgramInfoLog(program).decode("utf-8")
            raise RuntimeError(f"Shader link failed: {info}")

        GL.glDeleteShader(vertex_shader)
        GL.glDeleteShader(fragment_shader)
        return int(program)

    @staticmethod
    def _compile(source: str, shader_type: int, name: str) -> int:
        shader = GL.glCreateShader(shader_type)
        GL.glShaderSource(shader, source)
        GL.glCompileShader(shader)
        if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
            info = GL.glGetShaderInfoLog(shader).decode("utf-8")
            raise RuntimeError(f"Shader compile failed ({name}): {info}")
        return int(shader)
