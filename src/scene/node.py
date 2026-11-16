"""Scene graph node hierarchy."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.math.homogeneous import Mat4, multiply, translation_matrix
from src.renderer.mesh import Mesh


@dataclass
class Material:
    """Simple Phong material parameters."""

    color: tuple[float, float, float] = (0.8, 0.8, 0.8)
    shininess: float = 32.0


@dataclass
class SceneNode:
    """Transform node with optional renderable mesh."""

    name: str
    mesh: Mesh | None = None
    material: Material = field(default_factory=Material)
    translation: np.ndarray = field(default_factory=lambda: np.zeros(3))
    rotation_y: float = 0.0
    scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    children: list[SceneNode] = field(default_factory=list)

    def world_matrix(self, parent: Mat4 | None = None) -> Mat4:
        """Compute accumulated world transform."""
        local = translation_matrix(
            float(self.translation[0]),
            float(self.translation[1]),
            float(self.translation[2]),
        )
        if abs(self.rotation_y) > 1e-8:
            cos_y = float(np.cos(self.rotation_y))
            sin_y = float(np.sin(self.rotation_y))
            rot = np.eye(4, dtype=np.float64)
            rot[0, 0] = cos_y
            rot[0, 2] = sin_y
            rot[2, 0] = -sin_y
            rot[2, 2] = cos_y
            local = multiply(local, rot)

        scale_mat = np.eye(4, dtype=np.float64)
        scale_mat[0, 0] = float(self.scale[0])
        scale_mat[1, 1] = float(self.scale[1])
        scale_mat[2, 2] = float(self.scale[2])
        local = multiply(local, scale_mat)

        if parent is None:
            return local
        return multiply(parent, local)

    def traverse(self, parent: Mat4 | None = None):
        """Yield (node, world_matrix) pairs in depth-first order."""
        world = self.world_matrix(parent)
        yield self, world
        for child in self.children:
            yield from child.traverse(world)
