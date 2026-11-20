"""OpenGL rendering, shaders, and scene graph.

Exports are resolved lazily (PEP 562) so importing a leaf module such as
``src.renderer.obj_loader`` does not eagerly pull in ``SceneRenderer`` —
that heavy import chain reaches back into ``src.scene`` and would create a
circular import when a scene module is imported first (e.g. from tests).

``Mesh``/``MeshData`` (TASK_arch_refactor_1120 R2: moved to
``src.scene.mesh``, alongside ``TexturedMesh``/``TexturedMeshData`` in
``src.scene.textured_mesh`` — ``scene`` no longer needs anything from
``renderer`` for them) are no longer exported here; import them from
``src.scene.mesh`` directly.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING

_EXPORTS = {
    "VirtualCamera": "src.renderer.camera",
    "DebugGeometryRenderer": "src.renderer.debug_geometry",
    "DebugMetrics": "src.renderer.debug_overlay",
    "DebugOverlay": "src.renderer.debug_overlay",
    "load_obj": "src.renderer.obj_loader",
    "load_obj_mesh": "src.renderer.obj_loader",
    "SceneRenderer": "src.renderer.renderer",
    "ShaderProgram": "src.renderer.shader",
    "GlfwWindow": "src.renderer.window",
    "InputState": "src.renderer.window",
}

if TYPE_CHECKING:
    from src.renderer.camera import VirtualCamera
    from src.renderer.debug_geometry import DebugGeometryRenderer
    from src.renderer.debug_overlay import DebugMetrics, DebugOverlay
    from src.renderer.obj_loader import load_obj, load_obj_mesh
    from src.renderer.renderer import SceneRenderer
    from src.renderer.shader import ShaderProgram
    from src.renderer.window import GlfwWindow, InputState


def __getattr__(name: str):
    module_path = _EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = importlib.import_module(module_path)
    return getattr(module, name)


def __dir__() -> list[str]:
    return sorted(_EXPORTS)


__all__ = [
    "DebugMetrics",
    "DebugGeometryRenderer",
    "DebugOverlay",
    "GlfwWindow",
    "InputState",
    "SceneRenderer",
    "ShaderProgram",
    "VirtualCamera",
    "load_obj",
    "load_obj_mesh",
]
