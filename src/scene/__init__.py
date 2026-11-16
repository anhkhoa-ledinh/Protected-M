"""Scene graph and room construction."""

from src.scene.node import Material, SceneNode
from src.scene.primitives import create_box, create_mesh, create_plane, create_sphere
from src.scene.room import build_room_scene
from src.scene.video_scene import build_video_window_scene

__all__ = [
    "Material",
    "SceneNode",
    "build_room_scene",
    "build_video_window_scene",
    "create_box",
    "create_mesh",
    "create_plane",
    "create_sphere",
    "create_vertical_plane",
]
