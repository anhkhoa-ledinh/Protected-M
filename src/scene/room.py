"""Virtual room scene with furniture and props."""

from __future__ import annotations

import numpy as np

from src.scene.node import Material, SceneNode
from src.scene.primitives import create_box, create_mesh, create_plane, create_sphere


def build_room_scene() -> SceneNode:
    """
    Build a small furnished room demonstrating motion parallax.

    Room extends into negative Z (through the monitor window).
    """
    root = SceneNode(name="room")

    floor_mesh = create_mesh(create_plane(6.0, 8.0, y=0.0))
    root.children.append(
        SceneNode(
            name="floor",
            mesh=floor_mesh,
            material=Material(color=(0.35, 0.32, 0.28), shininess=16.0),
        )
    )

    ceiling_mesh = create_mesh(create_plane(6.0, 8.0, y=0.0))
    root.children.append(
        SceneNode(
            name="ceiling",
            mesh=ceiling_mesh,
            translation=np.array([0.0, 3.0, 0.0]),
            material=Material(color=(0.45, 0.45, 0.48), shininess=8.0),
        )
    )

    wall_color = (0.55, 0.58, 0.62)
    back_wall = SceneNode(
        name="back_wall",
        mesh=create_mesh(create_box(6.0, 3.0, 0.15)),
        translation=np.array([0.0, 1.5, -4.0]),
        material=Material(color=wall_color, shininess=12.0),
    )
    left_wall = SceneNode(
        name="left_wall",
        mesh=create_mesh(create_box(0.15, 3.0, 8.0)),
        translation=np.array([-3.0, 1.5, -2.0]),
        material=Material(color=(0.5, 0.53, 0.56), shininess=12.0),
    )
    right_wall = SceneNode(
        name="right_wall",
        mesh=create_mesh(create_box(0.15, 3.0, 8.0)),
        translation=np.array([3.0, 1.5, -2.0]),
        material=Material(color=(0.52, 0.55, 0.58), shininess=12.0),
    )
    root.children.extend([back_wall, left_wall, right_wall])

    root.children.append(
        SceneNode(
            name="near_cube",
            mesh=create_mesh(create_box(0.6, 0.6, 0.6)),
            translation=np.array([-0.8, 0.3, -0.8]),
            material=Material(color=(0.85, 0.35, 0.25), shininess=64.0),
        )
    )
    root.children.append(
        SceneNode(
            name="mid_sphere",
            mesh=create_mesh(create_sphere(0.45)),
            translation=np.array([0.6, 0.45, -1.8]),
            material=Material(color=(0.25, 0.55, 0.85), shininess=96.0),
        )
    )
    root.children.append(
        SceneNode(
            name="far_sphere",
            mesh=create_mesh(create_sphere(0.35)),
            translation=np.array([0.0, 0.35, -3.2]),
            material=Material(color=(0.75, 0.75, 0.2), shininess=48.0),
        )
    )

    table = SceneNode(
        name="table",
        mesh=create_mesh(create_box(1.2, 0.08, 0.7)),
        translation=np.array([-1.5, 0.75, -1.5]),
        material=Material(color=(0.45, 0.28, 0.15), shininess=24.0),
    )
    table.children.append(
        SceneNode(
            name="table_leg",
            mesh=create_mesh(create_box(0.08, 0.75, 0.08)),
            translation=np.array([0.0, -0.41, 0.0]),
            material=Material(color=(0.35, 0.22, 0.12), shininess=16.0),
        )
    )
    root.children.append(table)

    chair = SceneNode(
        name="chair_seat",
        mesh=create_mesh(create_box(0.55, 0.06, 0.55)),
        translation=np.array([1.4, 0.45, -1.2]),
        material=Material(color=(0.2, 0.25, 0.55), shininess=20.0),
    )
    chair.children.append(
        SceneNode(
            name="chair_back",
            mesh=create_mesh(create_box(0.55, 0.7, 0.06)),
            translation=np.array([0.0, 0.38, -0.24]),
            material=Material(color=(0.18, 0.22, 0.5), shininess=20.0),
        )
    )
    root.children.append(chair)

    statue = SceneNode(
        name="statue_base",
        mesh=create_mesh(create_box(0.35, 0.15, 0.35)),
        translation=np.array([2.0, 0.075, -2.5]),
        material=Material(color=(0.6, 0.6, 0.62), shininess=40.0),
    )
    statue.children.append(
        SceneNode(
            name="statue_body",
            mesh=create_mesh(create_sphere(0.22, segments=16, rings=12)),
            translation=np.array([0.0, 0.35, 0.0]),
            material=Material(color=(0.72, 0.7, 0.68), shininess=80.0),
        )
    )
    root.children.append(statue)

    monkey = SceneNode(
        name="monkey",
        mesh=create_mesh(create_sphere(0.28, segments=20, rings=14)),
        translation=np.array([-0.2, 1.1, -2.8]),
        material=Material(color=(0.55, 0.35, 0.2), shininess=56.0),
    )
    root.children.append(monkey)

    return root
