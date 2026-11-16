"""Video-through-window scene with layered depth cues for parallax."""

from __future__ import annotations

from src.config.schema import ContentSettings
from src.scene.node import Material, SceneNode
from src.scene.primitives import (
    create_mesh,
    create_vertical_plane,
    create_vertical_plane_at,
    create_window_frame_bezel,
)


#: Khoảng hở tối thiểu (m) giữa lớp xa nhất và mọi mặt của khung cảnh phía sau.
SCENE_CLEARANCE_M = 0.05


def farthest_content_z_m(content: ContentSettings) -> float:
    """
    Z (thế giới) của **mặt nội dung xa nhất** đang được vẽ.

    Lớp nằm ở ``plane_position_m.z + layer_z``, nên phải cộng cả hai; tắt tách
    lớp thì nội dung chỉ là một tấm ở ngay ``plane_position_m.z``.
    """
    plane_z = float(content.plane_position_m[2])
    if content.layer_clustering_enabled:
        return plane_z + min(float(content.layer_z_near_m),
                             float(content.layer_z_far_m))
    return plane_z


def behind_content_z_m(z_m: float, content: ContentSettings) -> float:
    """
    Đẩy một mặt của khung cảnh ra sau lớp xa nhất, nếu nó chưa ở sau.

    Khung viền và tấm nền là mặt **đục**, vẽ trước và có ghi depth, trong khi
    các lớp ảnh cố ý **không ghi depth**. Nên một mặt đục lọt vào giữa chồng
    lớp sẽ che mất ảnh — đúng cái khung chữ U đen hiện ra khi người dùng chọn
    ``layer_z_far_m`` = −0,60 (lớp xa −0,60 chui ra sau khung viền −0,58).

    Giá trị nào đã đủ sâu thì **giữ nguyên không đổi một chút nào**, nên cấu
    hình cũ (`layer_z_far_m` −0,35) dựng lại y hệt.
    """
    return min(float(z_m), farthest_content_z_m(content) - SCENE_CLEARANCE_M)


def build_video_window_scene(content: ContentSettings) -> SceneNode:
    """
    Void room with three depth layers plus a near window frame.

    Layer order (near → far): frame bezel, textured content plane (separate
    renderer), mid panels, far backdrop. Different Z positions make head
    motion parallax readable.
    """
    root = SceneNode(name="video_void")
    plane_h = content.resolved_plane_height_m()
    # Cả ba mặt dưới đây đều phải nằm SAU lớp xa nhất — xem ``behind_content_z_m``.

    root.children.append(
        SceneNode(
            name="backdrop",
            mesh=create_mesh(
                create_vertical_plane(
                    content.backdrop_width_m,
                    content.backdrop_height_m,
                    z=behind_content_z_m(content.backdrop_z_m, content),
                )
            ),
            material=Material(color=content.backdrop_color, shininess=4.0),
        )
    )

    mid_h = content.mid_layer_height_m
    mid_w = content.mid_layer_width_m
    mid_z = behind_content_z_m(content.mid_layer_z_m, content)
    offset = content.mid_layer_offset_x_m
    root.children.append(
        SceneNode(
            name="mid_panel_left",
            mesh=create_mesh(
                create_vertical_plane_at(-offset, 0.05, mid_w, mid_h, mid_z)
            ),
            material=Material(color=content.mid_layer_color_a, shininess=8.0),
        )
    )
    root.children.append(
        SceneNode(
            name="mid_panel_right",
            mesh=create_mesh(
                create_vertical_plane_at(offset, -0.08, mid_w * 0.85, mid_h * 0.9, mid_z)
            ),
            material=Material(color=content.mid_layer_color_b, shininess=8.0),
        )
    )

    root.children.append(
        SceneNode(
            name="window_frame",
            mesh=create_mesh(
                create_window_frame_bezel(
                    content.plane_width_m,
                    plane_h,
                    content.frame_bezel_m,
                    behind_content_z_m(content.frame_z_m, content),
                )
            ),
            material=Material(color=content.frame_color, shininess=24.0),
        )
    )

    return root
