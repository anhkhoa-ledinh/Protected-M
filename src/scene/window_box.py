"""
PB-034 — hộp quanh các lớp: miệng hộp **đúng bằng hình chữ nhật màn hình**.

Khác `build_video_window_scene` (void): ở đó cảnh là mấy tấm trôi trong khoảng
không, mép ảnh và khoảng trống lộ ra khi đầu dịch. Ở đây người xem nhìn vào
một cái hộp mở, bốn vách bao trọn tầm nhìn nên **không mép lớp nào lọt ra
được** — và vì thế `overscan_e_max_mm` về 0, không cần phóng lớp nữa.

Miệng hộp nằm ở z = 0, đúng mặt phẳng màn hình, đúng kích thước màn
(`screen.json`). Đó là điều kiện để bốn góc miệng chiếu trúng bốn góc khung
nhìn với mọi vị trí mắt — xem `tests/test_window_box.py`.
"""

from __future__ import annotations

from src.config.schema import ContentSettings
from src.scene.node import Material, SceneNode
from src.scene.video_scene import (behind_content_z_m,
                                   build_video_window_scene)
from src.scene.primitives import create_mesh, create_open_box


def box_cross_section_m(screen_w_m: float, screen_h_m: float,
                        image_aspect: float) -> float:
    """
    Bề rộng quad lớp để nó phủ kín tiết diện hộp: ``max(W, H × tỉ lệ ảnh)``.

    Ảnh ngang hơn màn thì chiều rộng quyết định; ảnh dọc hơn thì chiều cao
    quyết định, và quad phải rộng ra theo tỉ lệ ảnh cho khỏi hở hai bên.
    """
    aspect = float(image_aspect) if image_aspect and image_aspect > 0 else 1.0
    return max(float(screen_w_m), float(screen_h_m) * aspect)


def back_wall_z_m(content: ContentSettings) -> float:
    """
    Z của vách sau: ngay sau lớp xa nhất, **dùng chung đúng một công thức** với
    khung viền và tấm nền của chế độ void (``behind_content_z_m``).

    Bản đầu tự tính ``layer_z_far_m − 0,05`` và **bỏ quên ``plane_position_m.z``**
    — không lộ ra vì demo để plane Z = 0, nhưng sai với mọi cấu hình khác.
    """
    return behind_content_z_m(float(content.layer_z_far_m), content)


def box_layer_scale(t_mm: float, base_w_m: float, base_h_m: float,
                    screen_w_mm: float, screen_h_mm: float,
                    design_d_mm: float, offset_mm) -> float:
    """
    Hệ số phóng quad của một lớp ở chế độ hộp — **hai ràng buộc, lấy cái lớn**.

    1. *Khớp hình*: lớp sâu ``t`` phải phóng ``(Dd + t)/Dd`` thì nhìn từ mắt
       thiết kế nó mới choán đúng góc như lớp ở mặt kính; thiếu hệ số này thì
       các lớp lệch nhau ngay giữa ảnh.
    2. *Kín lòng hộp*: quad đã **dời tâm** theo mắt thiết kế ``offset_mm``, nên
       mép bên kia lùi vào đúng chừng ấy. Nửa quad phải còn phủ hết tiết diện
       ống hộp (``W/2 + |ox|``, ``H/2 + |oy|``), không thì hở ra một dải void.
       Trên demo, lớp xa dời −77,8 mm theo trục y và ràng buộc này mới là cái
       quyết định — chỉ (1) thì hụt 2,3 mm.

    Không cần biên ``e_max``: mọi tia nhìn ra ngoài ống hộp đều đã đâm vào vách
    trước khi tới độ sâu ``t``, nên vách che, không phải quad che.
    """
    d = max(float(design_d_mm), 1.0)
    depth_k = (d + max(float(t_mm), 0.0)) / d
    aspect = base_w_m / max(base_h_m, 1e-9)
    fit_w_m = box_cross_section_m(screen_w_mm / 1000.0, screen_h_mm / 1000.0,
                                  aspect) * depth_k
    cover_w_m = (screen_w_mm + 2.0 * abs(float(offset_mm[0]))) / 1000.0
    cover_h_m = (screen_h_mm + 2.0 * abs(float(offset_mm[1]))) / 1000.0
    return max(fit_w_m / max(base_w_m, 1e-9),
               cover_w_m / max(base_w_m, 1e-9),
               cover_h_m / max(base_h_m, 1e-9))


def build_window_box_scene(content: ContentSettings, screen_w_m: float,
                           screen_h_m: float) -> SceneNode:
    """Hộp mở năm mặt; miệng = hình chữ nhật màn hình ở z = 0."""
    root = SceneNode(name="window_box")
    root.children.append(
        SceneNode(
            name="box_walls",
            mesh=create_mesh(create_open_box(
                float(screen_w_m), float(screen_h_m), 0.0,
                back_wall_z_m(content))),
            material=Material(color=content.backdrop_color, shininess=4.0),
        )
    )
    return root


def build_scene_for(content: ContentSettings, screen_w_m: float,
                    screen_h_m: float) -> SceneNode:
    """
    Dựng cảnh bao quanh theo khoá ``surround``. **Một chỗ quyết định duy nhất**
    cho cả `app.py` lẫn `renderer.py`, nên hai nơi không thể lệch nhau.
    """
    if str(content.surround).lower().strip() == "box":
        return build_window_box_scene(content, screen_w_m, screen_h_m)
    return build_video_window_scene(content)
