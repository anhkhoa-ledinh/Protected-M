"""
F3.3 — loại con bọ nào chiếu ra ngoài ô viewport của màn chơi.

Màn chơi chỉ hở một ô ``[66, 56, 868, 552]`` trong toạ độ thiết kế 1280×720;
bọ nằm ngoài ô đó thì người chơi **không bao giờ nhìn thấy**, nên ván không
thể thắng. Lọc ở đây chứ không sửa ``place_props``: bộ tìm chỗ giấu là của
PB-015, không thuộc phạm vi giao diện.

Phép chiếu: mắt nghỉ → vật → cắt mặt phẳng màn ở z = 0 (tam giác đồng dạng),
rồi đổi mm màn sang pixel thiết kế.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

#: Ô viewport của ``moi_06_man_choi`` trong toạ độ thiết kế.
VIEWPORT_RECT = (66.0, 56.0, 868.0, 552.0)
DESIGN_SIZE = (1280.0, 720.0)


def project_to_design(prop_mm, rest_eye_mm, screen_mm=(344.0, 194.0),
                      centre_mm=(0.0, 0.0)):
    """
    ``(x, y)`` pixel thiết kế của một vật, nhìn từ mắt nghỉ.

    ``screen_mm`` + ``centre_mm`` là **tấm kính đang vẽ**, không phải cả màn
    hình. Từ D2, chạy trong cửa sổ thì không gian thiết kế 1280×720 phủ đúng
    vùng khách của cửa sổ, mà vùng ấy nhỏ hơn màn và lệch tâm; bỏ qua hai
    tham số này là tâm ngắm lệch đi đúng khoảng cửa sổ bị dời.
    """
    eye_x, eye_y, eye_z = rest_eye_mm
    px, py, pz = prop_mm
    # Vật nằm sau màn (z < 0 theo trục HCPD, ở đây pz tính bằng mm âm dần).
    denom = eye_z - pz
    if abs(denom) < 1e-6:
        return None
    t = eye_z / denom                      # tỉ lệ tới mặt phẳng màn z = 0
    sx = eye_x + (px - eye_x) * t
    sy = eye_y + (py - eye_y) * t
    width_mm, height_mm = screen_mm
    return (((sx - centre_mm[0]) / width_mm + 0.5) * DESIGN_SIZE[0],
            (0.5 - (sy - centre_mm[1]) / height_mm) * DESIGN_SIZE[1])


def inside_viewport(point, rect=VIEWPORT_RECT) -> bool:
    """Điểm có nằm trong ô viewport không."""
    if point is None:
        return False
    x, y = point
    vx, vy, vw, vh = rect
    return vx <= x <= vx + vw and vy <= y <= vy + vh


def filter_props(props, rest_eye_mm, keep: int, screen_mm=(344.0, 194.0),
                 rect=VIEWPORT_RECT, centre_mm=(0.0, 0.0)):
    """
    Giữ tối đa ``keep`` con bọ chiếu vào trong ô; trả ``(giữ, số bị loại)``.

    Không bao giờ trả rỗng khi vẫn còn bọ: nếu **không** con nào lọt ô thì
    giữ nguyên danh sách và để người gọi báo số — thà có bọ khó thấy còn hơn
    một ván không có con nào.
    """
    inside, outside = [], 0
    for prop in props:
        point = project_to_design(tuple(prop["pos_mm"]), rest_eye_mm, screen_mm,
                                  centre_mm)
        if inside_viewport(point, rect):
            inside.append(prop)
        else:
            outside += 1
    if not inside:
        logger.warning("Loc viewport: KHONG con bo nao lot o — giu nguyen %d con",
                       len(props))
        return list(props)[:keep], outside
    logger.info("Loc viewport: %d con lot o, %d con bi loai", len(inside), outside)
    return inside[:keep], outside
