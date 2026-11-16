"""
Bộ lọc bọ phải dùng ô viewport của **màn chơi**, không phải màn đang mở.

``_start_album_match`` chạy lúc người chơi còn đứng ở màn thử nghiêng. Ô
viewport ở đó là ``[78, 166, 632, 380]``, nhỏ hơn hẳn ô ``[28, 28, 1224, 664]``
của màn chơi. Lấy nhầm ô nhỏ thì ``filter_props`` vứt gần hết bọ — log phiên
07/11: "xin 6 · đặt 1 · ngoài ô 7", vào trận cấp 1 chỉ có đúng một con bọ.

Số liệu dưới đây là **vị trí bọ thật** của phiên ấy, chép từ log.
"""

from __future__ import annotations

from src.ui_album.viewport_filter import filter_props, project_to_design

#: Tám con bọ của phiên 07/11, mm trong hệ HCPD.
PROPS = [{"pos_mm": p} for p in [
    (0, -19, -531), (-103, 0, -500), (69, 19, -500), (-34, -78, -304),
    (34, 58, -500), (34, -78, -78), (69, -19, -500), (-34, 39, -469)]]

REST_EYE = (0.0, 0.0, 450.0)

#: Tấm kính lúc chạy trong cửa sổ 1280×720 trên màn 1366×768.
GLASS, CENTRE = (322.3, 181.9), (0.0, 0.0)

PLAY_RECT = (28, 28, 1224, 664)
TILT_RECT = (78, 166, 632, 380)


def test_play_rect_keeps_every_bug():
    """Ô của màn chơi gần kín khung hình — không con nào bị loại."""
    kept, outside = filter_props(PROPS, REST_EYE, 6, GLASS, PLAY_RECT, CENTRE)
    assert outside == 0
    assert len(kept) == 6                      # xin 6 thì được đúng 6


def test_tilt_rect_would_throw_bugs_away():
    """Ô của màn thử nghiêng loại bớt — đây chính là cái đã xảy ra."""
    _kept, outside = filter_props(PROPS, REST_EYE, 6, GLASS, TILT_RECT, CENTRE)
    assert outside > 0


def test_every_bug_projects_inside_the_play_rect():
    """Từng con một, không chỉ tính tổng: điểm chiếu nằm trong ô màn chơi."""
    x, y, w, h = PLAY_RECT
    for prop in PROPS:
        point = project_to_design(prop["pos_mm"], REST_EYE, GLASS, CENTRE)
        assert point is not None
        assert x <= point[0] <= x + w, prop["pos_mm"]
        assert y <= point[1] <= y + h, prop["pos_mm"]


def test_window_centre_shifts_the_projection():
    """
    Cửa sổ lệch tâm thì điểm chiếu dịch ngược lại đúng khoảng ấy (D2).

    Không truyền ``centre_mm`` là tâm ngắm lệch đi đúng khoảng cửa sổ bị dời.
    """
    middle = project_to_design((0, 0, -500), REST_EYE, GLASS, (0.0, 0.0))
    moved = project_to_design((0, 0, -500), REST_EYE, GLASS, (20.0, 0.0))
    assert moved[0] < middle[0]
    assert moved[1] == middle[1]


def test_filter_never_returns_empty_while_bugs_remain():
    """Không con nào lọt ô thì giữ nguyên danh sách — thà khó thấy còn hơn trống."""
    tiny = (0, 0, 1, 1)
    kept, _outside = filter_props(PROPS, REST_EYE, 6, GLASS, tiny, CENTRE)
    assert kept
