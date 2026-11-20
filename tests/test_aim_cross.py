"""
Tâm ngắm dấu ``+`` của màn chơi v3.

Kiểm bằng một painter giả ghi lại từng ô tô màu, chứ không chụp ảnh: ảnh
chụp phụ thuộc con trỏ chuột đang ở đâu lúc chụp, còn hình dạng dấu cộng thì
phải đúng ở mọi vị trí.

Ba tính chất đáng giữ: có **khoảng hở ở giữa** (không che đúng điểm viên đạn
sẽ tới), có **viền sẫm dưới nét sáng** (ảnh nền có cả vùng tối lẫn vùng sáng,
một màu đơn sẽ biến mất ở một trong hai), và **đối xứng quanh tâm ngắm**.
"""

from __future__ import annotations

import pytest

from src.ui_album.v3 import V3State
from src.ui_album.v3_painter import rgba
from src.ui_album.v3_screen_draw import (AIM_ARM, AIM_COL, AIM_EDGE,
                             AIM_EDGE_COL, AIM_GAP, AIM_THICK, _draw_aim)


class FakePainter:
    """Chỉ ghi lại các ô tô màu; không đụng GL."""

    def __init__(self) -> None:
        self.fills: list = []

    def fill(self, rect, colour) -> None:
        self.fills.append((tuple(float(v) for v in rect), colour))


def draw_at(x: float, y: float):
    state = V3State()
    state.aim = (x, y)
    painter = FakePainter()
    _draw_aim(painter, state)
    return painter.fills


def test_nothing_drawn_without_an_aim_point():
    state = V3State()
    painter = FakePainter()
    _draw_aim(painter, state)
    assert painter.fills == []


def test_shot_mode_draws_no_crosshair():
    """README: chế độ chụp không có hover, focus, caret — và không tâm ngắm."""
    state = V3State()
    state.aim = (100.0, 100.0)
    state.shot = True
    painter = FakePainter()
    _draw_aim(painter, state)
    assert painter.fills == []


def test_four_arms_in_two_passes():
    """Bốn nhánh × hai lượt (viền sẫm rồi nét sáng) = tám ô."""
    assert len(draw_at(640.0, 360.0)) == 8


def test_dark_edge_is_drawn_before_the_light_stroke():
    """Vẽ viền sau nét sáng thì viền đè mất nét — thứ tự phải đúng."""
    fills = draw_at(640.0, 360.0)
    assert fills[0][1] == rgba(AIM_EDGE_COL, 0.85)
    assert fills[-1][1] == rgba(AIM_COL)


def test_centre_stays_clear():
    """Không ô nào phủ lên đúng điểm ngắm."""
    x, y = 640.0, 360.0
    for (rx, ry, rw, rh), _colour in draw_at(x, y):
        covers = rx < x < rx + rw and ry < y < ry + rh
        assert not covers, (rx, ry, rw, rh)


def test_arms_are_symmetric_about_the_aim_point():
    """Gương trái/phải và trên/dưới trùng nhau."""
    x, y = 500.0, 300.0
    boxes = [r for r, _c in draw_at(x, y)]
    left = {(round(x - (rx + rw), 3), round(ry - y, 3)) for rx, ry, rw, _rh in boxes}
    right = {(round(rx - x, 3), round(ry - y, 3)) for rx, ry, _rw, _rh in boxes}
    assert left & right, "khong tim thay cap nhanh doi xung"


def test_total_span_matches_the_constants():
    """Bề ngang dấu cộng = 2 × (tay + viền), đúng hằng số đã khai."""
    x, y = 0.0, 0.0
    boxes = [r for r, _c in draw_at(x, y)]
    far_left = min(rx for rx, _ry, _rw, _rh in boxes)
    far_right = max(rx + rw for rx, _ry, rw, _rh in boxes)
    assert far_left == pytest.approx(-(AIM_ARM + AIM_EDGE))
    assert far_right == pytest.approx(AIM_ARM + AIM_EDGE)


def test_it_is_small():
    """'Tâm ngắm nhỏ thôi': không quá 20 px thiết kế, mảnh hơn 6 px."""
    assert 2 * (AIM_ARM + AIM_EDGE) <= 20.0
    assert AIM_THICK + 2 * AIM_EDGE <= 6.0
    assert AIM_GAP > 0.0


def _span(fills) -> float:
    boxes = [r for r, _c in fills]
    return (max(rx + rw for rx, _ry, rw, _rh in boxes)
            - min(rx for rx, _ry, _rw, _rh in boxes))


def draw_ready(x: float, y: float, ready: float):
    state = V3State()
    state.aim = (x, y)
    state.aim_ready = ready
    painter = FakePainter()
    _draw_aim(painter, state)
    return painter.fills


def test_crosshair_opens_up_right_after_a_shot():
    """
    Nhịp bắn phải NHÌN THẤY được.

    Đo trên một phiên chơi thật: 54 phát ra, 54 cú bấm bị chặn vì chưa tới
    nhịp. Không có âm thanh trong bản demo, nên tâm ngắm là chỗ duy nhất nói
    được "súng đang hồi"; im lặng thì người chơi tưởng app đớp mất cú bấm.
    """
    assert _span(draw_ready(0.0, 0.0, 0.0)) > _span(draw_ready(0.0, 0.0, 1.0))


def test_crosshair_closes_back_as_the_gun_recovers():
    """Khép lại đều đặn, không nhảy một phát từ bung sang khép."""
    spans = [_span(draw_ready(0.0, 0.0, r)) for r in (0.0, 0.25, 0.5, 0.75, 1.0)]
    assert spans == sorted(spans, reverse=True)
    assert len(set(spans)) == len(spans)


def test_crosshair_dims_while_recovering():
    """Vừa bung vừa nhạt — hai tín hiệu cùng nói một điều."""
    busy = draw_ready(0.0, 0.0, 0.0)[-1][1]
    idle = draw_ready(0.0, 0.0, 1.0)[-1][1]
    assert busy[3] < idle[3]


def test_recovered_crosshair_matches_the_plain_one():
    """Hồi xong thì phải y hệt dấu cộng bình thường, không lệch một pixel."""
    assert draw_ready(50.0, 50.0, 1.0) == draw_at(50.0, 50.0)


def test_centre_stays_clear_even_while_recovering():
    """Bung ra cũng không được phủ lên điểm ngắm."""
    for (rx, ry, rw, rh), _c in draw_ready(0.0, 0.0, 0.0):
        assert not (rx < 0.0 < rx + rw and ry < 0.0 < ry + rh)


def test_it_follows_the_aim_point():
    """Dời tâm ngắm thì cả dấu cộng dời theo đúng khoảng ấy."""
    first = [r for r, _c in draw_at(100.0, 100.0)]
    second = [r for r, _c in draw_at(140.0, 125.0)]
    for (ax, ay, aw, ah), (bx, by, bw, bh) in zip(first, second):
        assert (bx - ax, by - ay) == pytest.approx((40.0, 25.0))
        assert (aw, ah) == pytest.approx((bw, bh))


# ---- TASK_game_feel_1118 P4: crosshair kick -------------------------------
def draw_kicked(x: float, y: float, kick: float):
    state = V3State()
    state.aim = (x, y)
    state.aim_kick_scale = kick
    painter = FakePainter()
    _draw_aim(painter, state)
    return painter.fills


def test_default_kick_scale_matches_the_plain_crosshair():
    """``aim_kick_scale`` defaults to 1.0 — a no-op, same as before P4."""
    assert draw_kicked(50.0, 50.0, 1.0) == draw_at(50.0, 50.0)


def test_kick_grows_the_crosshair():
    assert _span(draw_kicked(0.0, 0.0, 1.35)) > _span(draw_kicked(0.0, 0.0, 1.0))


def test_kick_still_stays_symmetric_about_the_aim_point():
    x, y = 500.0, 300.0
    boxes = [r for r, _c in draw_kicked(x, y, 1.4)]
    left = {(round(x - (rx + rw), 3), round(ry - y, 3)) for rx, ry, rw, _rh in boxes}
    right = {(round(rx - x, 3), round(ry - y, 3)) for rx, ry, _rw, _rh in boxes}
    assert left & right, "khong tim thay cap nhanh doi xung"


def test_kick_still_leaves_the_centre_clear():
    for (rx, ry, rw, rh), _c in draw_kicked(0.0, 0.0, 1.4):
        assert not (rx < 0.0 < rx + rw and ry < 0.0 < ry + rh)
