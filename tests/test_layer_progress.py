"""
Ô ``LAYER a / N`` của HUD v3: ``N`` là số lớp THẬT, ``a`` là lớp nông nhất
còn giấu bọ chưa tìm.

Vì sao có file này: bản đầu gọi ``plane.z_world_m`` — đó là tên trường bên
``_LayerRenderable`` của renderer, còn ``LayerPlane`` đặt tên ``z_m``. Lỗi
không lộ ra ở bất kỳ test nào đang có, và nó giết **cả vòng vẽ** đúng lúc
người chơi vào trận (phiên 07/11). Nên test ở đây dựng ``LayerPlane`` thật,
không dùng stub: cái cần bắt chính là sai tên trường.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.content.hidden_props import LayerPlane
from src.engine.app import HcpdApplication


def plane(z_m: float) -> LayerPlane:
    """Một lớp tối giản; phép đo chỉ đụng tới ``z_m``."""
    return LayerPlane(alpha=np.zeros((2, 2), dtype=np.uint8), z_m=z_m,
                      quad_w_m=0.3, quad_h_m=0.2, centre_x_m=0.0,
                      centre_y_m=0.0, z_min_m=z_m, z_max_m=z_m)


def app_with(planes, props, found) -> HcpdApplication:
    """App dựng bằng ``__new__`` — chỉ gắn đúng thứ ``_layer_progress`` đọc."""
    app = HcpdApplication.__new__(HcpdApplication)
    app._game_planes = tuple(planes)

    class _Game:
        pass

    game = _Game()
    game.props = [{"pos_mm": (0.0, 0.0, z)} for z in props]
    game.found = list(found)
    app._game = game if props else None
    return app


#: Ba lớp, xa → gần (``layer_planes`` trả theo thứ tự này).
PLANES = [plane(-0.30), plane(-0.20), plane(-0.10)]


def test_counts_the_real_layers_not_four():
    """``N`` theo số lớp thật của ảnh, không phải hằng 4 của bản vẽ."""
    app = app_with(PLANES, [-100.0], [False])
    assert app._layer_progress()[1] == 3


def test_shallowest_unfound_layer_is_reported():
    """Bọ ở lớp gần nhất (z −100 mm) → đang ở lớp 1."""
    app = app_with(PLANES, [-100.0, -300.0], [False, False])
    assert app._layer_progress()[0] == 1


def test_clearing_the_near_layer_pushes_the_number_deeper():
    """Diệt hết bọ lớp nông thì con số tụt sâu — dãy pip đầy theo tiến độ."""
    app = app_with(PLANES, [-100.0, -300.0], [True, False])
    assert app._layer_progress() == (3, 3)


def test_no_planes_reports_zero_total():
    """Chưa tách lớp xong thì tổng là 0, để phía trên còn rơi về ước lượng."""
    app = app_with([], [], [])
    assert app._layer_progress() == (1, 0)


def test_uses_the_field_name_layer_plane_actually_has():
    """
    Chạm đúng chỗ đã hỏng: ``LayerPlane`` có ``z_m``, KHÔNG có ``z_world_m``.

    Khẳng định thẳng trên dataclass, nên nếu ai đổi tên trường thì test này
    nói rõ lý do chứ không chỉ ném AttributeError ở chỗ khác.
    """
    one = plane(-0.2)
    assert hasattr(one, "z_m")
    assert not hasattr(one, "z_world_m")


def test_all_bugs_found_still_returns_a_valid_layer():
    """Tìm hết bọ: không được trả 0 hay số âm cho ô 'LAYER a / N'."""
    app = app_with(PLANES, [-100.0, -300.0], [True, True])
    current, total = app._layer_progress()
    assert total == 3
    assert 1 <= current <= total


@pytest.mark.parametrize("z_mm,expected", [(-100.0, 1), (-200.0, 2), (-300.0, 3)])
def test_depth_maps_to_the_nearest_plane(z_mm, expected):
    """Mỗi con bọ quy về lớp có độ sâu gần nó nhất; 1 = lớp gần người chơi."""
    app = app_with(PLANES, [z_mm], [False])
    assert app._layer_progress()[0] == expected
