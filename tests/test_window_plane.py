"""
D2 — mặt phẳng Kooima phải là hình chữ nhật vật lý của **cửa sổ**.

Ba trường hợp trên màn 1920×1080, lấy đúng số trong ``assets/config/screen.json``
(344 × 194 mm): toàn màn hình, cửa sổ 1280×720 ở góc trên-trái, và cửa sổ
960×540 ở góc dưới-phải. Cả ba phải khớp công thức của kịch bản tới 1e-9.

Vì sao chấm lại chính công thức chứ không chỉ chấm vài con số viết tay: hai
chỗ dễ sai nhất ở đây là **dấu của trục Y** và **gốc toạ độ của màn hình**, và
cả hai đều cho ra con số trông hợp lý. Trường hợp 2 và 3 nằm ở hai góc chéo
nhau nên một cú lật dấu Y là thấy ngay.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.renderer.camera import VirtualCamera
from src.renderer.window_plane import window_plane

TOL = 1e-9

MONITOR_POS = (0, 0)
MONITOR_SIZE = (1920, 1080)


@pytest.fixture(scope="module")
def calibration() -> dict:
    """Số đo thật của máy, đọc từ ``assets/config/screen.json``."""
    path = Path(__file__).resolve().parents[1] / "assets" / "config" / "screen.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "width_m": float(payload["screen_width"]),
        "height_m": float(payload["screen_height"]),
        "center_m": np.asarray(payload["screen_center"], dtype=np.float64),
    }


def expected(calibration, window_pos, window_size):
    """Công thức của kịch bản, viết lại độc lập với mã sản phẩm."""
    screen_w, screen_h = MONITOR_SIZE
    sx = calibration["width_m"] / screen_w
    sy = calibration["height_m"] / screen_h
    window_w, window_h = window_size
    dx = (window_pos[0] - MONITOR_POS[0]) + window_w / 2 - screen_w / 2
    dy = (window_pos[1] - MONITOR_POS[1]) + window_h / 2 - screen_h / 2
    centre = calibration["center_m"] + np.array([dx * sx, -dy * sy, 0.0])
    return (window_w * sx, window_h * sy, centre)


def measured(calibration, window_pos, window_size):
    return window_plane(calibration["width_m"], calibration["height_m"],
                        calibration["center_m"], MONITOR_POS, MONITOR_SIZE,
                        window_pos, window_size)


CASES = [
    ("fullscreen", (0, 0), (1920, 1080)),
    ("window 1280x720 at (0, 0)", (0, 0), (1280, 720)),
    ("window 960x540 bottom right", (960, 540), (960, 540)),
]


@pytest.mark.parametrize("name,window_pos,window_size", CASES)
def test_matches_formula(calibration, name, window_pos, window_size):
    """Cả ba trường hợp khớp công thức tới 1e-9."""
    got_w, got_h, got_c = measured(calibration, window_pos, window_size)
    want_w, want_h, want_c = expected(calibration, window_pos, window_size)
    assert got_w == pytest.approx(want_w, abs=TOL), name
    assert got_h == pytest.approx(want_h, abs=TOL), name
    assert np.allclose(got_c, want_c, atol=TOL), f"{name}: {got_c} != {want_c}"


def test_fullscreen_equals_calibration(calibration):
    """Toàn màn hình phải cho ra ĐÚNG hiệu chỉnh, không xê dịch gì."""
    width, height, centre = measured(calibration, (0, 0), MONITOR_SIZE)
    assert width == pytest.approx(calibration["width_m"], abs=TOL)
    assert height == pytest.approx(calibration["height_m"], abs=TOL)
    assert np.allclose(centre, calibration["center_m"], atol=TOL)


def test_window_top_left_pushes_plane_left_and_up(calibration):
    """
    Cửa sổ ở góc trên-trái: tâm tấm kính phải lệch sang **trái** và lên **trên**.

    Đây là cái bẫy dấu Y: pixel đếm xuôi xuống, trục Y của HCPD hướng lên.
    """
    _w, _h, centre = measured(calibration, (0, 0), (1280, 720))
    assert centre[0] < calibration["center_m"][0]
    assert centre[1] > calibration["center_m"][1]
    assert centre[2] == pytest.approx(calibration["center_m"][2], abs=TOL)


def test_window_bottom_right_is_mirror_of_top_left(calibration):
    """Hai cửa sổ 960×540 ở hai góc chéo lệch tâm đối xứng qua tâm màn."""
    _w, _h, top_left = measured(calibration, (0, 0), (960, 540))
    _w2, _h2, bottom_right = measured(calibration, (960, 540), (960, 540))
    middle = calibration["center_m"]
    assert np.allclose(top_left - middle, -(bottom_right - middle), atol=TOL)


def test_window_plane_halves_the_glass(calibration):
    """Cửa sổ rộng nửa màn thì tấm kính rộng đúng nửa số đo thật."""
    width, height, _c = measured(calibration, (0, 0), (960, 540))
    assert width == pytest.approx(calibration["width_m"] * 0.5, abs=TOL)
    assert height == pytest.approx(calibration["height_m"] * 0.5, abs=TOL)


def test_set_screen_invalidates_cached_projection(calibration):
    """
    ``set_screen`` phải bỏ phép chiếu đang đệm.

    Không bỏ thì cảnh giữ khối nhìn cũ cho tới lúc đầu người dịch quá
    ``cache_epsilon_mm`` — dời cửa sổ xong ảnh đứng sai một nhịp.
    """
    camera = VirtualCamera(screen_width_m=calibration["width_m"],
                           screen_height_m=calibration["height_m"],
                           screen_center_m=calibration["center_m"].copy(),
                           cache_projection=True)
    _view, before = camera.view_projection_flat()
    before = np.array(before, copy=True)
    width, height, centre = measured(calibration, (0, 0), (1280, 720))
    camera.set_screen(width, height, centre)
    _view2, after = camera.view_projection_flat()
    assert not np.allclose(before, after), "phep chieu khong doi sau set_screen"
