"""
Giao diện v3 chỉ ăn phím của **người chơi**, không ăn đồ nghề sandbox.

Bản đầu khai ``V3_KEYS = ALBUM_KEYS - {T, F4, G}`` và đã sót: ``F6`` (mở
file) với ``F9`` (bật/tắt lấp khe) vẫn lọt vì chúng nằm sẵn trong
``ALBUM_KEYS``, người dùng gặp lại chúng giữa lúc chơi. Phép trừ dần như vậy
còn một bẫy nữa: thêm phím vào bộ asset cũ là tự động mở cửa cho v3.

Nên file này chấm theo hai chiều — những phím **phải có** và những phím
**không được có** — và chấm luôn rằng bộ cũ không bị động tới.
"""

from __future__ import annotations

import glfw
import pytest

from src.renderer.window import ALBUM_KEYS, DEMO_KEYS, V3_KEYS

#: Đồ nghề sandbox của lõi HCPD. Không cái nào được có mặt trong v3.
SANDBOX = {
    "F1 HUD go loi": glfw.KEY_F1,
    "F2 doi bo loc": glfw.KEY_F2,
    "F3 du doan": glfw.KEY_F3,
    "F4 webcam PiP": glfw.KEY_F4,
    "F5": glfw.KEY_F5,
    "F6 mo file": glfw.KEY_F6,
    "F7 suong mu": glfw.KEY_F7,
    "F8 khi quyen": glfw.KEY_F8,
    "F9 lap khe": glfw.KEY_F9,
    "F10 noi khoi": glfw.KEY_F10,
    "G do mat nghi": glfw.KEY_G,
    "T tat tracking": glfw.KEY_T,
}

#: Phím nhích A/B của lõi (zoom, plane Z, far layer Z, overscan).
NUDGE = {
    "+": glfw.KEY_EQUAL, "-": glfw.KEY_MINUS,
    "[": glfw.KEY_LEFT_BRACKET, "]": glfw.KEY_RIGHT_BRACKET,
    ",": glfw.KEY_COMMA, ".": glfw.KEY_PERIOD,
    ";": glfw.KEY_SEMICOLON, "'": glfw.KEY_APOSTROPHE,
}

#: Phím v3 PHẢI nhận, kèm việc của nó.
WANTED = {
    "F11 doi che do hien thi": glfw.KEY_F11,
    "F12 chup man": glfw.KEY_F12,
    "Esc lui mot buoc": glfw.KEY_ESCAPE,
    "Enter nut chinh": glfw.KEY_ENTER,
    "Tab sang o nhap": glfw.KEY_TAB,
    "Backspace xoa lui": glfw.KEY_BACKSPACE,
    "C can lai cho ngoi": glfw.KEY_C,
    "Up vong focus": glfw.KEY_UP,
    "Down vong focus": glfw.KEY_DOWN,
    "R choi lai": glfw.KEY_R,
    "1 o vat pham": glfw.KEY_1,
    "U chia se anh": glfw.KEY_U,
}


@pytest.mark.parametrize("name,key", sorted(SANDBOX.items()))
def test_sandbox_keys_are_out_of_v3(name, key):
    assert key not in V3_KEYS, f"{name} van con trong luong v3"


@pytest.mark.parametrize("name,key", sorted(NUDGE.items()))
def test_nudge_keys_are_out_of_v3(name, key):
    assert key not in V3_KEYS, f"phim nhich {name} van con trong luong v3"


@pytest.mark.parametrize("name,key", sorted(WANTED.items()))
def test_player_keys_are_in_v3(name, key):
    assert key in V3_KEYS, f"thieu {name}"


def test_v3_is_not_derived_from_the_old_set():
    """
    Thêm phím vào bộ cũ KHÔNG được tự động mở cửa cho v3.

    Chấm bằng cách tìm một phím chỉ bộ cũ mới có: nếu ``V3_KEYS`` là tập cha
    của ``ALBUM_KEYS`` thì nó đang đi theo bộ cũ, đúng cái bẫy đã sót F6/F9.
    """
    assert not ALBUM_KEYS <= V3_KEYS
    assert DEMO_KEYS & V3_KEYS == frozenset()


def test_old_album_pack_keeps_its_tools():
    """Bộ asset cũ là đường lùi cho buổi bảo vệ — không được đụng vào."""
    for key in (glfw.KEY_G, glfw.KEY_T, glfw.KEY_F4, glfw.KEY_F6):
        assert key in ALBUM_KEYS


def test_v3_set_is_small_enough_to_read():
    """Danh sách cho phép phải ngắn; dài ra là dấu hiệu lại trượt về trừ dần."""
    assert len(V3_KEYS) <= 16
