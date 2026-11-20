"""
Máy trạng thái màn hình của giao diện album: đang ở màn nào, bấm gì đi đâu.

Giữ đúng lối của ``prop_game.py``: bản ghi phẳng + hàm module, không lớp lồng
lớp. Phần PvP và tài khoản là **mô phỏng** (bot, JSON cục bộ) — kịch bản
TASK_album_ui_1106 cho phép, và báo cáo phải nói rõ.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from src.game.match import LEVEL_TABLE

logger = logging.getLogger(__name__)

LOGIN, SIGNUP, LOGIN_ERR = "moi_01_dang_nhap", "moi_01b_dang_ky", "moi_01c_loi_dang_nhap"
COVER = "moi_02_menu_bia"
PIPE0, PIPE, PIPE100 = ("moi_02b_dang_tach_lop_0", "moi_02b_dang_tach_lop",
                        "moi_02b_dang_tach_lop_100")
SETTINGS, OFFLINE = "moi_02c_cai_dat", "moi_02d_mat_ket_noi"
LEVELS, PHOTOS = "moi_03_chon_cap", "moi_04_chon_anh"
LOADOUT, TILT, PLAY = ("moi_05b_trang_bi_chua_chon", "moi_06a_thu_nghieng",
                       "moi_06_man_choi")
WIN, LOSE = "moi_07a_thang", "moi_07b_thua"
PVP_MODE, PVP_SEARCH = "moi_08_chon_kieu", "moi_08b_tim_tran"
PVP_INVITE, PVP_FOUND = "moi_08c_moi_ban", "moi_08d_da_tim"
PVP_HUNT, PVP_HIDE = "moi_09_man_choi_tho_san", "moi_09b_hud_hoa_bo"
PVP_RESULT, BOARD = "moi_10_ket_qua_pvp", "moi_11_bang_xep_hang"

TABS = {"tab_CHƠI ĐƠN": LEVELS, "tab_PVP": PVP_MODE,
        "tab_XẾP HẠNG": BOARD, "tab_CÀI ĐẶT": SETTINGS}

ITEM_WEIGHT = {"Pháo sáng": 3, "Thuốc giải": 5, "Khiên axit": 4,
               "Kính lọc": 4, "La bàn": 2, "Máy quét nhiệt": 8,
               "Nước tăng lực": 3, "Bom khói": 6}


@dataclass
class AlbumFlow:
    """Trạng thái giao diện. Một bản ghi phẳng, đọc được từ mọi chỗ."""

    screen: str = LOGIN
    theme: str = "ngay"
    player: str = ""
    password: str = ""
    #: Ô "nhập lại mật khẩu" là một chuỗi RIÊNG. Dùng chung một biến thì gõ ô
    #: này ô kia cũng đầy theo, và xoá một ô là xoá cả hai (người dùng gặp
    #: 06/11 23:01).
    password2: str = ""
    focus: str | None = None
    level: int = 1
    weapon: int | None = None
    stickers: list = field(default_factory=list)
    photo = None
    recent: list = field(default_factory=list)
    mode: str = "Cạnh tranh"
    room_code: str = ""
    search_started_s: float = 0.0
    search_target_s: float = 4.0
    found_at_s: float = 0.0
    pipeline_pct: float = 0.0
    paused: bool = False
    #: Lời báo hiện ngay trên giao diện album (toast cũ do renderer vẽ, mà ở
    #: chế độ album renderer chỉ vẽ trong ô viewport nên người dùng KHÔNG
    #: thấy — bấm "Vào trận" tưởng như không có gì xảy ra, 07/11 00:05).
    message: str = ""
    message_until_s: float = 0.0
    webcam_index: int = 0
    #: Tracker có đang thấy mặt không — ô ``head`` trên màn chơi đọc cờ này.
    face_seen: bool = False
    #: MediaPipe không nạp được → tracker chạy pose **mô phỏng**. Khi đó ô
    #: ``head`` không được nói "tốt": con số đang nhìn không phải đầu người.
    tracker_simulated: bool = False
    last_result: dict = field(default_factory=dict)

    def slots(self) -> int:
        return LEVEL_TABLE[self.level][3]

    def level_name(self) -> str:
        return LEVEL_TABLE[self.level][0]

    def go(self, screen: str) -> None:
        if screen != self.screen:
            logger.info("Album: %s -> %s", self.screen, screen)
        self.screen = screen

    def selected_hotspots(self) -> set:
        """Hotspot đang được chọn, để vẽ khung ``primary``."""
        picked = {f"cap_{self.level}_tui"}
        if self.weapon is not None:
            picked.add(f"vu_khi_{self.weapon}_tui")
        picked |= {f"sticker_{name}" for name in self.stickers}
        picked.add("o_Ngày" if self.theme == "ngay" else "o_Đêm")
        picked.add(f"che_do_{self.mode}_tui")
        if self.focus:
            picked.add(self.focus)
        return picked

    def disabled_hotspots(self) -> set:
        """Nút mờ 50 %: chưa chọn vũ khí thì chưa được bắt đầu."""
        if self.screen == LOADOUT and self.weapon is None:
            return {"nut_Bắt đầu"}
        return set()


#: Ô nhập → tên thuộc tính giữ chuỗi của nó.
FIELD_ATTR = {"o_TÊN NGƯỜI CHƠI": "player", "o_MẬT KHẨU": "password",
              "o_NHẬP LẠI MẬT KHẨU": "password2"}


def type_char(flow: AlbumFlow, char: str) -> None:
    """Gõ một ký tự vào **đúng** ô đang chọn (đăng nhập / đăng ký)."""
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        setattr(flow, attr, (getattr(flow, attr) + char)[:24])


#: Thứ tự ô nhập khi bấm Tab.
FIELD_ORDER = ("o_TÊN NGƯỜI CHƠI", "o_MẬT KHẨU", "o_NHẬP LẠI MẬT KHẨU")


def focus_next(flow: AlbumFlow, fields) -> None:
    """Tab: sang ô nhập kế tiếp của màn (vòng lại ô đầu)."""
    order = [name for name in FIELD_ORDER if name in fields]
    if not order:
        return
    if flow.focus in order:
        flow.focus = order[(order.index(flow.focus) + 1) % len(order)]
    else:
        flow.focus = order[0]


def backspace(flow: AlbumFlow) -> None:
    """Xoá lùi một ký tự ở **đúng** ô đang chọn."""
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        setattr(flow, attr, getattr(flow, attr)[:-1])


def update(flow: AlbumFlow, now: float) -> None:
    """Nhịp thời gian của các màn mô phỏng (tìm trận, đếm ngược)."""
    if flow.screen == PVP_SEARCH:
        if now - flow.search_started_s >= flow.search_target_s:
            flow.found_at_s = now
            flow.go(PVP_FOUND)
    elif flow.screen == PVP_FOUND and now - flow.found_at_s >= 3.0:
        flow.go(PVP_HUNT if flow.mode == "Cạnh tranh" else PVP_HIDE)


def say(flow: AlbumFlow, text: str, seconds: float = 4.0) -> None:
    """Hiện một dòng báo trên giao diện album."""
    flow.message = str(text)
    flow.message_until_s = time.perf_counter() + float(seconds)
    logger.info("Album bao: %s", text)


def start_search(flow: AlbumFlow, now: float) -> None:
    flow.search_started_s = now
    flow.search_target_s = random.uniform(3.0, 5.0)
    flow.go(PVP_SEARCH)
