"""
Bấm vào hotspot nào thì đi đâu — bảng định tuyến của giao diện album.

Tách khỏi ``flow.py`` để mỗi file một việc: ở đó là *trạng thái*, ở đây là
*phản ứng*. Hàm ``on_click`` trả về tên hành động mà app phải tự làm (mở hộp
chọn file, chạy tách lớp, đo mắt nghỉ…), hoặc ``""`` nếu chỉ đổi màn.
"""

from __future__ import annotations

import logging
import random

from src.ui_album import flow as F

logger = logging.getLogger(__name__)

#: Hành động app phải thực hiện (giao diện không tự làm được).
ACT_PICK_FILE = "pick_file"
ACT_START_PIPELINE = "start_pipeline"
ACT_REST_EYE = "rest_eye"
ACT_START_GAME = "start_game"
ACT_NEXT_LEVEL = "next_level"
ACT_QUIT = "quit"


def _login_actions(state, name: str) -> str:
    if name == "nut_Mở album":
        # Mật khẩu **phải có**; tài khoản vẫn là mô phỏng (chưa lưu ra file,
        # chưa băm) nên không kiểm nội dung, chỉ kiểm có nhập hay không.
        ok = bool(state.player.strip()) and bool(state.password)
        state.go(F.COVER if ok else F.LOGIN_ERR)
    elif name == "nut_Tạo album":
        # Tạo album: phải có tên, có mật khẩu, và hai ô mật khẩu khớp nhau.
        ok = (bool(state.player.strip()) and bool(state.password)
              and state.password == state.password2)
        state.go(F.COVER if ok else F.SIGNUP)
    elif name.startswith("o_"):
        state.focus = name
    elif name.startswith("text:Chưa có album"):
        state.focus = None
        state.go(F.SIGNUP)
    return ""


def _single_player(state, name: str) -> str:
    if name.startswith("cap_") and name.endswith("_tui"):
        state.level = int(name.split("_")[1])
    elif name == "nut_Tiếp: chọn ảnh":
        state.go(F.PHOTOS)
    elif name in ("nut_Chọn từ máy", "tui_trong_tui"):
        return ACT_PICK_FILE
    elif name.startswith("ky_uc_"):
        index = int(name.split("_")[2]) - 1
        if index < len(state.recent):
            state.photo = state.recent[index]
    elif name == "nut_Tách lớp & tiếp tục":
        return ACT_START_PIPELINE
    elif name == "nut_Tiếp: chọn trang bị":
        state.go(F.LOADOUT)
    elif name.startswith("vu_khi_"):
        state.weapon = int(name.split("_")[2])
    elif name.startswith("sticker_"):
        item = name[len("sticker_"):]
        if item in state.stickers:
            state.stickers.remove(item)
        elif len(state.stickers) < state.slots():
            state.stickers.append(item)
    elif name == "nut_Bắt đầu" and state.weapon is not None:
        state.go(F.TILT)
    elif name == "nut_Căn lại (C)":
        return ACT_REST_EYE
    elif name == "nut_Vào trận":
        return ACT_START_GAME
    return ""


def _results(state, name: str) -> str:
    if name == "nut_Cấp tiếp theo":
        return ACT_NEXT_LEVEL
    if name in ("nut_Chơi lại", "nut_Thử lại"):
        state.go(F.TILT)
    elif name == "nut_Chọn ảnh khác":
        state.go(F.PHOTOS)
    elif name == "nut_Về album":
        state.go(F.COVER)
    return ""


def _pvp(state, name: str, now: float) -> str:
    if name.startswith("che_do_"):
        state.mode = name[len("che_do_"):-len("_tui")]
    elif name == "nut_Tìm trận":
        F.start_search(state, now)
    elif name == "nut_Mời bạn":
        state.room_code = f"{random.randint(0, 999999):06d}"
        state.go(F.PVP_INVITE)
    elif name == "nut_Vào phòng":
        state.found_at_s = now
        state.go(F.PVP_FOUND)
    elif name in ("nut_Huỷ", "nut_Huỷ phòng"):
        state.go(F.PVP_MODE)
    elif name == "nut_Đấu lại":
        F.start_search(state, now)
    elif name == "nut_Bảng xếp hạng":
        state.go(F.BOARD)
    elif name == "nut_Thử lại ngay":
        state.go(F.PVP_MODE)
    return ""


def on_click(state, name: str | None, now: float) -> str:
    """Xử lý một cú bấm. Trả tên hành động cho app, hoặc chuỗi rỗng."""
    if not name:
        return ""
    logger.info("Album click: %s (man %s)", name, state.screen)
    if name in F.TABS and state.screen not in (F.LOGIN, F.SIGNUP, F.LOGIN_ERR):
        state.go(F.TABS[name])
        return ""
    if name == "text:Đăng xuất":
        state.player, state.password, state.password2 = "", "", ""
        state.focus = None
        state.go(F.LOGIN)
        return ""
    if name == "nut_Quay lại":
        state.go(F.COVER if state.screen in (F.LEVELS, F.PVP_MODE) else F.LEVELS)
        return ""
    if name == "o_Ngày":
        state.theme = "ngay"
        return ""
    if name == "o_Đêm":
        state.theme = "dem"
        return ""
    if name == "chon_webcam":
        state.webcam_index = (state.webcam_index + 1) % 2
        state.message = "chưa hỗ trợ đổi webcam khi đang chạy"
        return ""
    if name.startswith("tab_con_"):
        state.mode = name[len("tab_con_"):]
        return ""
    if state.screen in (F.LOGIN, F.SIGNUP, F.LOGIN_ERR):
        return _login_actions(state, name)
    if state.screen in (F.WIN, F.LOSE, F.PVP_RESULT, F.OFFLINE):
        return _results(state, name)
    if state.screen in (F.PVP_MODE, F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND,
                        F.PVP_HUNT, F.PVP_HIDE):
        return _pvp(state, name, now)
    return _single_player(state, name)
