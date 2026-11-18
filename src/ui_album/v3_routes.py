"""
Bấm vào hotspot nào thì đi đâu — bảng định tuyến của giao diện v3.

Tách khỏi ``v3_flow.py`` để mỗi file một việc: ở đó là *trạng thái*, ở đây là
*phản ứng*. ``on_click`` trả về tên hành động mà app phải tự làm (mở hộp chọn
file, chạy tách lớp, đo mắt nghỉ…), hoặc ``""`` nếu chỉ đổi màn — y hợp đồng
của ``routes.py``, nên ``app.py`` dùng lại nguyên bộ hành động cũ.
"""

from __future__ import annotations

import logging

from src.account import crypto as _account_crypto
from src.content import shared_album
from src.ui_album import v3_flow as F

logger = logging.getLogger(__name__)

#: Hành động app phải thực hiện (giao diện không tự làm được). Tên giữ y
#: ``routes.py`` để ``_album_action`` phục vụ được cả hai bộ giao diện.
ACT_PICK_FILE = "pick_file"
ACT_START_PIPELINE = "start_pipeline"
ACT_REST_EYE = "rest_eye"
ACT_START_GAME = "start_game"
ACT_NEXT_LEVEL = "next_level"
ACT_LOGOUT = "logout"
ACT_COPY_CODE = "copy_code"
ACT_DISPLAY_FULL = "display_full"
ACT_DISPLAY_WINDOW = "display_window"
ACT_SHARE_CONFIRMED = "share_confirmed"
ACT_RANDOM_FROM_ALBUM = "random_from_album"
#: TASK_pvp_lan_1116: PvP thật — app.py làm việc mạng, giao diện chỉ hỏi.
ACT_PVP_FIND = "pvp_find"
ACT_PVP_CREATE_ROOM = "pvp_create_room"
ACT_PVP_JOIN_ROOM = "pvp_join_room"
ACT_PVP_CANCEL = "pvp_cancel"
ACT_PVP_REMATCH = "pvp_rematch"
ACT_PVP_RECONNECT = "pvp_reconnect"
#: TASK_accounts_1119: băm/kiểm mật khẩu CHẠY TRÊN LUỒNG RIÊNG (app.py) —
#: giao diện chỉ hỏi, giống hệt các hành động mạng PvP ở trên.
ACT_SIGN_IN = "account_sign_in"
ACT_SIGN_UP = "account_sign_up"
ACT_FORGOT_OPEN = "account_forgot_open"
ACT_FORGOT_SUBMIT = "account_forgot_submit"
ACT_FORGOT_CLOSE = "account_forgot_close"
ACT_RECOVERY_SAVED = "account_recovery_saved"
ACT_COPY_RECOVERY_CODE = "account_copy_recovery_code"

#: Nút chính của từng màn — Enter bấm đúng nút này (U2 "Enter = the primary
#: button"). Màn không có tên trong bảng thì Enter không làm gì.
PRIMARY = {
    F.SIGN_IN: "btn_Sign in", F.SIGN_UP: "btn_Create album",
    F.SIGN_IN_ERR: "btn_Sign in", F.MENU: None,
    F.LEVELS: "btn_Next: choose a photo",
    F.PHOTOS: "btn_Split layers & continue",
    F.LOADING_READY: "continue", F.LOADOUT: "btn_Start",
    F.TILT: "btn_Enter match", F.WIN: "btn_Next level",
    F.LOSE: "btn_Try again", F.PVP_MODE: "btn_Find match",
    F.PVP_INVITE: "btn_Join room", F.PVP_RESULT: "btn_Rematch",
    F.OFFLINE: "btn_Try again now", F.SETTINGS: "btn_recalibrate",
}

#: Esc / ``btn_Back`` lùi về đâu. Màn vắng mặt thì lùi về màn chính.
BACK = {
    F.LEVELS: F.MENU, F.PHOTOS: F.LEVELS, F.LOADOUT: F.PHOTOS,
    F.TILT: F.LOADOUT, F.SETTINGS: F.MENU, F.BOARD: F.MENU,
    F.PVP_MODE: F.MENU, F.PVP_SEARCH: F.PVP_MODE, F.PVP_INVITE: F.PVP_MODE,
    F.PVP_FOUND: F.PVP_MODE, F.PVP_HUNT: F.PVP_MODE, F.PVP_HIDE: F.PVP_MODE,
    F.PVP_RESULT: F.MENU, F.WIN: F.MENU, F.LOSE: F.MENU,
    F.OFFLINE: F.PVP_MODE, F.LOADING: F.PHOTOS, F.LOADING_READY: F.PHOTOS,
    F.SIGN_UP: F.SIGN_IN, F.SIGN_IN_ERR: F.SIGN_IN,
}


def primary_button(manifest, state) -> str:
    """Tên nút chính của màn đang mở, nếu màn thật sự có hotspot đó."""
    name = PRIMARY.get(state.screen)
    if not name:
        return ""
    names = {h["n"] for h in
             manifest.screens[state.screen].get("hotspots", ())}
    return name if name in names else ""


def on_back(state, now: float) -> str:
    """
    Esc = lùi một bước.

    Ở màn gốc (đăng nhập, màn chính) không còn chỗ nào để lùi, nên Esc giữ
    nghĩa cũ là **thoát app** — bỏ hẳn đường thoát bằng bàn phím thì người
    trình bày bị kẹt trong cửa sổ toàn màn hình.
    """
    _ = now
    if state.screen in (F.SIGN_IN, F.MENU):
        return "quit"
    state.go(BACK.get(state.screen, F.MENU))
    return ""


def _sign_in(state, name: str) -> str:
    """
    TASK_accounts_1119: thật — không còn mô phỏng.

    Băm/kiểm mật khẩu CHẠY NGOÀI LUỒNG VẼ (``app.py``, tương tự mọi hành
    động mạng PvP) — hàm này chỉ kiểm được những gì KHÔNG cần đọc kho tài
    khoản (hình dạng tên, độ dài mật khẩu, mật khẩu khớp), và trả về một
    ``ACT_*`` để app.py tự quyết phần còn lại off-thread, đúng khuôn mọi
    hành động khác trong file này.
    """
    if name == "btn_Sign in":
        state.focus = None
        if not state.player.strip() or not state.password:
            state.sign_in_error = "Enter a username and password."
            state.go(F.SIGN_IN_ERR)
            return ""
        return ACT_SIGN_IN
    elif name == "btn_Create album" and state.screen == F.SIGN_UP:
        state.focus = None
        # Chạm hết ba ô: bấm thẳng nút mà chưa gõ ô nào (vd. chỉ gõ mật
        # khẩu rồi Tab qua nút) vẫn phải thấy đúng gợi ý, không im lặng.
        state.touched |= {"field_user", "field_pass", "field_pass2"}
        display, reason = _account_crypto.validate_username(state.player)
        if display is None:
            return ""                          # gợi ý đã hiện qua state_for
        pass_ok, _reason = _account_crypto.validate_password(state.password)
        if not pass_ok or state.password != state.password2:
            return ""
        return ACT_SIGN_UP
    elif name in ("btn_Create album", "btn_Create a new album"):
        state.focus = None
        state.go(F.SIGN_UP)                    # trên màn đăng nhập = sang đăng ký
    elif name == "footer_link":
        state.focus = None
        # Màn đăng ký: về đăng nhập. Màn lỗi: vào bằng khách.
        if state.screen == F.SIGN_UP:
            state.go(F.SIGN_IN)
        else:
            state.player = state.player or "guest"
            state.account_key = _account_crypto.GUEST_KEY
            # TASK_accounts_1119 Phase 5: khách không có tiến trình riêng
            # — nếu còn history/recent của tài khoản TRƯỚC đó (đăng xuất
            # rồi vào khách) thì phải dọn, không thì màn chính/bảng xếp
            # hạng hiện nhầm dữ liệu của người khác.
            state.history, state.recent = {}, []
            state.go(F.MENU)
    elif name == "link_guest":
        state.focus = None
        state.player = state.player or "guest"
        state.account_key = _account_crypto.GUEST_KEY
        state.history, state.recent = {}, []
        state.go(F.MENU)
    elif name == "link_forgot":
        # TASK_accounts_1119 Phase 6 làm hộp thoại thật; ở đây giữ nguyên
        # thông báo cũ cho tới lúc đó.
        F.say(state, "Password reset isn't available yet.")
    elif name == "switch_remember":
        state.remember = not state.remember
    elif name == "icon_show_password":
        state.show_password = not state.show_password
    elif name.startswith("field_"):
        state.focus = name
    return ""
    return ""


def _settings(state, name: str, point) -> str:
    if name == "theme_day":
        state.theme = "day"
    elif name == "theme_night":
        # Bộ v3 chỉ có một bảng màu. Nói thẳng, và KHÔNG đổi cái điều khiển:
        # để nó nhảy sang Night rồi màn vẫn sáng là hứa điều không có.
        F.say(state, "Night theme is not in v2 yet.")
    elif name == "display_full":
        return ACT_DISPLAY_FULL
    elif name == "display_window":
        return ACT_DISPLAY_WINDOW
    elif name == "dropdown_webcam":
        state.webcam_index = (state.webcam_index + 1) % 2
        F.say(state, "Switching webcam isn't supported while running.")
    elif name == "btn_recalibrate":
        return ACT_REST_EYE
    elif name == "slider":
        state.volume = _slider_value(state, point)
    elif name == "btn_Sign out":
        return ACT_LOGOUT
    return ""


#: Hotspot ``slider`` của màn cài đặt, trong không gian thiết kế. Rãnh âm
#: lượng (``bars[].rect``) hẹp hơn hotspot, nên phải quy chiếu theo rãnh.
SLIDER_TRACK = (520.0, 170.0)


def _slider_value(state, point) -> int:
    """Bấm/kéo trên thanh âm lượng → 0–100, cắt về trong khoảng."""
    if point is None:
        return state.volume
    start, width = SLIDER_TRACK
    ratio = (float(point[0]) - start) / max(1.0, width)
    return int(round(max(0.0, min(1.0, ratio)) * 100.0))


def _confirm_share(state, name: str) -> str:
    """Hộp xác nhận chia sẻ (TASK_shared_album_1111 P2.3): Yes / No."""
    path = state.shared_album_pending
    state.shared_album_pending = ""
    if name == "btn_confirm_no":
        return ""
    state.shared_album_confirmed.add(path)
    state.shared_album_upload_path = path
    return ACT_SHARE_CONFIRMED


def _solo(state, name: str) -> str:
    if name == "btn_Share to album":
        if not shared_album.can_upload():
            pass                     # nhãn nút đã nói "not configured"
        elif not state.photo:
            F.say(state, "Choose a photo first.")
        elif state.photo in state.shared_album_confirmed:
            state.shared_album_upload_path = state.photo
            return ACT_SHARE_CONFIRMED  # đã hỏi ảnh này rồi, khỏi hỏi lại
        else:
            state.shared_album_pending = state.photo
    elif name == "btn_Random from album":
        if shared_album.load_config() is not None:
            return ACT_RANDOM_FROM_ALBUM  # nhan da noi "not configured" neu khong
    elif name.startswith("level_row_"):
        state.level = int(name.rsplit("_", 1)[1])
    elif name == "btn_Next: choose a photo":
        state.go(F.PHOTOS)
    elif name in ("btn_Choose from computer", "dropzone"):
        return ACT_PICK_FILE
    elif name == "recent_memory":
        if state.recent:
            state.photo = state.recent[0]
    elif name == "btn_Split layers & continue":
        return ACT_START_PIPELINE
    elif name == "continue":
        state.go(F.LOADOUT)
    elif name.startswith("weapon_"):
        chosen = name[len("weapon_"):]
        if chosen in F.WEAPONS:
            state.weapon = F.WEAPONS.index(chosen)
    elif name.startswith("sticker_"):
        F.toggle_item(state, name[len("sticker_"):])
    elif name.startswith("slot_empty_"):
        # Bấm vào ô ĐANG CÓ vật phẩm = bỏ nó ra (U2 — Loadout). Ô trống thì
        # không có gì để bỏ, nên cú bấm rơi vào im lặng là đúng.
        F.remove_slot(state, int(name.rsplit("_", 1)[1]) - 1)
    elif name == "btn_Start":
        if state.weapon is None:
            F.say(state, "Choose a weapon first.")
        else:
            state.go(F.TILT)
    elif name in ("btn_Recalibrate", "btn_recalibrate"):
        return ACT_REST_EYE
    elif name == "btn_Enter match":
        if not state.tilt_ready():
            F.say(state, "Finish the three checks first.")
            return ""
        return ACT_START_GAME
    return ""


def _results(state, name: str) -> str:
    if name == "btn_Next level":
        return ACT_NEXT_LEVEL
    if name in ("btn_Play again", "btn_Try again"):
        state.go(F.TILT)
    elif name == "btn_Choose another photo":
        state.go(F.PHOTOS)
    elif name == "btn_Back to album":
        state.go(F.MENU)
    return ""


def _pvp(state, name: str, now: float) -> str:
    """
    TASK_pvp_lan_1116: chỉ ĐỔI MÀN và hỏi; việc mạng app.py làm (trả về một
    ``ACT_PVP_*``). Màn ``PVP_FOUND`` chỉ tới khi server báo ``matched``
    thật — xem ``app.py`` (hàm xử lý thông điệp mạng gọi ``state.go(F.PVP_FOUND)``
    trực tiếp), không còn hẹn giờ giả ở đây.
    """
    if name.startswith("mode_"):
        chosen = name[len("mode_"):]
        if chosen in F.PVP_MODES:
            state.mode = chosen
    elif name == "btn_Find match":
        F.start_search(state, now)          # chỉ để "search_time" có mốc
        return ACT_PVP_FIND
    elif name == "btn_Invite a friend":
        state.room_code = ""                 # trống tới khi server trả "room"
        state.join_typed = ""
        state.go(F.PVP_INVITE)
        return ACT_PVP_CREATE_ROOM
    elif name == "btn_Copy code":
        return ACT_COPY_CODE
    elif name == "field_join":
        state.focus = "field_join"
    elif name == "btn_Join room":
        if len(state.join_typed) != 4:
            F.say(state, "Enter the 4-digit room code first.")
            return ""
        return ACT_PVP_JOIN_ROOM
    elif name == "btn_Close room":
        state.go(F.PVP_MODE)
        return ACT_PVP_CANCEL
    elif name in ("btn_Cancel", "leave"):
        state.go(F.PVP_MODE)
        return ACT_PVP_CANCEL
    elif name == "btn_Rematch":
        return ACT_PVP_REMATCH
    elif name == "btn_Leaderboard":
        state.go(F.BOARD)
    elif name == "btn_Try again now":
        state.go(F.PVP_MODE)
        return ACT_PVP_RECONNECT
    elif name == "btn_Back to album":
        state.go(F.MENU)
    return ""


def on_click(state, name: str | None, now: float, point=None) -> str:
    """Xử lý một cú bấm. Trả tên hành động cho app, hoặc chuỗi rỗng."""
    if not name:
        return ""
    logger.info("UI v3 click: %s (man %s)", name, state.screen)
    if state.shared_album_pending and name in ("btn_confirm_yes", "btn_confirm_no"):
        return _confirm_share(state, name)
    if state.screen in F.LOCKED:
        return _sign_in(state, name)
    if name in F.TABS:
        state.focus = None
        state.go(F.TABS[name])
        return ""
    if name in ("sign_out", "btn_Sign out"):
        return ACT_LOGOUT
    if name == "btn_Back":
        return on_back(state, now)
    if name.startswith("seg_"):
        state.board_seg = name[len("seg_"):]   # U2: viên pill dịch, dữ liệu giữ
        return ""
    if name in F.MENU_BUTTONS:
        state.menu_focus = F.MENU_BUTTONS.index(name)
        state.focus = name
        state.go({"btn_Solo": F.LEVELS, "btn_PvP": F.PVP_MODE,
                  "btn_Leaderboard": F.BOARD,
                  "btn_Settings": F.SETTINGS}[name])
        return ""
    if state.screen == F.SETTINGS:
        return _settings(state, name, point)
    if state.screen in (F.WIN, F.LOSE):
        return _results(state, name)
    if state.screen in (F.PVP_MODE, F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND,
                        F.PVP_HUNT, F.PVP_HIDE, F.PVP_RESULT, F.OFFLINE):
        return _pvp(state, name, now)
    return _solo(state, name)
