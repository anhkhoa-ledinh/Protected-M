"""
Bấm vào hotspot nào thì đi đâu — bảng định tuyến của giao diện v3.

Tách khỏi ``v3_flow.py`` để mỗi file một việc: ở đó là *trạng thái*, ở đây là
*phản ứng*. ``on_click`` trả về tên hành động mà app phải tự làm (mở hộp chọn
file, chạy tách lớp, đo mắt nghỉ…), hoặc ``""`` nếu chỉ đổi màn — y hợp đồng
của ``routes.py``, nên ``app.py`` dùng lại nguyên bộ hành động cũ.
"""

from __future__ import annotations

import logging

from src.shared import account_rules
from src.content import shared_album
from src.ui_album import v3_flow as F
from src.ui_album import ids as ID

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
ACT_DISPLAY_FULL = ID.DISPLAY_FULL
ACT_DISPLAY_WINDOW = ID.DISPLAY_WINDOW
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
    F.SIGN_IN: ID.BTN_SIGN_IN, F.SIGN_UP: ID.BTN_CREATE_ALBUM,
    F.SIGN_IN_ERR: ID.BTN_SIGN_IN, F.MENU: None,
    F.LEVELS: ID.BTN_NEXT_CHOOSE_A_PHOTO,
    F.PHOTOS: ID.BTN_SPLIT_LAYERS_CONTINUE,
    F.LOADING_READY: ID.CONTINUE, F.LOADOUT: ID.BTN_START,
    F.TILT: ID.BTN_ENTER_MATCH, F.WIN: ID.BTN_NEXT_LEVEL,
    F.LOSE: ID.BTN_TRY_AGAIN, F.PVP_MODE: ID.BTN_FIND_MATCH,
    F.PVP_INVITE: ID.BTN_JOIN_ROOM, F.PVP_RESULT: ID.BTN_REMATCH,
    F.OFFLINE: ID.BTN_TRY_AGAIN_NOW, F.SETTINGS: ID.BTN_RECALIBRATE_LOWER,
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
    if name == ID.BTN_SIGN_IN:
        state.focus = None
        if not state.player.strip() or not state.password:
            state.sign_in_error = "Enter a username and password."
            state.go(F.SIGN_IN_ERR)
            return ""
        return ACT_SIGN_IN
    elif name == ID.BTN_CREATE_ALBUM and state.screen == F.SIGN_UP:
        state.focus = None
        # Chạm hết ba ô: bấm thẳng nút mà chưa gõ ô nào (vd. chỉ gõ mật
        # khẩu rồi Tab qua nút) vẫn phải thấy đúng gợi ý, không im lặng.
        state.touched |= {ID.FIELD_USER, ID.FIELD_PASS, ID.FIELD_PASS2}
        display, reason = account_rules.validate_username(state.player)
        if display is None:
            return ""                          # gợi ý đã hiện qua state_for
        pass_ok, _reason = account_rules.validate_password(state.password)
        if not pass_ok or state.password != state.password2:
            return ""
        return ACT_SIGN_UP
    elif name in (ID.BTN_CREATE_ALBUM, ID.BTN_CREATE_A_NEW_ALBUM):
        state.focus = None
        state.go(F.SIGN_UP)                    # trên màn đăng nhập = sang đăng ký
    elif name == ID.FOOTER_LINK:
        state.focus = None
        # Màn đăng ký: về đăng nhập. Màn lỗi: vào bằng khách.
        if state.screen == F.SIGN_UP:
            state.go(F.SIGN_IN)
        else:
            state.player = state.player or "guest"
            state.account_key = account_rules.GUEST_KEY
            # TASK_accounts_1119 Phase 5: khách không có tiến trình riêng
            # — nếu còn history/recent của tài khoản TRƯỚC đó (đăng xuất
            # rồi vào khách) thì phải dọn, không thì màn chính/bảng xếp
            # hạng hiện nhầm dữ liệu của người khác.
            state.history, state.recent = {}, []
            state.go(F.MENU)
    elif name == ID.LINK_GUEST:
        state.focus = None
        state.player = state.player or "guest"
        state.account_key = account_rules.GUEST_KEY
        state.history, state.recent = {}, []
        state.go(F.MENU)
    elif name == ID.LINK_FORGOT:
        state.forgot_open = True
        state.forgot_user = state.player
        state.forgot_code = ""
        state.forgot_new_pass = state.forgot_new_pass2 = ""
        state.forgot_error = ""
        state.forgot_touched = set()
        state.forgot_focus = ID.FIELD_FORGOT_USER
    elif name == ID.SWITCH_REMEMBER:
        state.remember = not state.remember
    elif name == ID.ICON_SHOW_PASSWORD:
        state.show_password = not state.show_password
    elif name.startswith("field_"):
        state.focus = name
    return ""


def _settings(state, name: str, point) -> str:
    if name == ID.THEME_DAY:
        state.theme = "day"
    elif name == ID.THEME_NIGHT:
        # Bộ v3 chỉ có một bảng màu. Nói thẳng, và KHÔNG đổi cái điều khiển:
        # để nó nhảy sang Night rồi màn vẫn sáng là hứa điều không có.
        F.say(state, "Night theme is not in v2 yet.")
    elif name == ID.DISPLAY_FULL:
        return ACT_DISPLAY_FULL
    elif name == ID.DISPLAY_WINDOW:
        return ACT_DISPLAY_WINDOW
    elif name == ID.DROPDOWN_WEBCAM:
        state.webcam_index = (state.webcam_index + 1) % 2
        F.say(state, "Switching webcam isn't supported while running.")
    elif name == ID.BTN_RECALIBRATE_LOWER:
        return ACT_REST_EYE
    elif name == ID.SLIDER:
        state.volume = _slider_value(state, point)
    elif name == ID.BTN_SIGN_OUT:
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
    if name == ID.BTN_CONFIRM_NO:
        return ""
    state.shared_album_confirmed.add(path)
    state.shared_album_upload_path = path
    return ACT_SHARE_CONFIRMED


def _recovery_dialog(state, name: str) -> str:
    """TASK_accounts_1119 Phase 6: the one-time recovery-code dialog —
    "Copy" (clipboard, app.py's job) and "I saved it" (closes it)."""
    if name == ID.BTN_RECOVERY_COPY:
        return ACT_COPY_RECOVERY_CODE
    elif name == ID.BTN_RECOVERY_SAVED:
        state.recovery_code_shown = ""
        return ACT_RECOVERY_SAVED
    return ""


def _forgot_dialog(state, name: str) -> str:
    """The forgot-password dialog's own four fields + Reset/Cancel.

    Same split as sign-up: only FORMAT-level checks happen here
    (synchronous, no store access) — the code itself can only be
    verified against the store, off-thread, via ``ACT_FORGOT_SUBMIT``.
    """
    if name.startswith("field_forgot_"):
        state.forgot_focus = name
    elif name == ID.BTN_FORGOT_CANCEL:
        state.forgot_open = False
        state.forgot_focus = None
        state.forgot_error = ""
    elif name == ID.BTN_FORGOT_SUBMIT:
        state.forgot_touched |= {ID.FIELD_FORGOT_USER, ID.FIELD_FORGOT_CODE,
                                 ID.FIELD_FORGOT_PASS, ID.FIELD_FORGOT_PASS2}
        display, reason = account_rules.validate_username(state.forgot_user)
        if display is None:
            state.forgot_error = reason
            return ""
        code = account_rules.normalize_recovery_code(state.forgot_code)
        if len(code) != account_rules.RECOVERY_CODE_CHARS:
            state.forgot_error = "Enter the 12-character code."
            return ""
        pass_ok, reason2 = account_rules.validate_password(state.forgot_new_pass)
        if not pass_ok:
            state.forgot_error = reason2
            return ""
        if state.forgot_new_pass != state.forgot_new_pass2:
            state.forgot_error = "Passwords don't match"
            return ""
        state.forgot_error = ""
        return ACT_FORGOT_SUBMIT
    return ""


def _solo(state, name: str) -> str:
    if name == ID.BTN_SHARE_TO_ALBUM:
        if not shared_album.can_upload():
            pass                     # nhãn nút đã nói "not configured"
        elif not state.photo:
            F.say(state, "Choose a photo first.")
        elif state.photo in state.shared_album_confirmed:
            state.shared_album_upload_path = state.photo
            return ACT_SHARE_CONFIRMED  # đã hỏi ảnh này rồi, khỏi hỏi lại
        else:
            state.shared_album_pending = state.photo
    elif name == ID.BTN_RANDOM_FROM_ALBUM:
        if shared_album.load_config() is not None:
            return ACT_RANDOM_FROM_ALBUM  # nhan da noi "not configured" neu khong
    elif name.startswith("level_row_"):
        state.level = int(name.rsplit("_", 1)[1])
    elif name == ID.BTN_NEXT_CHOOSE_A_PHOTO:
        state.go(F.PHOTOS)
    elif name in (ID.BTN_CHOOSE_FROM_COMPUTER, ID.DROPZONE):
        return ACT_PICK_FILE
    elif name == ID.RECENT_MEMORY:
        if state.recent:
            state.photo = state.recent[0]
    elif name == ID.BTN_SPLIT_LAYERS_CONTINUE:
        return ACT_START_PIPELINE
    elif name == ID.CONTINUE:
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
    elif name == ID.BTN_START:
        if state.weapon is None:
            F.say(state, "Choose a weapon first.")
        else:
            state.go(F.TILT)
    elif name in (ID.BTN_RECALIBRATE, ID.BTN_RECALIBRATE_LOWER):
        return ACT_REST_EYE
    elif name == ID.BTN_ENTER_MATCH:
        if not state.tilt_ready():
            F.say(state, "Finish the three checks first.")
            return ""
        return ACT_START_GAME
    return ""


def _results(state, name: str) -> str:
    if name == ID.BTN_NEXT_LEVEL:
        return ACT_NEXT_LEVEL
    if name in (ID.BTN_PLAY_AGAIN, ID.BTN_TRY_AGAIN):
        state.go(F.TILT)
    elif name == ID.BTN_CHOOSE_ANOTHER_PHOTO:
        state.go(F.PHOTOS)
    elif name == ID.BTN_BACK_TO_ALBUM:
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
    elif name == ID.BTN_FIND_MATCH:
        F.start_search(state, now)          # chỉ để "search_time" có mốc
        return ACT_PVP_FIND
    elif name == ID.BTN_INVITE_A_FRIEND:
        state.room_code = ""                 # trống tới khi server trả "room"
        state.join_typed = ""
        state.go(F.PVP_INVITE)
        return ACT_PVP_CREATE_ROOM
    elif name == ID.BTN_COPY_CODE:
        return ACT_COPY_CODE
    elif name == ID.FIELD_JOIN:
        state.focus = ID.FIELD_JOIN
    elif name == ID.BTN_JOIN_ROOM:
        if len(state.join_typed) != 4:
            F.say(state, "Enter the 4-digit room code first.")
            return ""
        return ACT_PVP_JOIN_ROOM
    elif name == ID.BTN_CLOSE_ROOM:
        state.go(F.PVP_MODE)
        return ACT_PVP_CANCEL
    elif name in (ID.BTN_CANCEL, ID.LEAVE):
        state.go(F.PVP_MODE)
        return ACT_PVP_CANCEL
    elif name == ID.BTN_REMATCH:
        return ACT_PVP_REMATCH
    elif name == ID.BTN_LEADERBOARD:
        state.go(F.BOARD)
    elif name == ID.BTN_TRY_AGAIN_NOW:
        state.go(F.PVP_MODE)
        return ACT_PVP_RECONNECT
    elif name == ID.BTN_BACK_TO_ALBUM:
        state.go(F.MENU)
    return ""


def on_click(state, name: str | None, now: float, point=None) -> str:
    """Xử lý một cú bấm. Trả tên hành động cho app, hoặc chuỗi rỗng."""
    if not name:
        return ""
    logger.info("UI v3 click: %s (man %s)", name, state.screen)
    if state.shared_album_pending and name in (ID.BTN_CONFIRM_YES, ID.BTN_CONFIRM_NO):
        return _confirm_share(state, name)
    if state.recovery_code_shown:
        return _recovery_dialog(state, name)
    if state.forgot_open:
        return _forgot_dialog(state, name)
    if state.screen in F.LOCKED:
        return _sign_in(state, name)
    if name in F.TABS:
        state.focus = None
        _open_tab(state, F.TABS[name])
        return ""
    if name in (ID.SIGN_OUT, ID.BTN_SIGN_OUT):
        return ACT_LOGOUT
    if name == ID.BTN_BACK:
        return on_back(state, now)
    if name.startswith("seg_"):
        state.board_seg = name[len("seg_"):]   # U2: viên pill dịch, dữ liệu giữ
        return ""
    if name in F.MENU_BUTTONS:
        state.menu_focus = F.MENU_BUTTONS.index(name)
        state.focus = name
        _open_tab(state, {ID.BTN_SOLO: F.LEVELS, ID.BTN_PVP: F.PVP_MODE,
                          ID.BTN_LEADERBOARD: F.BOARD,
                          ID.BTN_SETTINGS: F.SETTINGS}[name])
        return ""
    if state.screen == F.SETTINGS:
        return _settings(state, name, point)
    if state.screen in (F.WIN, F.LOSE):
        return _results(state, name)
    if state.screen in (F.PVP_MODE, F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND,
                        F.PVP_HUNT, F.PVP_HIDE, F.PVP_RESULT, F.OFFLINE):
        return _pvp(state, name, now)
    return _solo(state, name)


def _open_tab(state, target: str) -> None:
    """Go to a top-level screen; PvP stays shut (with the reason) until
    every solo level has been won."""
    if target == F.PVP_MODE and state.screen not in _PVP_SCREENS:
        reason = F.pvp_lock_reason(state)
        if reason is not None:
            F.say(state, reason)
            return
    state.go(target)


_PVP_SCREENS = (F.PVP_MODE, F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND,
                F.PVP_HUNT, F.PVP_HIDE, F.PVP_RESULT, F.OFFLINE)


# ---- TASK_perf_textures_1119 Fix 3: prefetch graph ------------------------
#
# For every v3 screen, the screens reachable by exactly one click — hand-
# built from ``on_click``/``BACK``/``TABS`` above (plus app.py's own
# ``state.go`` calls for the off-thread actions this module only REQUESTS:
# sign-in/up success, the split-layers pipeline, match start/end, PvP
# matchmaking). tests/test_screen_prefetch.py cross-checks every
# ``state.go(...)`` target actually found in THIS file against it.
#
# Tabs (``F.TABS``) work from every unlocked screen, so they're added to
# every entry below except the three ``F.LOCKED`` ones, instead of repeated
# by hand on each row.
_TAB_NEIGHBOURS = (F.LEVELS, F.PVP_MODE, F.BOARD, F.SETTINGS)

_NEXT_SCREENS_BASE: dict[str, tuple[str, ...]] = {
    F.SIGN_IN: (F.SIGN_UP, F.SIGN_IN_ERR, F.MENU),
    F.SIGN_UP: (F.SIGN_IN, F.MENU),
    F.SIGN_IN_ERR: (F.SIGN_IN, F.SIGN_UP, F.MENU),
    F.MENU: (F.LEVELS, F.PVP_MODE, F.BOARD, F.SETTINGS),
    F.SETTINGS: (F.MENU, F.SIGN_IN),
    F.OFFLINE: (F.PVP_MODE,),
    F.LEVELS: (F.MENU, F.PHOTOS),
    F.PHOTOS: (F.LEVELS, F.LOADING),
    F.LOADING: (F.PHOTOS, F.LOADING_READY),
    F.LOADING_READY: (F.PHOTOS, F.LOADOUT),
    F.LOADOUT: (F.PHOTOS, F.TILT),
    F.TILT: (F.LOADOUT, F.PLAY),
    F.PLAY: (F.WIN, F.LOSE),
    F.WIN: (F.MENU, F.TILT, F.PHOTOS, F.LEVELS),
    F.LOSE: (F.MENU, F.TILT, F.PHOTOS, F.LEVELS),
    F.PVP_MODE: (F.MENU, F.PVP_SEARCH, F.PVP_INVITE),
    F.PVP_SEARCH: (F.PVP_MODE, F.PVP_FOUND, F.OFFLINE),
    F.PVP_INVITE: (F.PVP_MODE, F.PVP_FOUND),
    F.PVP_FOUND: (F.PVP_HUNT, F.PVP_HIDE, F.PVP_MODE),
    F.PVP_HUNT: (F.PVP_RESULT, F.PVP_MODE),
    F.PVP_HIDE: (F.PVP_RESULT, F.PVP_MODE),
    F.PVP_RESULT: (F.MENU, F.PVP_MODE, F.BOARD),
    F.BOARD: (F.MENU,),
}


def _build_next_screens() -> dict[str, tuple[str, ...]]:
    out: dict[str, tuple[str, ...]] = {}
    for screen, neighbours in _NEXT_SCREENS_BASE.items():
        extra = () if screen in F.LOCKED else (s for s in _TAB_NEIGHBOURS if s != screen)
        out[screen] = tuple(dict.fromkeys((*neighbours, *extra)))
    return out


#: ``screen -> (screens one click away)``. See ``V3UI._prefetch_neighbours``.
NEXT_SCREENS: dict[str, tuple[str, ...]] = _build_next_screens()
