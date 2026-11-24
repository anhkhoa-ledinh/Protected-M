"""
Máy trạng thái màn hình của giao diện v3: đang ở màn nào, giữ những gì.

Giữ đúng lối của ``flow.py``: bản ghi phẳng + hàm module, không lớp lồng lớp.
Khác ``flow.py`` ở hai chỗ, và chỉ hai chỗ: tên màn là tên bảng Penpot v3, và
tên cấp/vật phẩm/vũ khí là **tiếng Anh** (bộ asset v3 in chữ Anh, nên khoá nội
bộ phải cùng ngôn ngữ với sprite ``item_tile:<item>``).

TASK_accounts_1119: tài khoản đã THẬT (``src/account``) — sống trên một
máy, không có server. ``player`` vẫn là nội dung ô "Username"/tên hiển
thị; ``account_key`` là khoá đã case-fold trong kho tài khoản (xem
``src.account.crypto.GUEST_KEY`` cho khoá chơi khách), hoặc ``""`` khi
chưa đăng nhập.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

from src.game.pvp import (ITEM_WEIGHT, ITEMS_HAVE_EFFECT, LEVEL_TABLE,
                          PVP_MODES, item_price)
from src.ui_album.ids import (BOARD, LEVELS, LOADING, LOADING_READY, LOADOUT,  # noqa: F401
                                LOSE, MENU, OFFLINE, PHOTOS, PLAY, PVP_FOUND,
                                PVP_HIDE, PVP_HUNT, PVP_INVITE, PVP_MODE,
                                PVP_RESULT, PVP_SEARCH, SETTINGS, SIGN_IN,
                                SIGN_IN_ERR, SIGN_UP, TILT, WIN)
from src.ui_album import ids as ID

logger = logging.getLogger(__name__)

#: Bí danh theo tên của ``flow.py``. ``app.py`` dùng CHUNG một bộ hàm cho cả
#: hai giao diện (``self._album.flow.PIPE0``…); giữ đúng tên thuộc tính ở đây
#: là khỏi phải nhân đôi vòng lặp chính. Bộ v3 gộp "0 %" và "đang chạy" vào
#: một màn, nên ``PIPE0`` và ``PIPE`` trỏ cùng chỗ.
COVER = MENU
PIPE0 = PIPE = LOADING
PIPE100 = LOADING_READY

#: Bọ vẽ bằng sprite của bộ asset cũ (bộ v3 không có hình bọ), mà bảng màu ở
#: đó đánh khoá tiếng Việt — ``BugPainter`` phải nhận đúng token ấy.
BUG_THEME = "ngay"

#: Tab trên cùng → màn. Chạy được ở **mọi** màn có bốn hotspot này.
TABS = {ID.TAB_SOLO: LEVELS, ID.TAB_PVP: PVP_MODE,
        ID.TAB_LEADERBOARD: BOARD, ID.TAB_SETTINGS: SETTINGS}

#: Màn không được rời bằng tab (chưa đăng nhập thì chưa có gì để xem).
LOCKED = (SIGN_IN, SIGN_UP, SIGN_IN_ERR)

WEAPONS = ("Precision gun", "Spread gun", "Heavy-mag gun")

#: Mô tả cấp trên màn chọn cấp.
LEVEL_DESC = {
    1: "Bugs sit still and hide behind things. Learn to peek.",
    2: "More bugs, less time, and some blend into the photo.",
    3: "Bugs hide better and start fighting back.",
    4: "Bugs spread across every layer and bite harder.",
    5: "Stamina counts, and overtime costs points.",
}

#: Dòng ``note`` trên màn trang bị.
LEVEL_NOTE = {
    1: "Level 1 · bugs only hide.",
    2: "Level 2 · some bugs blend in.",
    3: "Level 3 · bugs spit acid and poison.",
    4: "Level 4 · bugs bite harder.",
    5: "Level 5 · stamina and overtime count.",
}

#: Dòng ``tip_text`` trên màn trang bị.
LEVEL_TIP = {
    1: "Items are optional. Fewer items, higher score.",
    2: "Camouflaged bugs are easier to spot with a Filter lens.",
    3: "Bugs at this level spit acid and poison. Bring an Antidote or an "
       "Acid shield.",
    4: "Bugs bite harder. A Smoke bomb stops them fighting back.",
    5: "Missed shots cost stamina. An Energy drink restores it.",
}

#: Con bọ mới xuất hiện ở cấp 3 (khoá ``new_bug`` của màn chọn cấp).
NEW_BUG_LEVEL = 3
NEW_BUG_NAME = "Acid beetle"

#: Mất kết nối: đếm ngược ngần này giây rồi coi như thử lại được.
RECONNECT_S = 15.0

#: Nút "Enter match" tự bật sau ngần này giây, kể cả khi chưa đủ ba dấu kiểm:
#: không có cửa thoát thì webcam kém là tắc cả luồng.
TILT_TIMEOUT_S = 15.0

#: Ngưỡng nghiêng đầu của hai dấu kiểm trái/phải (độ).
TILT_ANGLE_DEG = 8.0

#: Bốn nút của màn chính, theo thứ tự ↑/↓ chạy qua.
MENU_BUTTONS = (ID.BTN_SOLO, ID.BTN_PVP, ID.BTN_LEADERBOARD, ID.BTN_SETTINGS)


@dataclass
class V3Flow:
    """Trạng thái giao diện v3. Một bản ghi phẳng, đọc được từ mọi chỗ."""

    screen: str = SIGN_IN
    player: str = ""
    password: str = ""
    password2: str = ""
    #: ``icon_show_password`` đổi dấu chấm ↔ chữ thật.
    show_password: bool = False
    remember: bool = False
    focus: str | None = None
    level: int = 1
    weapon: int | None = None
    #: Vật phẩm đã gắn vào ô, theo thứ tự ô. Tên tiếng Anh của ``ITEM_WEIGHT``.
    items: list = field(default_factory=list)
    photo = None
    recent: list = field(default_factory=list)
    mode: str = PVP_MODES[0]
    room_code: str = ""
    join_typed: str = ""
    search_started_s: float = 0.0
    search_target_s: float = 4.0
    found_at_s: float = 0.0
    offline_since_s: float = 0.0
    #: TASK_pvp_lan_1116: trạng thái PHIÊN PvP thật — gán khi ``matched`` tới.
    #: ``pvp_role``: "host"/"guest" (Competitive) hoặc "hunter"/"hider"
    #: (Hunt & Hide). ``pvp_started`` bật khi server gửi ``start`` (đếm
    #: ngược 3 s của ``PVP_FOUND`` chỉ chạy sau mốc này, KHÔNG phải từ lúc
    #: vào màn — xem ``update()``).
    pvp_role: str = ""
    pvp_token: str = ""
    pvp_match_id: str = ""
    pvp_started: bool = False
    pvp_opponent_name: str = ""
    #: Hunt & Hide: hạn 20 s để đặt bọ (``PVP_HIDE`` bên hider), đo bằng
    #: ``time.perf_counter()`` cùng đồng hồ với ``found_at_s``.
    pvp_setup_deadline_s: float = 0.0
    #: TASK_pvp_fix_1116: thay cho ``F.say`` 30 s — tính lại MỖI khung từ
    #: trạng thái thật (``app._pvp_lobby_message``), nên không bao giờ sống
    #: lâu hơn chính lý do nó tồn tại. Chỉ ``v3_values._pvp_lobby`` đọc, và
    #: chỉ vẽ ở ``PVP_FOUND`` — không rò sang ``PVP_HUNT``/``PVP_HIDE``/
    #: ``PVP_RESULT``.
    pvp_lobby_message: str = ""
    pipeline_pct: float = 0.0
    #: Bugs the loaded photo can really hold for this level, set once the
    #: layers are ready; ``None`` until then (23/11: honest count).
    bug_hold: int | None = None
    #: Số lớp THẬT của ảnh đang mở; bản vẽ in cứng 4, app phải nói số thật.
    layer_count: int = 0
    #: Số lớp app NHẮM tới (``content.layer_count``, adaptive K 2–4). Chỉ dùng
    #: khi chưa tách lớp xong, tức lúc chưa ai biết con số thật — vẫn hơn hẳn
    #: hằng 4 của bản vẽ, vì nó ít nhất là cấu hình của chính lượt chạy này.
    layer_target: int = 3
    message: str = ""
    message_until_s: float = 0.0
    webcam_index: int = 0
    webcam_name: str = "Built-in webcam"
    volume: int = 70
    display: str = "fullscreen"
    theme: str = "day"
    face_seen: bool = False
    tracker_simulated: bool = False
    #: Góc nghiêng ngang của đầu (độ) và hai dấu kiểm đã đạt.
    tilt_deg: float = 0.0
    tilt_left_done: bool = False
    tilt_right_done: bool = False
    tilt_entered_s: float = 0.0
    menu_focus: int = 0
    board_seg: str = ""
    #: Ô vật phẩm đang chọn trên HUD (1–3) và số lần đã dùng từng vật.
    slot_selected: int = 1
    used: dict = field(default_factory=dict)
    last_result: dict = field(default_factory=dict)
    #: Kết quả gần nhất theo từng ảnh: ``{đường dẫn: {...}}`` — khoá
    #: ``photo_damage`` và ``photo_date`` đọc ở đây.
    history: dict = field(default_factory=dict)
    #: TASK_shared_album_1111: đường dẫn ảnh đang chờ hộp xác nhận chia sẻ
    #: ("" = không hộp nào mở — chỉ việc này điều khiển ``_draw_confirm_dialog``).
    shared_album_pending: str = ""
    #: Set by ``V3UI`` from its ``SharedAlbumService`` before it routes a click
    #: or builds the values: the routes and values stay pure functions of state.
    shared_can_upload: bool = False
    shared_can_list: bool = False
    shared_not_configured: str = ""
    #: Tập ảnh đã xác nhận ít nhất một lần (bấm "Share to album" lại với
    #: ảnh này thì khỏi hỏi lại — P2.3 "before the first upload of each photo").
    shared_album_confirmed: set = field(default_factory=set)
    #: Đường dẫn app phải tải lên NGAY (khác ``shared_album_pending``: ô này
    #: không vẽ hộp thoại nào, chỉ là tín hiệu bàn giao cho ``_album_action``).
    shared_album_upload_path: str = ""
    # ---- TASK_accounts_1119: tài khoản cục bộ (không có server) -----------
    #: Khoá tài khoản (``crypto.username_key``) đang đăng nhập; ``""`` =
    #: chưa đăng nhập, ``"guest"`` = chơi khách. Khác ``player`` (tên hiển
    #: thị, cũng là nội dung ô "Username" trên màn đăng nhập/đăng ký).
    account_key: str = ""
    #: Spendable points of the signed-in account; ``None`` for a guest or
    #: nobody signed in (no wallet, items are free). Kept in step with the
    #: account store by ``V3UI.sync_economy``.
    wallet: int | None = None
    earned_total: int = 0
    levels_cleared: tuple = ()
    #: Ô nào trên màn đăng ký người dùng đã GÕ VÀO rồi — gợi ý lỗi chỉ hiện
    #: cho ô đã động tới (không doạ người mới vào màn với lỗi ngay lập tức).
    touched: set = field(default_factory=set)
    #: Dòng lỗi của màn ``v3_01c_sign_in_error`` (khoá text ``error``) và
    #: màn ``v3_01b_sign_up`` (khoá text ``hint``, khi có vấn đề).
    sign_in_error: str = ""
    hint_text: str = ""
    #: Có đang hiện icon "!" đỏ (``hint_bad``) thay vì dấu tích xanh
    #: (``hint_ok``) trên màn đăng ký hay không.
    hint_bad: bool = False
    #: TASK_accounts_1119 Phase 6: hộp thoại mã khôi phục (hiện MỘT LẦN
    #: ngay sau khi đăng ký) và hộp thoại quên mật khẩu.
    recovery_code_shown: str = ""
    forgot_open: bool = False
    forgot_focus: str | None = None
    forgot_user: str = ""
    forgot_code: str = ""
    forgot_new_pass: str = ""
    forgot_new_pass2: str = ""
    forgot_error: str = ""
    forgot_touched: set = field(default_factory=set)
    #: TASK_small_fixes_1120: character index of the caret inside WHICHEVER
    #: field currently has it (``focus`` or, while the modal dialog is
    #: open, ``forgot_focus``) — only one field can hold the caret at a
    #: time app-wide, so one shared index is enough.
    caret_index: int = 0

    # ---- bảng cấp --------------------------------------------------------
    def level_name(self) -> str:
        return LEVEL_TABLE[self.level][0]

    def bugs(self) -> int:
        return LEVEL_TABLE[self.level][1]

    def seconds(self) -> int:
        return LEVEL_TABLE[self.level][2]

    def level_slots(self) -> int:
        return LEVEL_TABLE[self.level][3]

    def slots(self) -> int:
        """Số ô dùng được: bản vẽ chỉ có ba ô nên trần là 3 (U2 — Loadout)."""
        return min(self.level_slots(), 3)

    def weapon_name(self) -> str:
        if self.weapon is None:
            return ""
        return WEAPONS[self.weapon % len(WEAPONS)]

    def weight_total(self) -> int:
        return sum(ITEM_WEIGHT.get(name, 0) for name in self.items)

    def price_total(self, items=None) -> int:
        """Points the equipped items (or ``items``) cost at match start."""
        return sum(item_price(name)
                   for name in (self.items if items is None else items))

    def go(self, screen: str) -> None:
        if screen != self.screen:
            logger.info("UI v3: %s -> %s", self.screen, screen)
            if self.screen in (PLAY, PVP_HUNT, PVP_HIDE):
                # TASK_game_feel_1118 P3: toast "trận" (vd "Reloading")
                # không được rò sang màn SAU trận — review ảnh 1117 bắt
                # được nó còn hiện nguyên trên WIN (``fx_win.png``).
                self.message = ""
                self.message_until_s = 0.0
            if screen == TILT:
                self.tilt_entered_s = time.perf_counter()
                self.tilt_left_done = self.tilt_right_done = False
            if screen == SIGN_UP and self.screen != SIGN_UP:
                # TASK_accounts_1119: hint/lỗi chỉ hiện cho ô đã gõ vào —
                # vào màn đăng ký MỚI (không phải bật qua tắt lại chính
                # màn này) thì chưa ô nào được coi là đã động tới.
                self.touched = set()
        self.screen = screen

    # ---- dấu kiểm của màn thử nghiêng -----------------------------------
    def tilt_checks(self) -> list[bool]:
        """Ba dấu: thấy mặt, đã nghiêng trái, đã nghiêng phải."""
        return [bool(self.face_seen), bool(self.tilt_left_done),
                bool(self.tilt_right_done)]

    def tilt_ready(self) -> bool:
        """Nút "Enter match" bật khi đủ ba dấu, HOẶC tracker mô phỏng, HOẶC quá giờ."""
        if all(self.tilt_checks()) or self.tracker_simulated:
            return True
        started = self.tilt_entered_s or time.perf_counter()
        return (time.perf_counter() - started) >= TILT_TIMEOUT_S


#: Ô nhập → tên thuộc tính giữ chuỗi của nó. Ba chuỗi RIÊNG: dùng chung một
#: biến thì gõ ô này ô kia cũng đầy theo (lỗi người dùng gặp 06/11 23:01).
FIELD_ATTR = {ID.FIELD_USER: "player", ID.FIELD_PASS: "password",
              ID.FIELD_PASS2: "password2"}

#: TASK_accounts_1119: trần ký tự riêng từng ô — mật khẩu hợp lệ dài tới 64
#: (``crypto.validate_password``), nên trần 24 cũ (dùng chung mọi ô) sẽ cắt
#: mật khẩu dài mà người dùng không hề biết ("Raise the field cap so a
#: password is never truncated silently").
FIELD_MAXLEN = {ID.FIELD_USER: 32, ID.FIELD_PASS: 64, ID.FIELD_PASS2: 64}

#: Thứ tự ô nhập khi bấm Tab.
FIELD_ORDER = (ID.FIELD_USER, ID.FIELD_PASS, ID.FIELD_PASS2)

#: TASK_accounts_1119 Phase 6: the forgot-password dialog's own fields —
#: declared here (next to ``FIELD_ATTR``) so ``field_value``/the caret
#: helpers below can read both tables without a forward reference.
FORGOT_FIELD_ATTR = {ID.FIELD_FORGOT_USER: "forgot_user",
                     ID.FIELD_FORGOT_CODE: "forgot_code",
                     ID.FIELD_FORGOT_PASS: "forgot_new_pass",
                     ID.FIELD_FORGOT_PASS2: "forgot_new_pass2"}
#: The code field gets slack for the dashes/spaces a user might type —
#: normalised (stripped + upper-cased) only at submit time.
FORGOT_FIELD_MAXLEN = {ID.FIELD_FORGOT_USER: 32, ID.FIELD_FORGOT_CODE: 20,
                       ID.FIELD_FORGOT_PASS: 64, ID.FIELD_FORGOT_PASS2: 64}
FORGOT_FIELD_ORDER = (ID.FIELD_FORGOT_USER, ID.FIELD_FORGOT_CODE,
                      ID.FIELD_FORGOT_PASS, ID.FIELD_FORGOT_PASS2)


def field_value(flow: V3Flow, name: str | None) -> str:
    """Current string content of editable field ``name`` — ``""`` for
    anything that isn't one (callers don't need to special-case buttons).

    TASK_small_fixes_1120: ``field_join`` (the PvP 4-digit room code) is
    deliberately NOT covered — it keeps its own append/backspace-last
    model, a PIN-style entry rather than a text field, and isn't one of
    the fields this task's gates (or its own wording) names."""
    attr = FIELD_ATTR.get(name or "") or FORGOT_FIELD_ATTR.get(name or "")
    return getattr(flow, attr) if attr else ""


def _is_caret_field(name: str | None) -> bool:
    return name in FIELD_ATTR or name in FORGOT_FIELD_ATTR


def _edit_insert(value: str, caret: int, char: str, maxlen: int) -> tuple[str, int]:
    """Inserts ``char`` AT the caret; a full field drops the keystroke
    instead of truncating from the front (TASK_small_fixes_1120: "max-
    length limits ... stay as they are")."""
    if len(value) >= maxlen:
        return value, caret
    caret = max(0, min(caret, len(value)))
    return value[:caret] + char + value[caret:], caret + 1


def _edit_backspace(value: str, caret: int) -> tuple[str, int]:
    """Deletes the character LEFT of the caret."""
    caret = max(0, min(caret, len(value)))
    if caret == 0:
        return value, caret
    return value[:caret - 1] + value[caret:], caret - 1


def _edit_delete(value: str, caret: int) -> tuple[str, int]:
    """Delete key: removes the character RIGHT of the caret."""
    caret = max(0, min(caret, len(value)))
    if caret >= len(value):
        return value, caret
    return value[:caret] + value[caret + 1:], caret


def type_char(flow: V3Flow, char: str) -> None:
    """Gõ một ký tự vào **đúng vị trí con trỏ** của ô đang chọn."""
    if flow.focus == ID.FIELD_JOIN:
        if char.isdigit() and len(flow.join_typed) < 4:
            flow.join_typed += char
        return
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        maxlen = FIELD_MAXLEN.get(flow.focus, 24)
        value, flow.caret_index = _edit_insert(getattr(flow, attr),
                                               flow.caret_index, char, maxlen)
        setattr(flow, attr, value)
        # TASK_accounts_1119: gợi ý lỗi trên màn đăng ký chỉ hiện cho ô
        # người dùng đã THẬT SỰ gõ vào — đánh dấu ngay lần gõ đầu tiên.
        flow.touched.add(flow.focus)


def backspace(flow: V3Flow) -> None:
    if flow.focus == ID.FIELD_JOIN:
        flow.join_typed = flow.join_typed[:-1]
        return
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        value, flow.caret_index = _edit_backspace(getattr(flow, attr),
                                                   flow.caret_index)
        setattr(flow, attr, value)
        flow.touched.add(flow.focus)


def delete_char(flow: V3Flow) -> None:
    """Delete key: removes the character to the RIGHT of the caret."""
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        value, flow.caret_index = _edit_delete(getattr(flow, attr), flow.caret_index)
        setattr(flow, attr, value)
        flow.touched.add(flow.focus)


def focus_next(flow: V3Flow, fields) -> None:
    """Tab: sang ô nhập kế tiếp của màn (vòng lại ô đầu)."""
    order = [name for name in FIELD_ORDER if name in fields]
    if not order:
        return
    if flow.focus in order:
        flow.focus = order[(order.index(flow.focus) + 1) % len(order)]
    else:
        flow.focus = order[0]
    flow.caret_index = len(field_value(flow, flow.focus))


def move_caret(flow: V3Flow, delta: int) -> None:
    """Left/Right: shift the caret one character, inside whichever field
    currently holds it (the screen's own, or the forgot dialog's)."""
    name = flow.forgot_focus if flow.forgot_open else flow.focus
    if not _is_caret_field(name):
        return
    flow.caret_index = max(0, min(len(field_value(flow, name)),
                                  flow.caret_index + int(delta)))


def caret_home(flow: V3Flow) -> None:
    name = flow.forgot_focus if flow.forgot_open else flow.focus
    if _is_caret_field(name):
        flow.caret_index = 0


def caret_end(flow: V3Flow) -> None:
    name = flow.forgot_focus if flow.forgot_open else flow.focus
    if _is_caret_field(name):
        flow.caret_index = len(field_value(flow, name))


# ---- TASK_accounts_1119 Phase 6: the forgot-password dialog's own ----------
#: fields — a SEPARATE focus (``flow.forgot_focus``) from the screen
#: UNDERNEATH it (normally SIGN_IN, which keeps its own ``flow.focus``
#: untouched while the dialog is open, same as the share-album confirm
#: dialog leaves the screen under it alone). The field tables themselves
#: moved up next to ``FIELD_ATTR`` (TASK_small_fixes_1120) — this comment
#: stays here as the focus split's own explanation.


def forgot_type_char(flow: V3Flow, char: str) -> None:
    attr = FORGOT_FIELD_ATTR.get(flow.forgot_focus or "")
    if attr:
        maxlen = FORGOT_FIELD_MAXLEN.get(flow.forgot_focus, 24)
        value, flow.caret_index = _edit_insert(getattr(flow, attr),
                                               flow.caret_index, char, maxlen)
        setattr(flow, attr, value)
        flow.forgot_touched.add(flow.forgot_focus)


def forgot_backspace(flow: V3Flow) -> None:
    attr = FORGOT_FIELD_ATTR.get(flow.forgot_focus or "")
    if attr:
        value, flow.caret_index = _edit_backspace(getattr(flow, attr),
                                                   flow.caret_index)
        setattr(flow, attr, value)
        flow.forgot_touched.add(flow.forgot_focus)


def forgot_delete_char(flow: V3Flow) -> None:
    attr = FORGOT_FIELD_ATTR.get(flow.forgot_focus or "")
    if attr:
        value, flow.caret_index = _edit_delete(getattr(flow, attr), flow.caret_index)
        setattr(flow, attr, value)
        flow.forgot_touched.add(flow.forgot_focus)


def forgot_focus_next(flow: V3Flow) -> None:
    order = FORGOT_FIELD_ORDER
    if flow.forgot_focus in order:
        flow.forgot_focus = order[(order.index(flow.forgot_focus) + 1) % len(order)]
    else:
        flow.forgot_focus = order[0]
    flow.caret_index = len(field_value(flow, flow.forgot_focus))


def move_menu_focus(flow: V3Flow, delta: int) -> None:
    """↑/↓ trên màn chính: vòng focus chạy qua bốn nút."""
    if flow.screen != MENU:
        return
    count = len(MENU_BUTTONS)
    flow.menu_focus = (flow.menu_focus + int(delta)) % count
    flow.focus = MENU_BUTTONS[flow.menu_focus]


def tilt_angle(flow: V3Flow, eye_x: float, eye_z: float) -> float:
    """
    Góc nghiêng ngang của đầu, độ: ``atan2(eye_x, eye_z)`` (U2 — Tilt).

    Ghi luôn hai dấu kiểm trái/phải: đã qua ±8° một lần là tính.
    """
    import math

    angle = math.degrees(math.atan2(float(eye_x), max(1.0, abs(float(eye_z)))))
    flow.tilt_deg = angle
    if angle <= -TILT_ANGLE_DEG:
        flow.tilt_left_done = True
    elif angle >= TILT_ANGLE_DEG:
        flow.tilt_right_done = True
    return angle


def update(flow: V3Flow, now: float) -> None:
    """
    Nhịp thời gian của màn ``PVP_FOUND``: đếm ngược 3 s rồi vào trận.

    TASK_pvp_lan_1116: tìm trận/mời phòng không còn là đồng hồ giả ở đây —
    ``app.py`` đưa màn sang ``PVP_FOUND`` khi server báo ``matched`` thật,
    và chỉ bật ``pvp_started`` khi server gửi ``start`` (ảnh đã truyền, cả
    hai máy đã sẵn sàng). Đếm ngược vì thế tính từ lúc ``start`` tới, không
    phải từ lúc vào màn — nếu không, hai máy có thể vào trận lệch nhau vài
    giây tuỳ máy nào bake lớp xong trước.
    """
    if flow.screen == PVP_FOUND and flow.pvp_started and now - flow.found_at_s >= 3.0:
        if flow.mode == PVP_MODES[0] or flow.pvp_role == "hunter":
            flow.go(PVP_HUNT)
        else:
            flow.go(PVP_HIDE)


def say(flow: V3Flow, text: str, seconds: float = 4.0) -> None:
    """Hiện một dòng báo trên giao diện."""
    flow.message = str(text)
    flow.message_until_s = time.perf_counter() + float(seconds)
    logger.info("UI v3 bao: %s", text)


def start_search(flow: V3Flow, now: float) -> None:
    flow.search_started_s = now
    flow.search_target_s = random.uniform(3.0, 5.0)
    flow.go(PVP_SEARCH)


#: PvP opens once every solo level has been won at least once.
PVP_UNLOCK_LEVELS = (1, 2, 3, 4, 5)


def pvp_lock_reason(flow: V3Flow) -> str | None:
    """Why PvP is closed to this player, or ``None`` when it is open."""
    if flow.wallet is None:                       # guest or nobody signed in
        return "Sign in and win levels 1–5 to unlock PvP"
    done = sum(1 for n in PVP_UNLOCK_LEVELS if n in flow.levels_cleared)
    if done < len(PVP_UNLOCK_LEVELS):
        return (f"Win levels 1–5 to unlock PvP — "
                f"{done}/{len(PVP_UNLOCK_LEVELS)} done")
    return None


def toggle_item(flow: V3Flow, item: str) -> None:
    """Gắn/bỏ một vật phẩm. Hết ô hoặc hết điểm thì báo, không im lặng bỏ qua."""
    if item in flow.items:
        flow.items.remove(item)
        return
    if len(flow.items) >= flow.slots():
        say(flow, f"All {flow.slots()} item slots are full.")
        return
    if flow.wallet is not None:
        need = flow.price_total(flow.items + [item])
        if need > flow.wallet:
            say(flow, f"Not enough points: {item} costs {item_price(item)}, "
                      f"you have {flow.wallet - flow.price_total()} left to spend.")
            return
    flow.items.append(item)


def remove_slot(flow: V3Flow, index: int) -> None:
    """Bấm vào một ô đã có vật phẩm = bỏ vật phẩm đó ra."""
    if 0 <= index < len(flow.items):
        del flow.items[index]
