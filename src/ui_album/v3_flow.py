"""
Máy trạng thái màn hình của giao diện v3: đang ở màn nào, giữ những gì.

Giữ đúng lối của ``flow.py``: bản ghi phẳng + hàm module, không lớp lồng lớp.
Khác ``flow.py`` ở hai chỗ, và chỉ hai chỗ: tên màn là tên bảng Penpot v3, và
tên cấp/vật phẩm/vũ khí là **tiếng Anh** (bộ asset v3 in chữ Anh, nên khoá nội
bộ phải cùng ngôn ngữ với sprite ``item_tile:<item>``).

Phần PvP và tài khoản vẫn là **mô phỏng** (bot, trạng thái trong bộ nhớ) —
báo cáo phải nói rõ.
"""

from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

SIGN_IN, SIGN_UP = "v3_01a_sign_in", "v3_01b_sign_up"
SIGN_IN_ERR = "v3_01c_sign_in_error"
MENU, SETTINGS = "v3_02a_menu", "v3_02b_settings"
OFFLINE = "v3_02c_disconnected"
LEVELS, PHOTOS = "v3_03_level_select", "v3_04_choose_photo"
LOADING, LOADING_READY = "v3_07_loading", "v3_07_loading_ready"
LOADOUT, TILT = "v3_05_loadout", "v3_06_head_tilt_check"
PLAY = "v3_08_gameplay_hud"
WIN, LOSE = "v3_09a_result_win", "v3_09b_result_lose"
PVP_MODE, PVP_SEARCH = "v3_10_pvp_mode", "v3_11a_pvp_searching"
PVP_INVITE, PVP_FOUND = "v3_11b_pvp_invite", "v3_11c_pvp_found"
PVP_HUNT, PVP_HIDE = "v3_12a_pvp_hunter_hud", "v3_12b_pvp_bug_hud"
PVP_RESULT, BOARD = "v3_13_pvp_result", "v3_14_leaderboard"

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
TABS = {"tab_Solo": LEVELS, "tab_PvP": PVP_MODE,
        "tab_Leaderboard": BOARD, "tab_Settings": SETTINGS}

#: Màn không được rời bằng tab (chưa đăng nhập thì chưa có gì để xem).
LOCKED = (SIGN_IN, SIGN_UP, SIGN_IN_ERR)

#: Bảng cấp: (tên, số bọ, giây, ô vật phẩm, phản công). Con số giữ nguyên của
#: ``flow.py``; chỉ cái tên đổi sang tiếng Anh theo U2.
LEVEL_TABLE = {
    1: ("Easy", 6, 180, 2, False), 2: ("Normal", 8, 150, 3, False),
    3: ("Hard", 10, 150, 3, True), 4: ("Infestation", 12, 120, 4, True),
    5: ("Asian Mode", 12, 120, 4, True),
}

ITEM_WEIGHT = {"Flare": 3, "Antidote": 5, "Acid shield": 4, "Filter lens": 4,
               "Compass": 2, "Thermal scan": 8, "Energy drink": 3,
               "Smoke bomb": 6}

WEAPONS = ("Precision gun", "Spread gun", "Heavy-mag gun")

PVP_MODES = ("Competitive", "Hunt & Hide")

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
MENU_BUTTONS = ("btn_Solo", "btn_PvP", "btn_Leaderboard", "btn_Settings")


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
    pipeline_pct: float = 0.0
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
    #: Tập ảnh đã xác nhận ít nhất một lần (bấm "Share to album" lại với
    #: ảnh này thì khỏi hỏi lại — P2.3 "before the first upload of each photo").
    shared_album_confirmed: set = field(default_factory=set)
    #: Đường dẫn app phải tải lên NGAY (khác ``shared_album_pending``: ô này
    #: không vẽ hộp thoại nào, chỉ là tín hiệu bàn giao cho ``_album_action``).
    shared_album_upload_path: str = ""

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

    def go(self, screen: str) -> None:
        if screen != self.screen:
            logger.info("UI v3: %s -> %s", self.screen, screen)
            if screen == TILT:
                self.tilt_entered_s = time.perf_counter()
                self.tilt_left_done = self.tilt_right_done = False
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
FIELD_ATTR = {"field_user": "player", "field_pass": "password",
              "field_pass2": "password2"}

#: Thứ tự ô nhập khi bấm Tab.
FIELD_ORDER = ("field_user", "field_pass", "field_pass2")


def type_char(flow: V3Flow, char: str) -> None:
    """Gõ một ký tự vào **đúng** ô đang chọn."""
    if flow.focus == "field_join":
        if char.isdigit() and len(flow.join_typed) < 4:
            flow.join_typed += char
        return
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        setattr(flow, attr, (getattr(flow, attr) + char)[:24])


def backspace(flow: V3Flow) -> None:
    if flow.focus == "field_join":
        flow.join_typed = flow.join_typed[:-1]
        return
    attr = FIELD_ATTR.get(flow.focus or "")
    if attr:
        setattr(flow, attr, getattr(flow, attr)[:-1])


def focus_next(flow: V3Flow, fields) -> None:
    """Tab: sang ô nhập kế tiếp của màn (vòng lại ô đầu)."""
    order = [name for name in FIELD_ORDER if name in fields]
    if not order:
        return
    if flow.focus in order:
        flow.focus = order[(order.index(flow.focus) + 1) % len(order)]
    else:
        flow.focus = order[0]


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


def toggle_item(flow: V3Flow, item: str) -> None:
    """Gắn/bỏ một vật phẩm. Hết ô thì báo, không im lặng bỏ qua."""
    if item in flow.items:
        flow.items.remove(item)
        return
    if len(flow.items) >= flow.slots():
        say(flow, f"All {flow.slots()} item slots are full.")
        return
    flow.items.append(item)


def remove_slot(flow: V3Flow, index: int) -> None:
    """Bấm vào một ô đã có vật phẩm = bỏ vật phẩm đó ra."""
    if 0 <= index < len(flow.items):
        del flow.items[index]
