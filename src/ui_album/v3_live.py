"""
U3 — dựng sẵn các trạng thái SỐNG để chụp ảnh cho luận văn.

Khác ``--ui-shots`` ở chỗ: lượt chụp kia vẽ giá trị ``sample`` của bản vẽ, còn
ở đây app chạy bằng **đường giá trị sống** (``v3_values.state_for``) — đúng
cái mà người chơi thấy. Nhờ vậy ảnh trong luận văn chứng minh được là giao
diện thật sự đọc trạng thái, không phải tô lại ảnh thiết kế.

Con số trong các preset dưới đây là **dữ liệu dựng cảnh để chụp**, không phải
kết quả đo — y như ``values.SHOT_MATCH`` của bộ asset cũ, và báo cáo phải nói
rõ. Riêng phần ván chơi thì đi qua ``AlbumMatch`` thật, nên điểm, phần trăm
ảnh bị gặm và các dòng kết quả vẫn do đúng công thức của trò chơi tính ra.
"""

from __future__ import annotations

import logging
import time

from src.ui_album import v3_flow as F
from src.game.match import AlbumMatch

logger = logging.getLogger(__name__)

#: Ảnh cuối chụp **cả màn hình** chứ không đọc framebuffer: nó phải cho thấy
#: cửa sổ nằm giữa desktop, tức là thứ duy nhất chứng minh chế độ Window là
#: cửa sổ thật có viền chứ không phải toàn màn hình vẽ nhỏ lại.
DESKTOP = "window_on_desktop"
GRAB_SIZE = (1280, 720)

#: Thứ tự chụp. Tên này cũng là tên file PNG trong ``results/ui_v3_1107/live``.
ORDER = ("settings_window", "level_select_1", "level_select_5", "loadout",
         "tilt_done", "loading_50", "hud_level_1", "hud_level_5",
         "result_win", "result_lose", "pvp_result", DESKTOP)


class _PropsStub:
    """Đủ cho ``AlbumMatch``: nó chỉ đọc ``props`` và ``found``."""

    def __init__(self, total: int, found: int) -> None:
        self.props = [{"pos_mm": (0.0, 0.0, 0.0)} for _ in range(total)]
        self.found = [index < found for index in range(total)]


def _match(flow, *, found: int, seconds_left: float, hp: float,
           stamina: float, used=None, effects=(), rounds=(9, 10),
           shots=()) -> AlbumMatch:
    """Một ván dựng sẵn, chạy bằng đúng lớp ``AlbumMatch`` của trò chơi."""
    total = flow.bugs()
    match = AlbumMatch(level=flow.level, state=_PropsStub(total, found),
                       started_s=time.perf_counter()
                       - (flow.seconds() - seconds_left),
                       weapon_name=flow.weapon_name() or F.WEAPONS[0],
                       items=list(flow.items), used=dict(used or {}),
                       hp=hp, stamina=stamina, weights=F.ITEM_WEIGHT)
    match.effects = list(effects)
    # ``rounds`` nay la thuoc tinh tinh tu bang dan, khong gan thang duoc.
    match.rounds_left, match.mag_size = int(rounds[0]), int(rounds[1])
    # Số lớp lấy của ảnh ĐANG mở, không đặt cứng: ảnh chụp luận văn mà in
    # một con số bịa thì nó phản lại chính điều đang muốn chứng minh.
    match.layer_total = int(flow.layer_count or 0)
    match.layer_index = 1
    # Vết đạn: ``age`` là tuổi tính bằng giây, nên ảnh chụp bắt được chúng ở
    # các độ mờ khác nhau — cho thấy chúng tàn dần chứ không đứng im.
    now = time.perf_counter()
    match.shots = [(x, y, now - age, hit) for x, y, age, hit in shots]
    return match


def _base(flow) -> None:
    """Nền chung: đã đăng nhập, có ảnh, có vũ khí và hai vật phẩm."""
    flow.player = flow.player or "anhkhoa"
    flow.weapon = 0
    flow.items = ["Antidote", "Flare"]


def apply(name: str, flow, window) -> AlbumMatch | None:
    """
    Đặt trạng thái của preset ``name``. Trả ván chơi (hoặc ``None``).

    ``window`` cần cho preset duy nhất đụng tới cửa sổ thật (chế độ Window).
    """
    _base(flow)
    if name == "settings_window":
        flow.go(F.SETTINGS)
        flow.volume = 70
        if window is not None and window.display_mode != "window":
            window.set_display_mode("window")
        flow.display = "window"
    elif name.startswith("level_select_"):
        flow.level = int(name.rsplit("_", 1)[1])
        flow.go(F.LEVELS)
    elif name == "loadout":
        flow.level = 3
        flow.go(F.LOADOUT)
    elif name == "tilt_done":
        flow.level = 3
        flow.go(F.TILT)
        # Ba dấu kiểm đã xong: màn này chờ đầu người thật, mà lượt chụp tự
        # động thì không có ai nghiêng đầu.
        flow.face_seen = True
        flow.tilt_left_done = flow.tilt_right_done = True
        flow.tilt_deg = 11.0
    elif name == "loading_50":
        flow.go(F.LOADING)
        flow.pipeline_pct = 50.0
    elif name.startswith("hud_level_"):
        flow.level = int(name.rsplit("_", 1)[1])
        flow.go(F.PLAY)
        flow.face_seen = True
        marks = ((470.0, 300.0, 0.15, True), (640.0, 420.0, 0.55, False),
                 (820.0, 250.0, 0.9, False))
        if flow.level == 5:
            return _match(flow, found=7, seconds_left=72.0, hp=72.0,
                          stamina=48.0, effects=[("poison", 6.0)], shots=marks)
        return _match(flow, found=4, seconds_left=95.0, hp=86.0, stamina=100.0,
                      shots=marks)
    elif name == "result_win":
        flow.level = 3
        flow.last_result = {"score": 37, "found": flow.bugs(),
                            "total": flow.bugs(), "time_left": 42,
                            "used": {"Antidote": 1}, "overtime_s": 0.0}
        flow.go(F.WIN)
    elif name == "result_lose":
        flow.level = 5
        flow.last_result = {"score": -38, "found": 3, "total": flow.bugs(),
                            "time_left": 0, "used": {"Energy drink": 1},
                            "overtime_s": 7.0}
        flow.go(F.LOSE)
    elif name == DESKTOP:
        flow.level = 3
        flow.go(F.PLAY)
        flow.face_seen = True
        if window is not None:
            if window.display_mode != "window":
                window.set_display_mode("window")
            window.resize_window(*GRAB_SIZE)
        return _match(flow, found=4, seconds_left=95.0, hp=86.0, stamina=100.0)
    elif name == "pvp_result":
        flow.level = 3
        flow.mode = F.PVP_MODES[0]
        flow.last_result = {
            "score": 31, "opponent_score": 10, "winner": "you",
            "opp_name": "Player 2",
            "rows": [["Bugs killed", "3 × 10"], ["Time left", 42],
                     ["Antidote", 1], ["Flare", 2]],
            "opponent_rows": [["Bugs killed", "1 × 10"], ["Time left", 10],
                              ["Thermal scan", 1]],
        }
        flow.go(F.PVP_RESULT)
    else:
        logger.warning("UI v3 live: khong biet preset %s", name)
    return None
