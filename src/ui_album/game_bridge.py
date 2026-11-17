"""
Nối giao diện album với ván chơi thật (``src/content/prop_game.py``).

Vật ẩn = **bọ**. Ở Phase 1 giữ nguyên cơ chế cũ: một con bọ tính là diệt khi
đã lộ liên tục ``FOUND_HOLD_S``. Lớp này chỉ thêm phần giao diện cần mà ván
chơi cũ không có: đồng hồ đếm ngược theo cấp, máu, stamina, điểm.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import NamedTuple

from src.ui_album.flow import ITEM_WEIGHT, LEVEL_TABLE

logger = logging.getLogger(__name__)

#: Quá giờ ở cấp Châu Á: −5 điểm mỗi giây, thua sau ngần này giây.
OVERTIME_MAX_S = 30.0
OVERTIME_PENALTY = 5.0


class WeaponSpec(NamedTuple):
    """Băng đạn, nhịp bắn và thời gian nạp của một khẩu."""

    mag: int
    fire_s: float        # giây tối thiểu giữa hai phát
    reload_s: float


#: Ba khẩu khác nhau ở cả ba con số, đúng lời mô tả in trên màn trang bị:
#: "Precision gun — small reticle, few rounds. Pick off hidden bugs one by
#: one" → băng nhỏ, bắn chậm, nạp nhanh. "Heavy-mag gun — big magazine.
#: Missed shots cost you less" → băng to, bắn nhanh, đổi lại nạp lâu.
#: "Spread gun — one shot hits several bugs standing close together" → nằm
#: giữa, băng nhỏ nhất vì mỗi phát ăn nhiều con.
#: Con số chọn theo **tỉ lệ thời gian đứng nạp**, không chọn cảm tính. Bộ
#: đầu (10/0,55/1,4 · 6/0,40/1,6 · 16/0,18/2,2) cho ra 20 % · 40 % · 43 % thời
#: gian chết — đo trên một phiên chơi thật: 59 phát ra, 45 cú bấm bị nuốt.
#: Gần một nửa số lần bấm không ra đạn là hỏng, nhất là ở buổi bảo vệ. Bộ
#: dưới đây giữ nguyên tính cách ba khẩu mà kéo tỉ lệ ấy về 15 % · 19 % · 23 %.
WEAPONS_SPEC = {
    "Precision gun": WeaponSpec(12, 0.50, 1.1),
    "Spread gun": WeaponSpec(8, 0.38, 0.9),
    "Heavy-mag gun": WeaponSpec(24, 0.18, 1.3),
}
DEFAULT_SPEC = WeaponSpec(12, 0.45, 1.1)

#: Giữ hai tên cũ cho chỗ nào còn tra băng đạn một mình.
WEAPON_MAG = {name: spec.mag for name, spec in WEAPONS_SPEC.items()}
DEFAULT_MAG = DEFAULT_SPEC.mag
RELOAD_S = DEFAULT_SPEC.reload_s


@dataclass
class AlbumMatch:
    """Một ván của giao diện album; bọc ``GameState`` của trò chơi cũ."""

    level: int
    state: object                      # prop_game.GameState
    started_s: float
    weapon_name: str = "Súng chính xác"
    ammo: str = "10/10"
    items: list = field(default_factory=list)
    used: dict = field(default_factory=dict)
    hp: float = 100.0
    stamina: float = 100.0
    head_ok: bool = True
    layer_label: str = "– / 3"
    effects: list = field(default_factory=list)
    missed_until_s: float = 0.0
    #: TASK_ui_v3_1107: bảng trọng số vật phẩm. Bộ album đánh khoá tiếng Việt,
    #: bộ v3 tiếng Anh; ``score()`` phải tra đúng bảng của giao diện đang chạy,
    #: nếu không mọi vật phẩm tính trọng số 0 và điểm luôn bằng số giây còn lại.
    weights: dict = field(default_factory=lambda: dict(ITEM_WEIGHT))
    #: Lớp đang xét và tổng số lớp THẬT của ảnh (HUD v3 ``layer``/``layer_pips``).
    layer_index: int = 1
    layer_total: int = 0
    #: Băng đạn. ``rounds_left < 0`` = chưa nạp lần đầu (xem ``__post_init__``).
    mag_size: int = 10
    rounds_left: int = -1
    reload_until_s: float = 0.0
    #: Nhịp bắn: giây tối thiểu giữa hai phát. 0 = bắn nhanh tuỳ tay.
    fire_interval_s: float = 0.0
    reload_s: float = RELOAD_S
    next_shot_s: float = 0.0
    #: Bật đếm đạn. CHỈ giao diện v3 bật; bộ album giữ nguyên lối bắn không
    #: giới hạn của nó, vì nó là đường lùi cho buổi bảo vệ.
    ammo_enabled: bool = False
    #: Vết đạn gần nhất: ``(x, y, mốc thời gian, trúng?)`` toạ độ thiết kế.
    shots: list = field(default_factory=list)
    #: TASK_pvp_lan_1116: trong PvP thật, ba ô này đổ dữ liệu ĐỐI THỦ lấy từ
    #: thông điệp server (``killed``/``used_item``/``eye``) — tên giữ
    #: nguyên "bot_*" dù không còn là bot, để khỏi phải sửa v3_values.py
    #: (nó chỉ ``getattr(game, "bot_found", 0)``). Ván solo không bao giờ
    #: đụng tới ba ô này.
    bot_found: int = 0
    bot_items: int = 0
    exposure_pct: float = 0.0
    #: Số bọ CHÍNH TÔI hạ, xác nhận từ server (``killed.by`` khớp tên mình)
    #: — khác ``bugs_found``/``state.found`` (đếm CẢ hai bên, vì bọ chết là
    #: chết trên MÀN của cả hai, bất kể ai hạ).
    my_kills: int = 0
    #: Mốc phát bắn gần nhất và chỗ bắn — để vẽ chớp đầu nòng/vệt đạn.
    shot_at_s: float = 0.0
    shot_point: tuple = (640.0, 360.0)
    ended: str = ""                    # "", "win", "lose"

    def __post_init__(self) -> None:
        if self.rounds_left < 0:
            self.rounds_left = int(self.mag_size)

    @property
    def rounds(self) -> tuple:
        """``(còn, tối đa)`` — HUD v3 vẽ cả con số lẫn dãy pip."""
        return (max(0, int(self.rounds_left)), int(self.mag_size))

    def reloading(self, now: float) -> bool:
        return now < self.reload_until_s

    def ready_to_fire(self, now: float) -> bool:
        """Đã qua nhịp bắn chưa (khác với 'đang nạp')."""
        return now >= self.next_shot_s

    def take_round(self, now: float) -> bool:
        """
        Trừ một viên. ``False`` = không bắn được.

        Ba lý do không bắn được, và chúng KHÁC nhau với người chơi: đang nạp
        (có báo), chưa tới nhịp bắn (im lặng — bấm nhanh hơn súng là chuyện
        thường), và hết băng (chuyển sang nạp).

        Hết viên cuối thì tự nạp: không có đường nạp thì bắn hết băng là ván
        tắc, mà bản vẽ không có nút nạp đạn nào.
        """
        if self.reloading(now) or not self.ready_to_fire(now):
            return False
        if self.rounds_left <= 0:
            self.reload_until_s = now + self.reload_s
            return False
        self.rounds_left -= 1
        self.next_shot_s = now + self.fire_interval_s
        if self.rounds_left == 0:
            self.reload_until_s = now + self.reload_s
        return True

    def tick_reload(self, now: float) -> None:
        if self.reload_until_s and now >= self.reload_until_s:
            self.reload_until_s = 0.0
            self.rounds_left = int(self.mag_size)

    @property
    def seconds(self) -> float:
        return float(LEVEL_TABLE[self.level][2])

    @property
    def bugs_total(self) -> int:
        return len(self.state.props)

    @property
    def bugs_found(self) -> int:
        return int(sum(self.state.found))

    @property
    def elapsed_s(self) -> float:
        return time.perf_counter() - self.started_s

    @property
    def time_left_s(self) -> float:
        return self.seconds - self.elapsed_s

    @property
    def overtime_s(self) -> float:
        return max(0.0, -self.time_left_s)

    @property
    def eaten_pct(self) -> float:
        """% ảnh đã bị gặm: phần bọ còn lại × phần thời gian đã trôi."""
        if not self.bugs_total:
            return 0.0
        left = 1.0 - self.bugs_found / self.bugs_total
        return 100.0 * left * min(1.0, self.elapsed_s / max(1.0, self.seconds))

    def score(self) -> int:
        """Điểm = giây còn lại − Σ(trọng số vật phẩm × số lần dùng)."""
        table = self.weights or ITEM_WEIGHT
        spent = sum(table.get(name, 0) * count
                    for name, count in self.used.items())
        base = max(0.0, self.time_left_s) - spent
        if self.level == 5:
            base -= OVERTIME_PENALTY * self.overtime_s
        return int(round(base))

    def tick(self, head_ok: bool, layer_label: str = "") -> str:
        """Nhịp mỗi khung; trả ``""``/``"win"``/``"lose"``."""
        self.tick_reload(time.perf_counter())
        self.head_ok = bool(head_ok)
        if layer_label:
            self.layer_label = layer_label
        if self.ended:
            return self.ended
        if self.bugs_total and self.bugs_found >= self.bugs_total:
            self.ended = "win"
        elif self.hp <= 0.0:
            self.ended = "lose"
        elif self.time_left_s <= 0.0:
            # Cấp 1–4 hết giờ là thua; Châu Á chơi tiếp, thua sau 30 s quá giờ.
            if self.level != 5 or self.overtime_s >= OVERTIME_MAX_S:
                self.ended = "lose"
        if self.ended:
            logger.info("Album match: %s — cap %d, %d/%d bo, diem %d",
                        self.ended, self.level, self.bugs_found,
                        self.bugs_total, self.score())
        return self.ended


MISS_STAMINA = 5.0
MISS_SHOW_S = 0.8

#: Giữ lại mấy vết đạn gần nhất để người chơi thấy mình vừa bắn vào đâu.
SHOT_MARKS = 6


@dataclass(frozen=True)
class AimContext:
    """
    Tất cả những gì cần để biết một phát bắn găm vào đâu (F1).

    Gom thành một bó chứ không truyền rời: ``fire`` cần năm thứ này cùng lúc,
    và F2 còn bắn nhiều viên một phát — mỗi viên phải dùng **đúng** một bộ.

    ``bugs`` là ``{chỉ số: (pos_mm, radius_mm, size_factor, SpriteMask)}`` của
    MỌI con **chưa diệt** (TASK_hit_0200: không lọc ``is_exposed`` nữa — tập
    này phải khớp đúng tập ``BugPainter.draw`` vẽ ra).

    ``is_exposed`` là ``{chỉ số: bool}`` — CHỈ để ghi log (bước 2), không ai
    dùng nó để lọc ``bugs`` hay quyết định trúng/hụt nữa.
    """

    eye_mm: tuple
    bugs: dict
    planes: tuple = ()
    glass_mm: tuple = (344.0, 194.0)
    centre_mm: tuple = (0.0, 0.0)
    alpha_threshold: float | None = None
    is_exposed: dict = field(default_factory=dict)

    def resolve(self, point) -> tuple:
        """``(chỉ số, chí mạng)`` của con bọ mà tâm ngắm ``point`` găm vào."""
        from src.ui_album.hit import pick_target

        return pick_target(self.bugs, point, self.eye_mm, self.planes,
                           self.glass_mm, self.centre_mm,
                           alpha_threshold=self.alpha_threshold)


def fire(match: AlbumMatch, crosshair, aim: AimContext) -> tuple | None:
    """
    Một phát bắn. Trả ``(chỉ số con bọ, chí mạng)``; ``(-1, False)`` nếu hụt,
    ``None`` nếu **không bắn được** (hết băng hoặc đang nạp).

    F1: hộp trúng là hình bóng THẬT của sprite (alpha > 0,25, nới 4 px), và
    tia ngắm gặp lớp gần hơn thì phát đó găm vào lớp — hụt. Vòng tròn 40 px
    quanh tâm chiếu đã bỏ: sprite vẽ ở ``radius × 3`` nên phần lớn thân con bọ
    nằm ngoài vòng ấy, bắn vào chân luôn báo hụt.
    """
    now = time.perf_counter()
    if match.ammo_enabled and not match.take_round(now):
        # Tách hai lý do: "đang nạp" là thời gian CHẾT (đáng lo, chỉnh được
        # bằng WEAPONS_SPEC), còn "chưa tới nhịp" chỉ là bấm nhanh hơn súng —
        # chuyện thường. Gộp một dòng thì không biết số nào đang lớn.
        logger.info("Bam co dan: %s",
                    "dang nap" if match.reloading(now) else "chua toi nhip")
        return None
    from src.ui_album.hit import shot_diagnostic

    logger.info("Ban chan doan: %s", shot_diagnostic(
        aim.bugs, crosshair, aim.eye_mm, aim.planes, aim.glass_mm,
        aim.centre_mm, alpha_threshold=aim.alpha_threshold,
        is_exposed=aim.is_exposed))
    best, crit = aim.resolve(crosshair)
    match.shot_at_s = now
    match.shot_point = tuple(crosshair)
    # Vết đạn: giữ vài phát gần nhất, cũ nhất rụng trước.
    #
    # TASK_game_feel_1118 P6: this is the ONE scene-anchored overlay
    # (besides the Sakuga hit/miss fx, spawned the same way — see
    # ``v3_ui.py``'s ``spawn_hit_fx``/``spawn_miss_fx``) — and it needs NO
    # ``s = c + z(g - c)`` transform while scoped. It's always stored at
    # ``crosshair`` itself (the design-px point the scope's own Kooima
    # sub-window is centred on), and that point is a FIXED POINT of the
    # warp: plugging ``g = c`` into the formula gives ``s = c`` for ANY
    # zoom. A shot mark at the firing crosshair is correctly placed at
    # every later frame — scoped or not, whatever the zoom is then —
    # without tracking a separate 3D position at all.
    match.shots.append((float(crosshair[0]), float(crosshair[1]), now,
                        best >= 0))
    del match.shots[:-SHOT_MARKS]
    if best < 0:
        match.missed_until_s = now + MISS_SHOW_S
        if match.level == 5:
            match.stamina = max(0.0, match.stamina - MISS_STAMINA)
        logger.info("Ban hut tai (%.0f, %.0f)", crosshair[0], crosshair[1])
        return (-1, False)
    match.state.found[best] = True
    logger.info("Ban trung bo #%d%s", best, " CHI MANG" if crit else "")
    return (best, crit)


def start_match(level: int, state, weapon_name: str, items,
                weights: dict | None = None) -> AlbumMatch:
    """Mở một ván mới cho cấp ``level`` với ván chơi ``state`` vừa đặt bọ."""
    match = AlbumMatch(level=int(level), state=state,
                       started_s=time.perf_counter(),
                       weapon_name=str(weapon_name), items=list(items),
                       weights=dict(weights or ITEM_WEIGHT))
    logger.info("Album match bat dau: cap %d (%s), %d bo, %d o vat pham",
                match.level, LEVEL_TABLE[match.level][0], match.bugs_total,
                LEVEL_TABLE[match.level][3])
    return match
