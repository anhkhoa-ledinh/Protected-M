"""
Giao diện v3 ("v2 · Redesign"): nạp ``assets/ui_v3/manifest.json`` và vẽ.

Thứ tự vẽ là **hợp đồng** của bộ asset (README mục "Draw order"):
``base`` → ``photos`` → ``top`` → ``bars`` → ``pips`` → ``pills`` →
``sprites`` → ``texts`` → hover/focus. ``assets/ui_v3/tools/preview.py`` là
bản tham chiếu bằng PIL của đúng thứ tự ấy, và ``expected/*.jpg`` là đầu ra
của nó; mọi phép tính chữ/thanh/viên ở đây **chép theo** preview.py, không tự
nghĩ lại — lệch một pixel là trượt ngưỡng 24 của ``tools/compare.py``.

Khác với lớp vẽ cũ (``draw.py``), module này không làm việc PIL nào trong
đường chạy mỗi khung: chữ, viên thuốc, đầu thanh bo góc và pip đều dựng một
lần rồi nằm trong bộ đệm texture (luật F4). ``base``, ``top`` và sprite của
một màn nạp **một lần cho mỗi màn**.
"""

from __future__ import annotations

import json
import logging
import math
from collections import OrderedDict
from pathlib import Path

import numpy as np
from OpenGL import GL
from PIL import Image, ImageDraw

from src.renderer.shader import ShaderProgram
from src.ui_album.fonts import FontBook
from src.ui_album.manifest import letterbox
from src.ui_album.scope import (RETICLE_ARM_PX, RETICLE_COL, RETICLE_GAP_PX,
                                RETICLE_THICK_PX, VIGNETTE_ALPHA,
                                VIGNETTE_COL, VIGNETTE_RADIUS_PX,
                                VIGNETTE_SOFT_PX, ZOOM_SETTLED_EPS,
                                screen_scope_transform)
from src.ui_album.shot_feel import (CRIT_IMPACT_COL,
                                    HITMARK_ANGLES_DEG, HITMARK_COL,
                                    HITMARK_CRIT_COL, HITMARK_CRIT_SCALE,
                                    HITMARK_INNER_PX, HITMARK_LIFE_S,
                                    HITMARK_OUTER_PX, HITMARK_OUTLINE_COL,
                                    HITMARK_OUTLINE_PX, HITMARK_WIDTH_PX,
                                    TRACER_CORE_COL, TRACER_CORE_FRACTION,
                                    TRACER_EDGE_COL, TRACER_LIFE_S,
                                    TRACER_TARGET_WIDTH_PX)
from src.ui_album.textures import TextureCache, load_rgba, upload_rgba

logger = logging.getLogger(__name__)

#: Thư mục bộ asset v3, tính từ gốc project.
ASSET_DIR = Path("assets") / "ui_v3"

#: Không gian thiết kế của bộ v3. Mọi ``rect`` trong manifest nằm trong hệ này.
DESIGN_SIZE = (1280.0, 720.0)

#: Màu chữ mặc định khi ``runs[].col`` rỗng (preview.py dùng đúng hằng này).
INK = "#24415F"

#: Lề quanh ảnh chữ. Glyph có thể tràn ra ngoài bút (bearing âm, dấu cao hơn
#: ascent); không chừa lề thì chữ bị cắt mép và điểm compare.py nhảy vọt.
TEXT_PAD = 8

#: Vùng vẽ 3D và webcam ở **chế độ chụp** — README mục "Shot mode".
SHOT_VIEWPORT_RGB = (60, 70, 80)
SHOT_CAMERA_RGB = (90, 90, 100)

#: Vòng hover của hộp xác nhận chia sẻ (``_draw_confirm_dialog`` — hộp
#: thoại không đi qua ``v3_hover.py``, chỉ hai nút Yes/No của nó).
HOVER_PX = 2.0
#: Vòng focus bàn phím: 2,5 px, lùi ra 3 px (``v3_hover.py`` dùng cả hai).
FOCUS_PX, FOCUS_INSET = 2.5, 3.0
#: TASK_bug_art_0530: bán kính bo góc của field_user/field_pass, đo trực tiếp
#: trên ``v3_01a_sign_in/base.png`` (quét cột đầu mép bo ở mỗi hàng, bán kính
#: = khoảng cách từ mép tới điểm offset về 0 — ra ~22 px ở ảnh ×1,5, tức 15 px
#: không gian thiết kế). Không có trường bán kính nào trong manifest.
FIELD_CORNER_RADIUS_PX = 15.0

BTN_CORNER_RADIUS_PX = 12.0
ROW_CORNER_RADIUS_PX = 18.0

#: TASK_ui_sakuga_1117 Bug B — ``_ring_radius`` trả 0 (khung vuông) cho MỌI
#: hotspot ngoài năm tiền tố trên, kể cả những cái CÓ bo góc thật trên bản vẽ
#: (viên nhộng ``tab_Solo``, đoạn bo ``display_full``) hoặc là chữ/liên kết
#: (``link_guest`` — khung vuông kéo dài hết cả thẻ). Thay bằng một bảng
#: DỨT ĐIỂM, phủ hết mọi hotspot trong bản kê, không rơi về 0 một cách im
#: lặng — tên lạ thì ``hotspot_shape`` ném lỗi ngay.
#:
#: Bán kính đo trực tiếp trên ``base.png`` bằng ``measure_corner_radius_px``
#: (quét hàng y=0 của rect, giải phương trình đường tròn (x-r)²+(y-r)²=r²
#: — xem hàm đó). ``sticker_``/``slot_`` không có thẻ nền trong ``base.png``
#: (vẽ bằng sprite PNG riêng lúc chạy — xem ``item_tile:*``); đo trên chính
#: sprite PNG đó (``measure_alpha_radius``) thay vì base.png.
#:
#: Số đo (thiết kế px, làm tròn — báo cáo P2 có số thật):
#: - ``btn_``: 13,3–15,7 tại các hàng thử — giữ 12 (giá trị ĐÃ DÙNG, không
#:   lệch quá ±2-3 px so với số đo mới, và nhiều nút đã khớp gần đúng ~12-13).
#: - ``level_row_``/``weapon_``: đo ĐÚNG 18.0 — khớp hằng cũ.
#: - ``field_``: đo 14,67 — khớp hằng cũ (15) trong 0,33 px.
#: - ``sticker_``: ~26 (sprite PNG, rìa mờ vì có đổ bóng).
#: - ``slot_``: ~25 (sprite PNG, cùng lý do).
#: - ``mode_``: ~24,7.
#: - ``dropzone``: đo ĐÚNG 18.0.
#: - ``recent_memory``: ~13,3.
#: - ``dropdown_``: ~9,3.
STICKER_CORNER_RADIUS_PX = 26.0
SLOT_CORNER_RADIUS_PX = 25.0
MODE_CORNER_RADIUS_PX = 25.0
DROPZONE_CORNER_RADIUS_PX = 18.0
RECENT_MEMORY_CORNER_RADIUS_PX = 13.0
DROPDOWN_CORNER_RADIUS_PX = 9.0


class HotspotShape:
    """Hình vòng hover/focus của một hotspot. ``kind`` là một trong
    ``rounded|pill|circle|underline|thumb|none``; ``radius`` chỉ có nghĩa
    với ``rounded`` (``pill``/``circle`` tự tính từ rect lúc vẽ)."""

    __slots__ = ("kind", "radius")

    def __init__(self, kind: str, radius: float = 0.0) -> None:
        self.kind = kind
        self.radius = radius

    def __repr__(self) -> str:
        return f"HotspotShape({self.kind!r}, {self.radius!r})"

    def __eq__(self, other) -> bool:
        return (isinstance(other, HotspotShape) and self.kind == other.kind
                and self.radius == other.radius)


def _rounded(radius: float) -> HotspotShape:
    return HotspotShape("rounded", radius)


#: Tên CHÍNH XÁC (không phải tiền tố) — mỗi tên chỉ có một hotspot, không có
#: hậu tố thay đổi (khác ``btn_Sign in``/``btn_Create album``/...).
_EXACT_SHAPES: dict[str, HotspotShape] = {
    "continue": HotspotShape("none"),
    "dropzone": _rounded(DROPZONE_CORNER_RADIUS_PX),
    "recent_memory": _rounded(RECENT_MEMORY_CORNER_RADIUS_PX),
    "sign_out": HotspotShape("underline"),
    "leave": HotspotShape("underline"),
    "slider": HotspotShape("thumb"),
    "footer_link": HotspotShape("underline"),
    "icon_show_password": HotspotShape("circle"),
}

#: Tiền tố — nhiều hotspot cùng kiểu, khác hậu tố (``btn_Sign in``,
#: ``level_row_1``, ``mode_Competitive``...). Thứ tự không quan trọng: mỗi
#: tên chỉ khớp đúng MỘT tiền tố trong bản kê hiện tại.
_PREFIX_SHAPES: tuple[tuple[str, HotspotShape], ...] = (
    ("btn_", _rounded(BTN_CORNER_RADIUS_PX)),
    ("level_row_", _rounded(ROW_CORNER_RADIUS_PX)),
    ("weapon_", _rounded(ROW_CORNER_RADIUS_PX)),
    ("field_", _rounded(FIELD_CORNER_RADIUS_PX)),
    ("sticker_", _rounded(STICKER_CORNER_RADIUS_PX)),
    ("slot_", _rounded(SLOT_CORNER_RADIUS_PX)),
    ("mode_", _rounded(MODE_CORNER_RADIUS_PX)),
    ("dropdown_", _rounded(DROPDOWN_CORNER_RADIUS_PX)),
    ("tab_", HotspotShape("pill")),
    ("seg_", HotspotShape("pill")),
    ("display_", HotspotShape("pill")),
    ("theme_", HotspotShape("pill")),
    ("switch_", HotspotShape("pill")),
    ("link_", HotspotShape("underline")),
)

#: Hotspot phủ từ ngần này phần trăm diện tích thiết kế (1280×720) trở lên
#: thì không vẽ vòng nào, bất kể tên — ``continue`` (cả màn) là ví dụ, nhưng
#: luật tính theo DIỆN TÍCH chứ không chỉ theo tên, phòng khi có hotspot
#: to khác sau này.
FULL_SCREEN_RING_AREA_FRACTION = 0.5


def hotspot_shape(name: str, rect) -> HotspotShape:
    """
    Hình vòng hover/focus của hotspot ``name`` — KHÔNG rơi về 0 một cách im
    lặng. Tên không khớp tiền tố/tên nào trong bảng thì NÉM LỖI ngay, để lộ
    ra sớm (hotspot mới trong bản kê mà quên thêm vào bảng) thay vì âm thầm
    vẽ khung vuông sai như lỗi cũ.
    """
    width, height = float(rect[2]), float(rect[3])
    if width * height >= FULL_SCREEN_RING_AREA_FRACTION * DESIGN_SIZE[0] * DESIGN_SIZE[1]:
        return HotspotShape("none")
    shape = _EXACT_SHAPES.get(name)
    if shape is not None:
        return shape
    for prefix, candidate in _PREFIX_SHAPES:
        if name.startswith(prefix):
            return candidate
    raise ValueError(
        f"hotspot_shape: khong co trong HOTSPOT_SHAPES: {name!r} "
        "(them vao _EXACT_SHAPES/_PREFIX_SHAPES trong src/ui_album/v3.py)")


def ring_draw_radius(shape: HotspotShape, rect) -> float | None:
    """
    Bán kính để vẽ vòng NGAY BÂY GIỜ (Phase 2) — ``rounded``/``pill``/
    ``circle`` vẫn vẽ khung bo tròn như hôm nay, chỉ đúng bán kính hơn.
    ``underline``/``thumb``/``none`` trả ``None`` (không vẽ khung vuông sai
    nữa); hình vẽ RIÊNG của chúng (gạch dưới, quầng tròn con trượt...) là
    việc của Phase 5 (``src/ui_album/v3_hover.py``).
    """
    if shape.kind == "rounded":
        return shape.radius
    if shape.kind == "pill":
        return float(rect[3]) * 0.5
    if shape.kind == "circle":
        return min(float(rect[2]), float(rect[3])) * 0.5
    return None


def measure_corner_radius_px(image, rect, image_scale: float,
                              probe_y: int = 0) -> float | None:
    """
    Bán kính bo góc trái-trên của ``rect`` (px thiết kế), đo trên ``image``
    (PIL, đã ``convert("RGB")``). ``None`` nếu không phát hiện được góc bo
    (hình vuông, hoặc không đủ dữ liệu để quét).

    Phương pháp: màu "trong" lấy ở CÙNG HÀNG với điểm quét nhưng giữa rect
    (tránh chữ/icon giữa tâm), màu "ngoài" lấy đúng điểm góc ``(rect.x,
    rect.y)`` (luôn ở ngoài mọi góc bo thật). Qua giữa hai màu đó là biên
    hình; tại hàng ``probe_y``, điểm biên ``(x, probe_y)`` nằm trên đường
    tròn tâm ``(r, r)`` bán kính ``r``, nên
    ``r² - 2r(x+y) + (x²+y²) = 0`` → ``r = (x+y) + sqrt(2xy)``
    (nghiệm dương; ``probe_y=0`` thì đơn giản còn ``r = x``).
    """
    px = image.load()
    ix, iy, iw, ih = (round(v * image_scale) for v in rect)
    outside = px[ix, iy]
    inside = px[ix + iw // 2, iy + probe_y]

    def dist(c1, c2) -> int:
        return sum(abs(a - b) for a, b in zip(c1, c2))

    max_scan = min(iw, ih) // 2 + 6
    x_at_y = None
    for dx in range(max_scan):
        if dist(px[ix + dx, iy + probe_y], inside) < dist(px[ix + dx, iy + probe_y], outside):
            x_at_y = dx
            break
    if x_at_y is None:
        return None
    x, y = float(x_at_y), float(probe_y)
    s = x + y
    disc = s * s - (x * x + y * y)
    if disc < 0:
        return None
    return (s + math.sqrt(disc)) / image_scale


def measure_alpha_radius_px(image, image_scale: float) -> float | None:
    """
    Bán kính bo góc trái-trên của ``image`` (PIL RGBA, đã CẮT ĐÚNG khung
    sprite — vd ``sprites/v3_05_loadout__sticker_Flare.png``), đo bằng kênh
    alpha thay vì màu: dùng cho sprite vẽ lúc chạy, không nằm trong
    ``base.png`` (``sticker_``/``slot_`` — xem ghi chú trên
    ``STICKER_CORNER_RADIUS_PX``).

    Các sprite này có viền đổ bóng mờ quanh thẻ, nên quét hàng ĐẦU TIÊN có
    alpha ≥ 250 ở đâu đó (chắc chắn là thẻ đặc, không phải quầng bóng mờ)
    làm hàng ``y=0`` cục bộ, rồi lấy điểm đầu tiên alpha > 127 trên hàng đó
    làm bán kính (công thức rút gọn ở ``probe_y=0``: ``r = x``).
    """
    px = image.load()
    w, h = image.size
    local_y0 = None
    for y in range(h):
        if any(px[x, y][3] >= 250 for x in range(w)):
            local_y0 = y
            break
    if local_y0 is None:
        return None
    for x in range(w):
        if px[x, local_y0][3] > 127:
            return x / image_scale
    return None


def measure_ink_bounds(image, rect, image_scale: float, threshold: int = 150):
    """
    Hộp bao các điểm ảnh KHÔNG PHẢI nền trong ``rect`` (px thiết kế tuyệt
    đối, cùng hệ toạ độ với ``rect``) — dùng để đo đúng bề rộng CHỮ của một
    liên kết, thay vì cả ô bấm rộng hơn nhiều (Bug C).

    Nền lấy theo màu PHỔ BIẾN NHẤT trong ``rect`` (chữ luôn chiếm ít diện
    tích hơn nền). Ngưỡng 150 (tổng |hiệu| ba kênh RGB) cố ý cao — biên khử
    răng cưa giữa nền-thẻ-trắng và nền-trang-xanh-nhạt chỉ lệch ~30-60, còn
    chữ (xanh đậm #24415F-ish) lệch nền trắng ~396; 150 tách rõ chữ thật
    khỏi nhiễu viền thẻ mà không cần biết trước màu chữ.

    ``None`` nếu không có điểm nào vượt ngưỡng (rect trống/toàn nền).
    """
    px = image.load()
    ix, iy, iw, ih = (round(v * image_scale) for v in rect)
    from collections import Counter

    counts = Counter(px[ix + x, iy + y] for y in range(ih) for x in range(iw))
    bg = counts.most_common(1)[0][0]

    def dist(c) -> int:
        return sum(abs(a - b) for a, b in zip(c, bg))

    min_x = min_y = max_x = max_y = None
    for y in range(ih):
        for x in range(iw):
            if dist(px[ix + x, iy + y]) > threshold:
                min_x = x if min_x is None else min(min_x, x)
                max_x = x if max_x is None else max(max_x, x)
                min_y = y if min_y is None else min(min_y, y)
                max_y = y if max_y is None else max(max_y, y)
    if min_x is None:
        return None
    x0 = rect[0] + min_x / image_scale
    y0 = rect[1] + min_y / image_scale
    x1 = rect[0] + (max_x + 1) / image_scale
    y1 = rect[1] + (max_y + 1) / image_scale
    return (x0, y0, x1 - x0, y1 - y0)


#: TASK_ui_sakuga_1117 Bug C — ``link_guest`` (và các ô liên kết khác) có hộp
#: bấm rộng bằng CẢ THẺ, nên bấm vào khoảng trắng cũng đăng nhập khách. Đè
#: hộp bấm bằng (biên mực CHỮ đo trên ``base.png`` + 8 px mỗi bên NGANG),
#: giữ nguyên y/chiều cao gốc của bản kê — lỗi chỉ ở BỀ RỘNG, không phải
#: chiều cao (``measure_ink_bounds`` cho số thật, xem báo cáo P2).
#: Khoá ``(màn, tên)`` vì ``footer_link`` hiện chữ khác nhau tuỳ màn.
LINK_HIT_RECTS: dict[tuple[str, str], tuple[float, float, float, float]] = {
    ("v3_01a_sign_in", "link_forgot"): (692.0, 404.0, 134.0, 17.0),
    ("v3_01c_sign_in_error", "link_forgot"): (692.0, 404.0, 134.0, 17.0),
    ("v3_01a_sign_in", "link_guest"): (591.0, 626.0, 98.0, 16.0),
    ("v3_01c_sign_in_error", "footer_link"): (593.0, 626.0, 96.0, 16.0),
    ("v3_01b_sign_up", "footer_link"): (537.0, 578.0, 206.0, 16.0),
    ("v3_02a_menu", "sign_out"): (1167.0, 49.0, 63.0, 15.0),
}


def hotspot_rect(screen: str, hotspot: dict):
    """``rect`` thật của ``hotspot`` — đè bởi ``LINK_HIT_RECTS`` nếu có
    (Bug C); mọi nơi đọc rect của hotspot (bắt click, vẽ vòng) phải đi qua
    đây, không đọc thẳng ``hotspot["rect"]``."""
    return LINK_HIT_RECTS.get((screen, hotspot["n"]), hotspot["rect"])

#: Tâm ngắm: một dấu ``+`` nhỏ, tính bằng pixel thiết kế.
#:
#: Bộ asset có sẵn sprite ``crosshair`` nhưng đó là vòng tròn 92 px — to gần
#: bằng một con bọ, che mất đúng chỗ người chơi đang soi. Dấu cộng vẽ bằng
#: bốn ô tô màu nên không tốn texture nào, và luôn nét ở mọi cỡ cửa sổ.
#: Viền sẫm vẽ dưới, nét sáng vẽ trên: ảnh nền có cả vùng tối lẫn vùng sáng,
#: một màu đơn sẽ biến mất ở một trong hai.
AIM_ARM = 7.0
AIM_THICK = 2.0
AIM_GAP = 2.0
AIM_EDGE = 1.0
AIM_COL, AIM_EDGE_COL = "#FFFFFF", "#24415F"

#: Bắn xong thì tâm ngắm **bung ra** và mờ đi, rồi khép lại theo nhịp súng.
#:
#: Đo trên một phiên chơi thật: 54 phát ra, 54 cú bấm bị chặn vì chưa tới
#: nhịp. Người chơi bấm nhanh gấp đôi súng, mà mỗi cú bị chặn thì màn hình
#: im lặng hoàn toàn — không tiếng (bản demo không có âm), không hình. Nhịp
#: bắn vô hình thì cảm giác như app đớp mất cú bấm. Cho tâm ngắm khép dần
#: lại là nói được "súng đang hồi" mà không phải đổi một con số cân bằng nào.
AIM_RECOIL = 6.0
AIM_DIM = 0.45

#: Lỗ đạn trên ảnh. Người chơi đang bắn vào một tấm ảnh, nên dấu để lại là
#: một cái **lỗ**, không phải chấm đỏ báo lỗi.
#:
#: Ba lớp tròn lồng nhau: quầng ngoài nhạt (mảnh vụn), vành sẫm, lõi gần
#: đen. Kích thước CỐ ĐỊNH nên cả vệt mờ dần chỉ tốn ba texture — xem
#: ``V3Painter.disc``.
HOLE_LIFE_S = 3.0
HOLE_FADE_S = 0.8               # chỉ mờ đi ở quãng cuối
HOLE_CORE, HOLE_RING, HOLE_DUST = 5.0, 9.0, 15.0
HOLE_CORE_COL, HOLE_RING_COL, HOLE_DUST_COL = "#141C28", "#2E3A4C", "#8DA3BA"

#: Trúng bọ thì thêm một chớp xanh ngắn đè lên lỗ — để biết là đã diệt được.
HIT_FLASH_S = 0.45
HIT_FLASH_SIZE = 26.0
HIT_FLASH_COL = "#3AA17A"

#: Dòng báo nổi ở giữa dưới (toast của giao diện).
TOAST_RECT = (340.0, 626.0, 600.0, 44.0)
TOAST_RADIUS = 22.0
TOAST_BG, TOAST_FG = "#24415F", "#FFFFFF"
TOAST_FONT = {"ff": "Nunito", "fw": 800, "fs": 15.0, "col": TOAST_FG}

#: Code-drawn album buttons, with fixed rectangles matching the manifest
#: hotspots. Share sits on the memory caption row; Random in the drop card.
CHOOSE_PHOTO_SCREEN = "v3_04_choose_photo"
SHARE_BTN_RECT = (944.0, 384.0, 228.0, 30.0)
RANDOM_BTN_RECT = (264.0, 478.0, 240.0, 52.0)
SHARE_BTN_FONT = {"ff": "Nunito", "fw": 800, "fs": 14.0, "ls": 0.0,
                  "col": "#3B72B4"}
#: Màu ``primary`` của manifest — hợp đồng dùng hex trực tiếp (xem ``CARET_COL``).
BUTTON_PRIMARY_COL = "#3B72B4"

#: Hộp xác nhận trước lượt chia sẻ đầu tiên của mỗi ảnh (P2.3) — chữ ĐÚNG
#: nguyên văn đề bài, không một ký tự khác.
CONFIRM_TEXT = ("This photo will be public: anyone with the app can "
               "download it. Do not share photos of people without their "
               "consent. Upload?")
CONFIRM_CARD_RECT = (380.0, 230.0, 520.0, 260.0)
CONFIRM_YES_RECT = (766.0, 422.0, 110.0, 44.0)
CONFIRM_NO_RECT = (640.0, 422.0, 110.0, 44.0)
CONFIRM_TEXT_FONT = {"ff": "Nunito", "fw": 700, "fs": 15.0, "ls": 0.0,
                     "col": INK}
CONFIRM_BTN_FONT = {"ff": "Nunito", "fw": 800, "fs": 15.0, "ls": 0.0}

#: Con trỏ nhập của ô đang chọn (ghi trong ``screens[].notes``).
CARET_SIZE = (2.0, 20.0)
CARET_GAP = 2.0
CARET_COL = "#3B72B4"


def rgba(colour: str | None, alpha: float = 1.0) -> tuple[float, float, float, float]:
    """``"#RRGGBB"`` → ``(r, g, b, a)`` 0–1. ``None`` trả màu chữ mặc định."""
    text = str(colour or INK).lstrip("#")
    if len(text) == 6:
        text += "FF"
    try:
        value = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6)]
    except ValueError:
        return (1.0, 0.0, 1.0, 1.0)             # màu báo lỗi, thấy ngay trên màn
    return (value[0], value[1], value[2], value[3] * float(alpha))


def _byte(colour: str | None, alpha: float = 1.0) -> tuple[int, int, int, int]:
    """Màu cho PIL: ``(r, g, b, a)`` 0–255."""
    return tuple(int(round(c * 255)) for c in rgba(colour, alpha))


class V3Manifest:
    """Bản kê v3 đã nạp. Chỉ đọc; không giữ trạng thái vẽ nào."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / ASSET_DIR
        path = self.root / "manifest.json"
        self.data = json.loads(path.read_text(encoding="utf-8"))
        version = int(self.data.get("version", 0))
        if version != 3:
            raise ValueError(f"{path}: can manifest version 3, doc duoc {version}")
        self.image_scale = float(self.data.get("image_scale", 1.5))
        self.design_size = tuple(self.data.get("design_size", DESIGN_SIZE))
        self.screens = self.data["screens"]
        self.colors = self.data.get("colors", {})
        # ``FontBook`` đọc đúng hai thuộc tính này của bản kê.
        self.fonts = self.data.get("fonts", {})
        self.sprites = {}
        missing = self.validate()
        logger.info("UI v3: %d man, %d sprite, thieu %d file",
                    len(self.screens), self.sprite_count(), len(missing))
        if missing:
            logger.warning("UI v3: thieu file %s", missing[:5])

    def screen_names(self) -> list[str]:
        """Tên 23 màn, theo thứ tự trong manifest."""
        return list(self.screens)

    def sprite_count(self) -> int:
        return sum(len(s.get("sprites", ())) for s in self.screens.values())

    def path(self, relative: str | None) -> Path | None:
        return None if not relative else self.root / relative

    def sample_photo(self) -> Path | None:
        return self.path(self.data.get("sample_photo"))

    def text(self, screen: str, key: str) -> dict | None:
        """Bản ghi text ``key`` của màn ``screen``; ``None`` nếu không có."""
        for record in self.screens.get(screen, {}).get("texts", ()):
            if record["key"] == key:
                return record
        return None

    def sprite(self, screen: str, key: str) -> dict | None:
        """Bản ghi sprite ``key`` của màn ``screen``; ``None`` nếu không có."""
        for record in self.screens.get(screen, {}).get("sprites", ()):
            if record["key"] == key:
                return record
        return None

    def validate(self) -> list[str]:
        """File khai trong manifest mà không có trên đĩa."""
        missing = []
        for name, screen in self.screens.items():
            for layer in ("base", "top"):
                relative = screen.get(layer)
                path = self.path(relative)
                if path is not None and not path.exists():
                    missing.append(relative)
            for record in screen.get("sprites", ()):
                path = self.path(record.get("file"))
                if path is not None and not path.exists():
                    missing.append(record["file"])
            _ = name
        return missing


class _PilCache:
    """
    LRU "dựng một lần bằng PIL, dùng lại mãi": khoá → (texture, số đo).

    Vì sao không dùng thẳng ``TextureCache.small``: chữ và viên thuốc cần
    **số đo** (bề rộng chuỗi, ascent) để đặt quad đúng chỗ, mà số đo ấy chỉ
    tính được lúc dựng. Trả kèm số đo nghĩa là đường chạy mỗi khung không gọi
    ``font.getlength`` lần nào nữa.
    """

    def __init__(self, limit: int = 768) -> None:
        self._store: OrderedDict[tuple, tuple] = OrderedDict()
        self._limit = int(limit)
        self.builds = 0

    def get(self, key: tuple, build):
        """``(texture, w_px, h_px, meta)``; gọi ``build()`` khi chưa có."""
        if key in self._store:
            self._store.move_to_end(key)
            return self._store[key]
        array, meta = build()
        if array is None:
            return None
        entry = (upload_rgba(array), array.shape[1], array.shape[0], meta)
        self._store[key] = entry
        self.builds += 1
        while len(self._store) > self._limit:
            _key, old = self._store.popitem(last=False)
            GL.glDeleteTextures(1, [old[0]])
        return entry

    def release(self) -> None:
        for entry in self._store.values():
            GL.glDeleteTextures(1, [entry[0]])
        self._store.clear()


class V3Painter:
    """
    Vẽ quad trong **không gian thiết kế** 1280×720; tự lo tỉ lệ và blend.

    ``set_frame`` nhận cỡ framebuffer THẬT mỗi khung, nên đổi cỡ cửa sổ (D1)
    không cần dựng lại gì ngoài những texture phụ thuộc tỉ lệ.
    """

    def __init__(self, manifest: V3Manifest, shader_dir: Path) -> None:
        self.manifest = manifest
        self.cache = TextureCache()
        self.pil = _PilCache()
        self.fonts = FontBook(manifest)
        self._shader = ShaderProgram(shader_dir / "album.vert",
                                     shader_dir / "album.frag")
        self._vao = self._vbo = 0
        self._fb = (1280, 720)
        self._box = letterbox(DESIGN_SIZE, self._fb)
        #: Texture của màn đang mở: ``(path, max_width)`` → ``(tex, w, h)``.
        self._screen_tex: dict[tuple, tuple[int, int, int]] = {}
        self._screen_key: tuple | None = None

    # ---- hạ tầng ---------------------------------------------------------
    def initialize(self) -> None:
        """Tạo VAO/VBO một lần, sau khi đã có context GL."""
        self._vao = GL.glGenVertexArrays(1)
        self._vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 96, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                 GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def set_frame(self, framebuffer) -> None:
        self._fb = (max(1, int(framebuffer[0])), max(1, int(framebuffer[1])))
        self._box = letterbox(DESIGN_SIZE, self._fb)

    @property
    def scale(self) -> float:
        """Pixel framebuffer trên mỗi pixel thiết kế."""
        return self._box.scale

    @property
    def k(self) -> float:
        """
        Tỉ lệ **của bộ asset** (``image_scale`` = 1,5), dùng cho chữ và viên.

        Vì sao không dựng chữ ở tỉ lệ framebuffer, là thứ trông có vẻ đúng
        hơn: FreeType làm tròn bước tiến của **từng** glyph về pixel nguyên,
        nên bề rộng một chuỗi KHÔNG tỉ lệ thuận với cỡ chữ. Dựng ở 1366/1280
        = 1,0667 thay vì 1,5 làm "anhkhoa" rộng thêm 5 %, dãy tám dấu chấm
        hẹp đi 11 %, và điểm ``compare.py`` nhảy từ ~3 lên 42. Dựng đúng tỉ lệ
        mà ``tools/preview.py`` đã dùng thì số đo khớp từng pixel, còn việc
        co giãn để GPU làm — y như nó đang làm với ``base`` và sprite, vốn
        cũng xuất ở 1,5×.
        """
        return float(self.manifest.image_scale)

    @property
    def box(self):
        return self._box

    def unmap(self, x: float, y: float) -> tuple[float, float]:
        """Pixel cửa sổ → toạ độ thiết kế (bắt chuột, U1.2)."""
        return self._box.unmap(x, y)

    def map(self, x: float, y: float) -> tuple[float, float]:
        """Toạ độ thiết kế → pixel cửa sổ — nghịch đảo của ``unmap``."""
        return self._box.map(x, y)

    def begin(self) -> None:
        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glDisable(GL.GL_CULL_FACE)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)

    def end(self) -> None:
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    def viewport_px(self, rect) -> tuple[int, int, int, int]:
        """``rect`` thiết kế → ``(x, y, w, h)`` pixel GL (gốc dưới-trái)."""
        x, y, w, h = self._box.rect(rect)
        return (int(round(x)), int(round(self._fb[1] - y - h)),
                max(1, int(round(w))), max(1, int(round(h))))

    def quad(self, rect, texture, tint=(1.0, 1.0, 1.0, 1.0),
             uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        """Một quad ở ``rect`` thiết kế, lấy vùng ``uv`` của texture."""
        if texture is None or rect[2] <= 0.0 or rect[3] <= 0.0:
            return
        x, y, w, h = self._box.rect(rect)
        fb_w, fb_h = self._fb
        x0, x1 = 2.0 * x / fb_w - 1.0, 2.0 * (x + w) / fb_w - 1.0
        y0, y1 = 1.0 - 2.0 * y / fb_h, 1.0 - 2.0 * (y + h) / fb_h
        u0, v0, u1, v1 = uv
        data = np.array([x0, y1, u0, v1, x1, y1, u1, v1,
                         x1, y0, u1, v0, x0, y1, u0, v1,
                         x1, y0, u1, v0, x0, y0, u0, v0], dtype=np.float32)
        self._draw_quad(data, texture, tint)

    def quad_corners(self, corners, texture, tint=(1.0, 1.0, 1.0, 1.0),
                     uv=(0.0, 0.0, 1.0, 1.0)) -> None:
        """
        Một quad từ 4 góc ``(trên-trái, trên-phải, dưới-phải, dưới-trái)``
        toạ độ THIẾT KẾ bất kỳ — dùng cho chữ hiệu ứng đã QUAY, thứ
        ``quad()`` không vẽ được vì nó chỉ nhận ``rect`` thẳng trục.
        """
        if texture is None:
            return
        box, fb_w, fb_h = self._box, *self._fb
        def _ndc(point):
            px, py = box.ox + point[0] * box.scale, box.oy + point[1] * box.scale
            return (2.0 * px / fb_w - 1.0, 1.0 - 2.0 * py / fb_h)
        tl, tr, br, bl = (_ndc(p) for p in corners)
        u0, v0, u1, v1 = uv
        data = np.array([bl[0], bl[1], u0, v1, br[0], br[1], u1, v1,
                         tr[0], tr[1], u1, v0, bl[0], bl[1], u0, v1,
                         tr[0], tr[1], u1, v0, tl[0], tl[1], u0, v0],
                        dtype=np.float32)
        self._draw_quad(data, texture, tint)

    def _draw_quad(self, data, texture, tint) -> None:
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        self._shader.use()
        GL.glUniform1i(GL.glGetUniformLocation(self._shader.program_id,
                                               "uTexture"), 0)
        GL.glUniform4f(GL.glGetUniformLocation(self._shader.program_id,
                                               "uTint"), *tint)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        GL.glBindVertexArray(self._vao)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)

    def fill(self, rect, colour) -> None:
        """Hình chữ nhật tô một màu (texture 1×1 trắng × tint)."""
        entry = self.cache.white()
        if entry:
            self.quad(rect, entry[0], colour)

    def frame(self, rect, colour, width: float) -> None:
        """Khung rỗng bốn thanh — vòng hover và vòng focus."""
        x, y, w, h = rect
        self.fill((x, y, w, width), colour)
        self.fill((x, y + h - width, w, width), colour)
        self.fill((x, y, width, h), colour)
        self.fill((x + w - width, y, width, h), colour)

    def rounded_frame(self, rect, colour, width: float, radius: float) -> None:
        """
        Khung rỗng bo góc — vòng hover/focus/selection quanh một ô BO GÓC.

        ``radius`` 0 (phần tử không bo, xem ``_ring_radius``) đi thẳng về
        ``frame()`` vuông góc: khỏi tốn một texture đệm cho trường hợp phổ
        biến nhất. ``radius`` > 0 dựng bằng PIL rồi đưa lên GPU, đệm theo
        khoá — cùng khuôn với ``cap``/``disc``/``pip`` ở trên.
        """
        if radius < 0.5:
            self.frame(rect, colour, width)
            return
        scale = self.scale
        w_px = max(8, int(round(rect[2] * scale)))
        h_px = max(8, int(round(rect[3] * scale)))
        key = ("rframe", w_px, h_px, round(width * scale, 2),
               round(radius * scale, 2))

        def build():
            image = Image.new("RGBA", (w_px, h_px), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            sw = max(1, int(round(width * scale)))
            half = sw / 2.0
            pen.rounded_rectangle(
                [half, half, w_px - 1 - half, h_px - 1 - half],
                radius=radius * scale, outline=(255, 255, 255, 255), width=sw)
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad(rect, entry[0], colour)

    # ---- texture của một màn --------------------------------------------
    def bind_screen(self, screen: str, extra_files=()) -> None:
        """
        Nạp ``base``, ``top`` và sprite của màn — **một lần cho mỗi màn**.

        ``base``/``top`` thu về đúng bề rộng framebuffer: ảnh xuất ở 1920 px
        mà cửa sổ 1280 thì giữ nguyên cỡ chỉ tốn RAM GPU và fillrate. Sprite
        giữ cỡ gốc (``item_tile``/``weapon_tile`` xuất ở 4× để còn nét).
        """
        manifest = self.manifest
        record = manifest.screens[screen]
        big = max(1, int(self._fb[0]))
        wanted: dict[tuple, int | None] = {}
        for layer in ("base", "top"):
            path = manifest.path(record.get(layer))
            if path is not None:
                wanted[(str(path), big)] = big
        for sprite in record.get("sprites", ()):
            path = manifest.path(sprite.get("file"))
            if path is not None:
                wanted[(str(path), 0)] = None
        for relative in extra_files:
            path = manifest.path(relative)
            if path is not None:
                wanted[(str(path), 0)] = None
        key = (screen, big, tuple(sorted(extra_files)))
        if key == self._screen_key:
            return
        for old_key, entry in list(self._screen_tex.items()):
            if old_key not in wanted:
                GL.glDeleteTextures(1, [entry[0]])
                del self._screen_tex[old_key]
        for tex_key, max_width in wanted.items():
            if tex_key in self._screen_tex:
                continue
            array = load_rgba(tex_key[0], max_width=max_width)
            if array is None:
                continue
            self._screen_tex[tex_key] = (upload_rgba(array), array.shape[1],
                                         array.shape[0])
        self._screen_key = key
        logger.debug("UI v3: man %s — %d texture", screen, len(self._screen_tex))

    def screen_image(self, relative: str | None, *, big: bool = False):
        """``(tex, w, h)`` của một file đã nạp cho màn đang mở."""
        path = self.manifest.path(relative)
        if path is None:
            return None
        return self._screen_tex.get((str(path), int(self._fb[0]) if big else 0))

    def layer(self, rect, relative: str | None) -> None:
        """Vẽ ``base``/``top`` phủ kín ``rect``."""
        entry = self.screen_image(relative, big=True)
        if entry:
            self.quad(rect, entry[0])

    def sprite(self, rect, relative: str | None, alpha: float = 1.0,
               mirror: bool = False) -> None:
        """``mirror`` lật ngang bằng UV — không cần file sprite thứ hai."""
        entry = self.screen_image(relative)
        if entry:
            uv = (1.0, 0.0, 0.0, 1.0) if mirror else (0.0, 0.0, 1.0, 1.0)
            self.quad(rect, entry[0], (1.0, 1.0, 1.0, float(alpha)), uv)

    # ---- chữ -------------------------------------------------------------
    def _text_entry(self, runs, texts):
        """
        Texture + số đo của một dòng chữ (một baseline, nhiều run).

        Hai tỉ lệ, và phải tách bạch:

        - **Số đo** lấy ở ``k`` = 1,5, đúng tỉ lệ ``tools/preview.py`` dùng.
          Bước tiến từng glyph bị FreeType làm tròn về pixel nguyên, nên bề
          rộng chuỗi không tỉ lệ thuận với cỡ chữ; đo ở tỉ lệ khác là lệch
          tới 11 % (xem ``V3Painter.k``).
        - **Nét chữ** rasterise ở tỉ lệ framebuffer thật. Dựng ở 1,5× rồi để
          GL thu nhỏ bằng bilinear (không mipmap) làm chữ rỗ và mảnh đi:
          hình học đúng tới ±1 px mà ``compare.py`` vẫn 30–50.

        Nên: vị trí bút tính ở ``k`` rồi quy về pixel thiết kế, còn glyph vẽ
        bằng font cỡ ``k``. Không có kerning (PIL layout cơ bản — đã kiểm:
        tổng bước tiến từng glyph khớp đúng bề rộng cả chuỗi), nên vẽ rời
        từng glyph cho ra đúng chữ mà ``preview.py`` vẽ liền một hơi.

        Đã thử rasterise ở tỉ lệ framebuffer cho chữ nét hơn: nó **xa**
        ``expected/`` hơn chứ không gần hơn (ô tệ nhất 49,6 → 59,1). Lý do:
        ``expected/`` cũng là bản 1,5× bị thu nhỏ, nên muốn giống nó thì
        phải đi qua đúng cái phễu thu nhỏ ấy. Mà hệ số thu ở đây là 0,711 —
        vẫn nằm trong mip 0, nên GL chỉ lọc bilinear một nhịp, không nhoè.
        """
        k = self.k
        scale = k
        style = tuple((r.get("ff"), r.get("fw"), round(float(r.get("fs", 16)), 3),
                       r.get("col"), round(float(r.get("ls") or 0.0), 3),
                       bool(r.get("upper")))
                      for r in runs)
        key = ("text", style, tuple(texts), round(scale, 5))

        def build():
            items, total = [], 0.0
            for record, raw in zip(runs, texts):
                text = raw.upper() if record.get("upper") else raw
                size = float(record.get("fs", 16))
                metric = self.fonts.get(record.get("ff"), record.get("fw"),
                                        round(size * k))
                glyph = self.fonts.get(record.get("ff"), record.get("fw"),
                                       round(size * scale))
                spacing = float(record.get("ls") or 0.0)
                items.append((metric, glyph, text, _byte(record.get("col")),
                              spacing))
                total += _run_width(metric, text, spacing * k) / k
            biggest = max(runs, key=lambda r: float(r.get("fs", 16)))
            ascent_k, descent_k = self.fonts.get(
                biggest.get("ff"), biggest.get("fw"),
                round(float(biggest.get("fs", 16)) * k)).getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            width = max(1, int(math.ceil(total * scale)) + 2 * TEXT_PAD)
            height = max(1, int(math.ceil((ascent + descent) * scale))
                         + 2 * TEXT_PAD)
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            cursor = 0.0
            baseline = float(TEXT_PAD) + ascent * scale
            for metric, glyph, text, colour, spacing in items:
                for char in text:
                    pen.text((TEXT_PAD + cursor * scale, baseline), char,
                             font=glyph, fill=colour, anchor="ls")
                    cursor += metric.getlength(char) / k + spacing
            return (np.asarray(image, dtype=np.uint8),
                    (float(total), float(ascent), float(descent)))

        return self.pil.get(key, build)

    def line(self, rect, runs, texts, align: str = "left",
             alpha: float = 1.0) -> float:
        """
        Vẽ một dòng chữ theo đúng phép canh của preview.py.

        Trả mép phải của chuỗi, tính bằng **pixel thiết kế** (con trỏ nhập
        bám vào đó). Canh dọc: hộp dòng của run **lớn nhất** nằm giữa
        ``rect``. Canh ngang theo ``align`` trên tổng bề rộng mọi run.
        """
        entry = self._text_entry(runs, texts)
        if not entry:
            return 0.0
        texture, width_px, height_px, (total, ascent, descent) = entry
        scale = self.k
        x, y, w, h = (float(v) for v in rect)
        if align == "right":
            start = x + w - total
        elif align == "center":
            start = x + (w - total) * 0.5
        else:
            start = x
        baseline = y + (h - (ascent + descent)) * 0.5 + ascent
        self.quad((start - TEXT_PAD / scale,
                   baseline - ascent - TEXT_PAD / scale,
                   width_px / scale, height_px / scale),
                  texture, (1.0, 1.0, 1.0, float(alpha)))
        return start + total

    def text_width(self, runs, texts) -> float:
        """Bề rộng một dòng chữ, tính bằng **pixel thiết kế**."""
        entry = self._text_entry(runs, texts)
        return 0.0 if not entry else entry[3][0]

    # ---- hình khối bo góc ------------------------------------------------
    def rounded(self, rect, radius: float, colour) -> None:
        """
        Hình chữ nhật bo góc, **không** gọi PIL mỗi khung.

        Chia ba: hai đầu lấy từ một texture đệm cao đúng ``h`` và rộng
        ``2r+2``, khúc giữa là một ô tô màu phẳng. Thanh tiến độ đổi bề rộng
        mỗi khung nên dựng lại ảnh PIL theo từng bề rộng là đường chắc chắn
        vỡ ngưỡng FPS (luật F4).
        """
        x, y, w, h = (float(v) for v in rect)
        if w <= 0.0 or h <= 0.0:
            return
        radius = max(0.0, min(float(radius), h * 0.5, w * 0.5))
        if radius < 0.5:
            self.fill(rect, colour)
            return
        scale = self.scale
        cap_w = 2.0 * radius + 2.0
        key = ("cap", round(cap_w * scale), round(h * scale),
               round(radius * scale, 2))

        def build():
            width = max(2, int(round(cap_w * scale)))
            height = max(2, int(round(h * scale)))
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(image).rounded_rectangle(
                [0, 0, width - 1, height - 1], radius=radius * scale,
                fill=(255, 255, 255, 255))
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if not entry:
            return
        texture = entry[0]
        half = radius + 1.0
        if w <= cap_w:
            # Hẹp hơn cả hai đầu: kéo nguyên ảnh đệm. README bảo thanh không
            # bao giờ hẹp hơn 2×radius, nên đây chỉ là lưới an toàn.
            self.quad((x, y, w, h), texture, colour)
            return
        middle = w - 2.0 * half
        self.quad((x, y, half, h), texture, colour, (0.0, 0.0, 0.5, 1.0))
        self.quad((x + half, y, middle, h), texture, colour,
                  (0.5, 0.0, 0.5, 1.0))
        self.quad((x + w - half, y, half, h), texture, colour,
                  (0.5, 0.0, 1.0, 1.0))

    def disc(self, rect, colour) -> None:
        """
        Hình tròn đặc, mềm biên. Texture TRẮNG đệm theo cỡ, màu đi qua tint.

        Vì sao không dùng ``pip``: khoá đệm của nó gồm cả màu, mà lỗ đạn thì
        mờ dần từng khung — mỗi khung một màu là mỗi khung một texture mới.
        Tách màu ra khỏi khoá thì cả vệt mờ dần chỉ tốn đúng một texture.
        """
        scale = self.scale
        size = max(1, int(round(float(rect[2]) * scale)))
        key = ("disc", size, max(1, int(round(float(rect[3]) * scale))))

        def build():
            width = size + 1
            height = max(1, int(round(float(rect[3]) * scale))) + 1
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            ImageDraw.Draw(image).ellipse([0, 0, width - 1, height - 1],
                                          fill=(255, 255, 255, 255))
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad((rect[0], rect[1], entry[1] / scale, entry[2] / scale),
                      entry[0], colour)

    def pip(self, rect, shape: str, radius: float, style: dict) -> None:
        """
        Một pip: bầu dục hoặc chữ nhật bo góc, tô và/hoặc viền.

        ``fill_op`` được tôn trọng thật. ``preview.py`` vẽ pip mờ bằng PIL
        lên canvas RGBA rồi ``convert('RGB')`` ở cuối — thao tác đó **vứt**
        kênh alpha chứ không trộn, nên trong ``expected/`` pip "off" hiện ra
        đặc kịt. ``design/*.jpg`` (README: "the visual truth") cho thấy nó
        phải mờ, và đó là thứ dãy pip dùng để nói "lớp này chưa tới".
        """
        scale = self.k
        key = ("pip", shape, round(rect[2] * scale), round(rect[3] * scale),
               round(radius * scale, 2), style.get("fill"),
               round(float(style.get("fill_op", 1.0) or 1.0), 3),
               style.get("stroke"), round(float(style.get("sw", 1) or 1) * scale, 2))

        def build():
            width = max(1, int(round(rect[2] * scale))) + 1
            height = max(1, int(round(rect[3] * scale))) + 1
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            fill = (_byte(style["fill"], float(style.get("fill_op", 1.0) or 1.0))
                    if style.get("fill") else None)
            outline = _byte(style["stroke"]) if style.get("stroke") else None
            stroke_px = max(1, int(round(float(style.get("sw", 1) or 1) * scale)))
            box = [0, 0, width - 1, height - 1]
            if shape == "ellipse":
                pen.ellipse(box, fill=fill, outline=outline, width=stroke_px)
            else:
                pen.rounded_rectangle(box, radius=radius * scale, fill=fill,
                                      outline=outline, width=stroke_px)
            return np.asarray(image, dtype=np.uint8), None

        entry = self.pil.get(key, build)
        if entry:
            self.quad((rect[0], rect[1], entry[1] / scale, entry[2] / scale),
                      entry[0])

    def pill(self, record: dict, text: str, style: dict) -> None:
        """
        Viên thuốc: nền bo góc + chữ canh giữa, dựng thành **một** texture.

        Bề rộng = bề rộng chữ + 2×``pad``, đúng công thức của preview.py (kể
        cả chỗ nó nhân ``ls`` với ``len`` chứ không phải ``len - 1``).
        """
        k = self.k
        scale = k
        font_spec = record["font"]
        x, y, _w, h = (float(v) for v in record["rect"])
        radius = float(record.get("radius", 0.0))
        pad = float(record.get("pad", 0.0))
        key = ("pill", text, font_spec["ff"], font_spec["fw"],
               round(float(font_spec["fs"]), 3),
               round(float(font_spec.get("ls") or 0.0), 3),
               style.get("bg"), style.get("fg"), round(h, 2), round(radius, 2),
               round(pad, 2), round(scale, 5))

        def build():
            size = float(font_spec["fs"])
            metric = self.fonts.get(font_spec["ff"], font_spec["fw"],
                                    round(size * k))
            glyph = self.fonts.get(font_spec["ff"], font_spec["fw"],
                                   round(size * scale))
            spacing = float(font_spec.get("ls") or 0.0)
            # Công thức bề rộng của preview.py, kể cả chỗ nó nhân ``ls`` với
            # ``len`` chứ không phải ``len - 1``.
            design_w = metric.getlength(text) / k + spacing * len(text) + 2.0 * pad
            width = max(2, int(round(design_w * scale))) + 1
            height = max(2, int(round(h * scale))) + 1
            image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
            pen = ImageDraw.Draw(image)
            pen.rounded_rectangle([0, 0, width - 1, height - 1],
                                  radius=radius * scale, fill=_byte(style.get("bg")))
            total = _run_width(metric, text, spacing * k) / k
            ascent_k, descent_k = metric.getmetrics()
            ascent, descent = ascent_k / k, descent_k / k
            cursor = (width - 1 - total * scale) * 0.5
            baseline = (height - 1 - (ascent + descent) * scale) * 0.5 + ascent * scale
            colour = _byte(style.get("fg"))
            for char in text:
                pen.text((cursor, baseline), char, font=glyph, fill=colour,
                         anchor="ls")
                cursor += (metric.getlength(char) / k + spacing) * scale
            return np.asarray(image, dtype=np.uint8), design_w

        entry = self.pil.get(key, build)
        if entry:
            self.quad((x, y, entry[1] / scale, entry[2] / scale), entry[0])

    # ---- ảnh người dùng --------------------------------------------------
    def cover(self, rect, source, clip=None) -> bool:
        """Ảnh cover-fit (phủ kín, cắt giữa) vào ``rect``; ``clip`` thu hẹp."""
        entry = self.cache.image(source)
        if not entry:
            return False
        _tex, tex_w, tex_h = entry
        factor = max(rect[2] / tex_w, rect[3] / tex_h)
        width, height = tex_w * factor, tex_h * factor
        placed = (rect[0] + (rect[2] - width) * 0.5,
                  rect[1] + (rect[3] - height) * 0.5, width, height)
        GL.glEnable(GL.GL_SCISSOR_TEST)
        for area in (clip or [rect]):
            GL.glScissor(*self.viewport_px(area))
            self.quad(placed, entry[0])
        GL.glDisable(GL.GL_SCISSOR_TEST)
        return True

    def release(self) -> None:
        self.cache.release()
        self.pil.release()
        for entry in self._screen_tex.values():
            GL.glDeleteTextures(1, [entry[0]])
        self._screen_tex.clear()
        if self._vao:
            GL.glDeleteVertexArrays(1, [self._vao])
        if self._vbo:
            GL.glDeleteBuffers(1, [self._vbo])
        self._shader.release()


def _run_width(font, text: str, spacing: float) -> float:
    """Bề rộng một run, kể cả letter-spacing (y như ``preview.run_width``)."""
    if spacing:
        return font.getlength(text) + spacing * max(0, len(text) - 1)
    return font.getlength(text)


class V3State:
    """
    Mọi thứ đổi theo thời gian của một màn v3.

    Lớp vẽ cố tình **không biết** luật nào: ``v3_values.py`` quyết định sprite
    nào bật, chữ nào thay, thanh đầy bao nhiêu, rồi đổ vào đây. Nhờ vậy lượt
    chụp (``--ui-shots``) chỉ là một bộ giá trị khác, không phải một nhánh mã
    khác — mà README nói rõ ``expected/`` chính là chế độ chụp.
    """

    __slots__ = ("texts", "colours", "hidden", "sprites", "borrowed", "bars",
                 "pips", "pills", "photo", "camera_tex", "viewport_draw",
                 "hover", "focus", "caret", "alpha", "shot", "message", "wrap",
                 "extra", "mirror", "aim", "shots", "aim_ready", "confirm",
                 "text_rect", "fx_on", "fx_draw", "selected", "pressed",
                 "hover_fx", "aim_kick_scale", "tracers", "hit_markers",
                 "crit_impact_alpha", "scene_shake", "scope_zoom")

    def __init__(self) -> None:
        #: ``key`` → chuỗi (thay run đầu) hoặc danh sách chuỗi (thay từng run).
        self.texts: dict = {}
        #: ``key`` → mã màu, hoặc danh sách mã màu theo từng run.
        self.colours: dict = {}
        self.hidden: set = set()
        #: ``key`` → ``True`` (vẽ ở ``rect`` của nó) hoặc ``rect`` đặt chỗ khác.
        self.sprites: dict = {}
        #: Sprite app tự đặt chỗ: ``nhãn`` → ``(màn, key, rect)``. Nhãn là duy
        #: nhất nên **một** sprite vẽ được nhiều lần (ba ô vật phẩm, ba hàng
        #: dấu kiểm), và ``màn`` khác màn đang mở thì là sprite vay — HUD lấy
        #: ``item_tile:*``/``weapon_tile:*`` của màn trang bị (README).
        self.borrowed: dict = {}
        #: ``key`` → tỉ lệ đầy 0–1.
        self.bars: dict = {}
        #: ``key`` → số pip đang bật, hoặc danh sách cờ, hoặc ``("n", trần)``.
        self.pips: dict = {}
        #: ``key`` → (chuỗi, tên kiểu).
        self.pills: dict = {}
        self.photo = None
        self.camera_tex = None
        self.viewport_draw = None
        self.hover: str | None = None
        self.focus: str | None = None
        #: ``key`` của ô chữ đang nhận chữ gõ — vẽ con trỏ sau nó.
        self.caret: str | None = None
        #: ``key`` → độ mờ 0–1 (ô vật phẩm cấp 1 vẽ 45 %).
        self.alpha: dict = {}
        self.shot = False
        self.message: str = ""
        self.wrap = True
        #: TASK_shared_album_1111: đường dẫn ảnh đang chờ xác nhận chia sẻ;
        #: "" (mặc định của ``str``) = không có hộp thoại nào đang mở.
        self.confirm: str = ""
        #: Nhãn sprite cần lật ngang (mũi tên nghiêng đầu khi góc > 0).
        self.mirror: set = set()
        #: Tâm ngắm ``(x, y)`` trong toạ độ thiết kế; ``None`` = không vẽ.
        self.aim = None
        #: Vết đạn ``(x, y, tuổi giây, trúng?)``, mới nhất ở cuối.
        self.shots: list = []
        #: Súng đã hồi tới đâu, 0–1. Dưới 1 thì tâm ngắm bung ra và mờ đi.
        self.aim_ready: float = 1.0
        #: Dòng chữ app tự đặt chỗ: ``(rect, runs, texts, align)``. Bản vẽ chỉ
        #: xuất MỘT thực thể cho mỗi kiểu ô vật phẩm / dòng điểm, còn ván chơi
        #: cần ba ô và số dòng thay đổi — chỗ lặp lại nằm ở đây, không phải
        #: trong manifest.
        self.extra: list = []
        #: TASK_ui_sakuga_1117 bug A: ``key`` → ``rect`` đặt chỗ khác cho MỘT
        #: chữ sẵn có trong bản vẽ (khác ``extra``, vốn dùng khi cần NHIỀU
        #: bản của cùng một kiểu chữ). ``check_hint`` cần đúng việc này: dời
        #: theo hàng đang hỏi, giữ nguyên kiểu chữ/x.
        self.text_rect: dict = {}
        #: TASK_ui_sakuga_1117: Sakuga VFX on for this frame? Only gates
        #: the old green ``HIT_FLASH`` (shot marks always draw either way)
        #: — the real on/off switch for spawning/drawing effects lives in
        #: ``fx_draw`` below and in the caller that decides whether to
        #: spawn anything at all.
        self.fx_on: bool = False
        #: Zero-arg callback that draws the live Sakuga effects for this
        #: frame, or ``None``. Kept as a callback (not a direct dependency
        #: on ``FxLibrary``) so this module stays GL-texture-agnostic;
        #: ``app.py`` wires it to ``FxLibrary.draw(player)``.
        self.fx_draw = None
        #: TASK_ui_sakuga_1117 Phase 5: hotspot names that are the CURRENT
        #: selection (active tab/level/weapon/display/theme) — their hover
        #: ring draws at half alpha even while hovered, so hovering an
        #: already-selected row doesn't look identical to hovering a
        #: plain unselected one.
        self.selected: set = set()
        #: Left mouse button held down on ``hover`` right now (not just
        #: hovered) — the pressed tint/inset ring in ``v3_hover.py``.
        self.pressed: bool = False
        #: The persistent ``v3_hover.HoverFx`` instance (springs survive
        #: across frames; ``V3State`` itself does not) — ``None`` means
        #: "no hover feedback this frame" (``shot_state``'s screenshots).
        self.hover_fx = None
        #: TASK_game_feel_1118 P4 — all five default to "nothing kicked/
        #: shaking/marked right now"; ``v3_ui.py`` fills them in on
        #: PLAY/PVP_HUNT from its persistent ``ShotFeel``.
        self.aim_kick_scale: float = 1.0
        #: ``[(muzzle, target, age_s, muzzle_width_px), ...]``.
        self.tracers: list = []
        #: ``[(x, y, age_s, crit), ...]``.
        self.hit_markers: list = []
        self.crit_impact_alpha: float = 0.0
        #: Design-px ``(dx, dy)`` the 3D scene viewport and the fx layer
        #: are shifted by — HUD panels/text never move with it.
        self.scene_shake: tuple = (0.0, 0.0)
        #: TASK_game_feel_1118 P6: Precision-gun scope zoom, 1.0 = not
        #: scoped (no vignette/reticle drawn) up to 2.5 at full zoom.
        self.scope_zoom: float = 1.0


def shot_state(manifest: V3Manifest, screen: str) -> V3State:
    """
    Trạng thái của **chế độ chụp** = đúng luật ``expected/`` trong README.

    Mọi chữ lấy ``sample``, mọi sprite ``default: true`` vẽ ở ``rect`` của nó,
    pip theo ``sample_on``, thanh theo ``sample_frac``, viên thuốc lấy kiểu
    đầu tiên, ô ảnh lấy ``sample/sample_room.png``, không hover/focus/caret.
    Riêng ``wrap`` phải TẮT: ``preview.py`` bỏ qua cờ đó, nên ``expected/``
    vẽ ``tip_text`` thành một dòng.
    """
    record = manifest.screens[screen]
    state = V3State()
    state.shot = True
    state.wrap = False
    state.photo = manifest.sample_photo()
    state.sprites = {s["key"]: True for s in record.get("sprites", ())
                     if s.get("default")}
    state.bars = {b["key"]: float(b.get("sample_frac", 0.0))
                  for b in record.get("bars", ())}
    state.pips = {p["key"]: list(p.get("sample_on", ()))
                  for p in record.get("pips", ())}
    state.pills = {p["key"]: (p.get("sample") or "",
                              next(iter(p["styles"]), None))
                   for p in record.get("pills", ())}
    return state


def _texts_for(record: dict, state: V3State) -> list[str]:
    """Chuỗi của từng run sau khi thay giá trị sống."""
    runs = record["runs"]
    value = state.texts.get(record["key"])
    if value is None:
        return [str(r.get("c") or "") for r in runs]
    if isinstance(value, (list, tuple)):
        parts = list(value)
        # Nhiều giá trị hơn số run thì DỒN phần thừa vào run cuối. Cắt bỏ
        # chúng là mất chữ mà không báo gì: ô "LAYER 2/4" chỉ có một run, đưa
        # vào ``["2", "/4"]`` thì màn chơi hiện mỗi "2".
        if len(parts) > len(runs) and runs:
            parts = parts[:len(runs) - 1] + ["".join(
                "" if p is None else str(p) for p in parts[len(runs) - 1:])]
        return [str(r.get("c") or "") if v is None else str(v)
                for r, v in zip(runs, parts + [None] * len(runs))]
    # Một giá trị cho nhiều run: run đầu là con số sống, run sau giữ nguyên
    # ("7" + "/12" — README mục Draw order 8).
    return [str(value)] + [str(r.get("c") or "") for r in runs[1:]]


def _runs_for(record: dict, state: V3State) -> list[dict]:
    """Run sau khi thay màu sống (``key`` hoặc từng run)."""
    runs = record["runs"]
    colour = state.colours.get(record["key"])
    if colour is None:
        return runs
    values = colour if isinstance(colour, (list, tuple)) else [colour]
    out = []
    for index, run in enumerate(runs):
        new = values[index] if index < len(values) else None
        out.append({**run, "col": new} if new else run)
    return out


def _wrap_lines(painter: V3Painter, runs, texts, width: float) -> list[list[str]]:
    """Cắt dòng theo bề rộng ``rect.w`` (chỉ dùng cho text có ``wrap``)."""
    words = (texts[0] or "").split()
    if not words:
        return [texts]
    lines, current = [], ""
    for word in words:
        probe = f"{current} {word}".strip()
        if current and painter.text_width(runs[:1], [probe]) > width:
            lines.append(current)
            current = word
        else:
            current = probe
    lines.append(current)
    return [[line] + texts[1:] for line in lines]


def draw_screen(painter: V3Painter, manifest: V3Manifest, screen: str,
                state: V3State) -> None:
    """Vẽ trọn một màn theo thứ tự lớp của README."""
    record = manifest.screens[screen]
    full = (0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1])
    painter.layer(full, record.get("base"))
    _draw_photos(painter, manifest, record, state)
    # TASK_game_feel_fix_1118 Fix 4: the crit impact frame draws here —
    # BEFORE ``top.png``, not after. ``top.png`` is where the HUD CARD
    # BACKGROUNDS are actually baked (the dynamic numbers/bars/pips drawn
    # further below were always on top and stayed readable); drawing the
    # wash AFTER ``top.png`` (the 1118 version) painted it straight over
    # those opaque card backgrounds, visibly darkening them — exactly
    # what ``hud_layer_check.png``/G3 catches. Now the card art (drawn
    # next) covers the wash wherever a card sits, leaving it visible only
    # in the open scene area.
    #
    # ``_draw_scope_vignette`` below is left as-is (still after
    # ``top.png``, so it has the SAME card-darkening issue today) — out
    # of Fix 4's own scope, which names only the crit impact frame; the
    # scope's own HUD handling is Fix 10's "every card fades to 0 while
    # scoped", a bigger, deliberate redesign rather than a one-line
    # reorder like this one.
    _draw_crit_impact(painter, state)
    painter.layer(full, record.get("top"))
    _draw_scope_vignette(painter, state)
    _draw_bars(painter, record, state)
    _draw_pips(painter, record, state)
    _draw_pills(painter, record, state)
    _draw_sprites(painter, manifest, screen, record, state)
    _draw_texts(painter, record, state)
    _draw_choose_photo_extra(painter, screen, state)
    _draw_shots(painter, state)
    _draw_tracers(painter, state)
    _draw_aim(painter, state)
    _draw_scope_reticle(painter, state)
    _draw_hit_markers(painter, state)
    _draw_rings(painter, manifest, screen, record, state)
    _draw_toast(painter, state, screen)
    # TASK_ui_sakuga_1117: above HUD/toast (so a hit's burst reads clearly
    # over gameplay) but below the share-album confirm dialog (so an open
    # dialog's own ui_tap effect, drawn separately by the dialog's own
    # click handling, is never occluded by an unrelated effect still
    # playing underneath it).
    if state.fx_draw is not None:
        state.fx_draw()
    _draw_confirm_dialog(painter, state)


def _draw_photos(painter, manifest, record, state) -> None:
    for photo in record.get("photos", ()):
        kind, rect = photo.get("kind"), photo["rect"]
        if kind == "photo":
            source = state.photo
            if source is not None and Path(source).exists():
                painter.cover(rect, source, photo.get("clip"))
            elif state.shot:
                painter.cover(rect, manifest.sample_photo(), photo.get("clip"))
        elif kind == "camera":
            if state.shot:
                painter.fill(rect, rgba("#%02X%02X%02X" % SHOT_CAMERA_RGB))
            elif state.camera_tex is not None:
                _cover_texture(painter, rect, state.camera_tex, mirror=True)
        elif kind == "viewport":
            if state.shot:
                painter.fill(rect, rgba("#%02X%02X%02X" % SHOT_VIEWPORT_RGB))
            elif state.viewport_draw is not None:
                # TASK_game_feel_1118 P4: screen shake moves the 3D scene
                # viewport (this rect) and the fx layer only — never the
                # HUD, so it stays readable while the scene shakes.
                dx, dy = state.scene_shake
                shaken = (rect[0] + dx, rect[1] + dy, rect[2], rect[3]) \
                    if (dx or dy) else rect
                state.viewport_draw(shaken)


def _cover_texture(painter, rect, entry, mirror: bool = False) -> None:
    """Texture cover-fit vào ``rect`` (khung webcam, lật ngang nếu cần)."""
    texture, tex_w, tex_h = entry
    factor = max(rect[2] / tex_w, rect[3] / tex_h)
    width, height = tex_w * factor, tex_h * factor
    uv = (1.0, 0.0, 0.0, 1.0) if mirror else (0.0, 0.0, 1.0, 1.0)
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glScissor(*painter.viewport_px(rect))
    painter.quad((rect[0] + (rect[2] - width) * 0.5,
                  rect[1] + (rect[3] - height) * 0.5, width, height),
                 texture, uv=uv)
    GL.glDisable(GL.GL_SCISSOR_TEST)


def _draw_bars(painter, record, state) -> None:
    for bar in record.get("bars", ()):
        key = bar["key"]
        if key in state.hidden:
            continue
        x, y, w, h = (float(v) for v in bar["rect"])
        radius = float(bar.get("radius", 0.0))
        fraction = state.bars.get(key)
        if fraction is None:
            fraction = float(bar.get("sample_frac", 0.0))
        fraction = max(0.0, min(1.0, float(fraction)))
        if bar.get("rest_col"):
            painter.rounded((x, y, w, h), radius, rgba(bar["rest_col"]))
        # Không bao giờ hẹp hơn 2×radius (README): một viên bo góc bị bóp
        # dưới mức đó thành vệt nhòn nhọn, không còn là "thanh đầy 0 %".
        painter.rounded((x, y, max(w * fraction, 2.0 * radius), h), radius,
                        rgba(bar["col"]))


def _pip_flags(spec: dict, state: V3State) -> list[bool]:
    """Cờ bật/tắt từng pip: giá trị sống, nếu không thì ``sample_on``."""
    total = len(spec["rects"])
    value = state.pips.get(spec["key"])
    if value is None:
        flags = list(spec.get("sample_on", ()))
        return [bool(flags[i]) if i < len(flags) else False for i in range(total)]
    if isinstance(value, (list, tuple)):
        return [bool(value[i]) if i < len(value) else False for i in range(total)]
    return [i < int(value) for i in range(total)]


def _draw_pips(painter, record, state) -> None:
    for spec in record.get("pips", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        flags = _pip_flags(spec, state)
        limit = state.pips.get(f"{key}:limit")
        for index, rect in enumerate(spec["rects"]):
            if limit is not None and index >= int(limit):
                continue          # pip vượt quá N thì ẩn hẳn (HUD lớp/đạn)
            style = spec["on"] if flags[index] else spec["off"]
            painter.pip(rect, spec.get("shape", "rect"),
                        float(spec.get("radius", 0.0)), style)


def _draw_pills(painter, record, state) -> None:
    for spec in record.get("pills", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        text, style_name = state.pills.get(key, (None, None))
        if text is None:
            text = spec.get("sample") or ""
        styles = spec["styles"]
        style = styles.get(style_name) if style_name else None
        if style is None:
            style = next(iter(styles.values()))
        painter.pill(spec, str(text), style)


def _draw_sprites(painter, manifest, screen, record, state) -> None:
    for sprite in record.get("sprites", ()):
        key = sprite["key"]
        if key in state.hidden or key not in state.sprites:
            continue
        where = state.sprites[key]
        rect = sprite["rect"] if where is True else where
        painter.sprite(rect, sprite.get("file"),
                       float(state.alpha.get(key, 1.0)), key in state.mirror)
    for label, (other, key, rect) in state.borrowed.items():
        borrowed = manifest.sprite(other, key)
        if borrowed is not None:
            painter.sprite(rect, borrowed.get("file"),
                           float(state.alpha.get(label, 1.0)),
                           label in state.mirror)
    _ = screen


def _draw_texts(painter, record, state) -> None:
    for spec in record.get("texts", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        runs = _runs_for(spec, state)
        texts = _texts_for(spec, state)
        alpha = float(state.alpha.get(key, 1.0))
        rect = [float(v) for v in state.text_rect.get(key, spec["rect"])]
        if spec.get("wrap") and state.wrap:
            line_h = float(spec.get("line_h") or rect[3])
            lines = _wrap_lines(painter, runs, texts, rect[2])
            for index, parts in enumerate(lines):
                painter.line((rect[0], rect[1] + index * line_h, rect[2], line_h),
                             runs, parts, spec.get("al", "left"), alpha)
            continue
        right = painter.line(rect, runs, texts, spec.get("al", "left"), alpha)
        if state.caret == key:
            painter.fill((right + CARET_GAP,
                          rect[1] + (rect[3] - CARET_SIZE[1]) * 0.5,
                          CARET_SIZE[0], CARET_SIZE[1]), rgba(CARET_COL))
    for rect, runs, texts, align in state.extra:
        painter.line(rect, runs, texts, align)


def _hole(painter, x: float, y: float, size: float, colour) -> None:
    """Một lớp tròn của lỗ đạn, canh giữa ``(x, y)``."""
    painter.disc((x - size * 0.5, y - size * 0.5, size, size), colour)


def _scope_adjust(x: float, y: float, state) -> tuple[float, float, bool]:
    """TASK_game_feel_fix_1118 Fix 5: a scene-anchored overlay point
    (stored once, un-zoomed, at spawn time — the glass point ``g``) ->
    its CURRENT screen position while scoped, ``c_now + z_now·(g -
    c_now)``, or unchanged when not scoped. Third element: ``False`` if
    the (possibly moved) point now falls outside the scope's own vignette
    circle — "hide them outside the scope circle" — the caller should
    skip drawing in that case."""
    if state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS or state.aim is None:
        return x, y, True
    sx, sy = screen_scope_transform((x, y), state.aim, state.scope_zoom)
    if math.hypot(sx - state.aim[0], sy - state.aim[1]) > VIGNETTE_RADIUS_PX:
        return sx, sy, False
    return sx, sy, True


def _draw_shots(painter, state) -> None:
    """
    Lỗ đạn ở chỗ vừa bắn, cộng một chớp xanh khi trúng bọ.

    Giữ nguyên lỗ gần như suốt tuổi thọ rồi mới mờ hẳn ở quãng cuối: lỗ đạn
    trên ảnh thì không nhạt dần ngay, và người chơi cần nhìn lại được mình
    vừa bắn chệch về phía nào. Giữ vài phát gần nhất chứ không chỉ phát cuối.

    TASK_game_feel_fix_1118 Fix 5: each mark's stored ``(x, y)`` is the
    glass point it was fired at (unzoomed design px, by construction —
    see ``game_bridge.fire``'s own comment) — ``_scope_adjust`` re-derives
    where that point sits on screen THIS frame, scoped or not.
    """
    if state.shot:
        return
    for x, y, age, hit in state.shots:
        age = float(age)
        if age >= HOLE_LIFE_S:
            continue
        x, y, visible = _scope_adjust(x, y, state)
        if not visible:
            continue
        # Mờ dần CHỈ ở quãng cuối, trước đó giữ nguyên độ đậm.
        left = min(1.0, max(0.0, (HOLE_LIFE_S - age) / HOLE_FADE_S))
        _hole(painter, x, y, HOLE_DUST, rgba(HOLE_DUST_COL, 0.22 * left))
        _hole(painter, x, y, HOLE_RING, rgba(HOLE_RING_COL, 0.72 * left))
        _hole(painter, x, y, HOLE_CORE, rgba(HOLE_CORE_COL, 0.92 * left))
        if hit and age < HIT_FLASH_S and not state.fx_on:
            # Sakuga's own hit_kill/hit_crit sprite replaces this flash
            # when fx is on (TASK_ui_sakuga_1117) — off, it's unchanged.
            flash = 1.0 - age / HIT_FLASH_S
            _hole(painter, x, y, HIT_FLASH_SIZE * (1.0 - flash * 0.5),
                  rgba(HIT_FLASH_COL, 0.55 * flash))


def _draw_toast(painter, state, screen=None) -> None:
    """
    Dòng báo nổi ở giữa dưới.

    Thiếu nó thì mọi lời ``v3_flow.say`` — "Code copied", "Night theme is not
    in v2 yet", "Reloading" — được đặt rồi **không ai vẽ**, nên người dùng bấm
    nút mà tưởng như không có gì xảy ra. Toast của renderer không thay được:
    nó vẽ trong pass cảnh 3D, mà màn không có ô viewport thì pass ấy bị cắt
    còn 1 pixel.
    """
    if not state.message or state.shot:
        return
    x, y, w, h = TOAST_RECT
    if screen == CHOOSE_PHOTO_SCREEN:
        y = 574.0
    text = str(state.message)
    width = min(w, painter.text_width([TOAST_FONT], [text]) + 2 * TOAST_RADIUS)
    left = x + (w - width) * 0.5
    painter.rounded((left, y, width, h), TOAST_RADIUS, rgba(TOAST_BG, 0.92))
    painter.line((left, y, width, h), [TOAST_FONT], [text], "center")


def _draw_choose_photo_extra(painter, screen, state) -> None:
    """Nút "Share to album" / "Random from album" — xem ``SHARE_BTN_RECT``."""
    if state.shot or screen != CHOOSE_PHOTO_SCREEN:
        return
    for rect, label_key, alpha_key, default in (
            (SHARE_BTN_RECT, "share_to_album_label", "share_to_album_btn",
             "Share to album"),
            (RANDOM_BTN_RECT, "random_from_album_label", "random_from_album_btn",
             "Random from album")):
        label = state.texts.get(label_key, default)
        alpha = float(state.alpha.get(alpha_key, 1.0))
        painter.rounded(rect, BTN_CORNER_RADIUS_PX, rgba("#FFFFFF"))
        painter.rounded_frame(rect, rgba(BUTTON_PRIMARY_COL, alpha), 1.5,
                              BTN_CORNER_RADIUS_PX)
        font = dict(SHARE_BTN_FONT)
        if rect == SHARE_BTN_RECT:
            font["fs"] = 12.0
        # The disabled label must fit even in the small caption-row button.
        width = painter.text_width([font], [str(label)])
        if width > rect[2] - 16:
            font["fs"] *= (rect[2] - 16) / width
        painter.line(rect, [font], [str(label)], "center", alpha)


def _draw_confirm_dialog(painter, state) -> None:
    """Hộp xác nhận chia sẻ — nổi trên mọi thứ, kể cả toast (P2.3)."""
    if not state.confirm or state.shot:
        return
    painter.fill((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1]), rgba("#0B1520", 0.45))
    x, y, w, h = CONFIRM_CARD_RECT
    painter.rounded((x, y, w, h), 18.0, rgba("#FFFFFF"))
    pad, line_h = 28.0, 22.0
    lines = _wrap_lines(painter, [CONFIRM_TEXT_FONT], [CONFIRM_TEXT], w - 2 * pad)
    for index, parts in enumerate(lines):
        painter.line((x + pad, y + pad + index * line_h, w - 2 * pad, line_h),
                     [CONFIRM_TEXT_FONT], parts, "left")
    for name, rect, label, bg in (
            ("btn_confirm_no", CONFIRM_NO_RECT, "No", "#DCE9F7"),
            ("btn_confirm_yes", CONFIRM_YES_RECT, "Yes", BUTTON_PRIMARY_COL)):
        fg = INK if bg != BUTTON_PRIMARY_COL else "#FFFFFF"
        painter.rounded(rect, 10.0, rgba(bg))
        painter.line(rect, [{**CONFIRM_BTN_FONT, "col": fg}], [label], "center")
        if state.hover == name:
            painter.rounded_frame(rect, rgba(CARET_COL), HOVER_PX, 10.0)


def _draw_aim(painter, state) -> None:
    """
    Dấu ``+`` ở tâm ngắm: bốn nhánh, chừa một khoảng hở ở giữa.

    Chừa hở để chính cái tâm — chỗ viên đạn sẽ tới — không bị nét vẽ che.
    """
    if state.aim is None or state.shot:
        return
    if state.scope_zoom > 1.0 + ZOOM_SETTLED_EPS:
        return          # TASK_game_feel_1118 P6: the scope's own reticle instead
    x, y = float(state.aim[0]), float(state.aim[1])
    ready = max(0.0, min(1.0, float(state.aim_ready)))
    # Chưa hồi xong: bốn nhánh dạt ra và nhạt bớt, rồi khép lại dần.
    spread = AIM_RECOIL * (1.0 - ready)
    fade = AIM_DIM + (1.0 - AIM_DIM) * ready
    # TASK_game_feel_1118 P4: crosshair kick — a ``Spring`` kicked out to
    # the weapon's scale on every shot, settling back to 1.0. Scales the
    # whole cross (arm length AND thickness) around its own centre.
    kick = max(0.1, float(getattr(state, "aim_kick_scale", 1.0)))
    for base, grow in ((AIM_EDGE_COL, AIM_EDGE), (AIM_COL, 0.0)):
        alpha = (0.85 if grow else 1.0) * fade
        colour = rgba(base, alpha)
        half = (AIM_THICK * 0.5 + grow) * kick
        inner = (AIM_GAP + spread - grow) * kick
        outer = (AIM_ARM + spread + grow) * kick
        painter.fill((x - outer, y - half, outer - inner, half * 2.0), colour)
        painter.fill((x + inner, y - half, outer - inner, half * 2.0), colour)
        painter.fill((x - half, y - outer, half * 2.0, outer - inner), colour)
        painter.fill((x - half, y + inner, half * 2.0, outer - inner), colour)


def _flat_quad(painter, corners, colour) -> None:
    """A flat-coloured arbitrary quad — ``quad_corners`` with the painter's
    cached 1x1 white texture, same trick as ``fill()`` uses for axis-
    aligned rects."""
    entry = painter.cache.white()
    if entry:
        painter.quad_corners(corners, entry[0], colour)


def _draw_tracers(painter, state) -> None:
    """TASK_game_feel_1118 P4: a tapered streak from the muzzle to the
    crosshair, white core over a dark edge, fading over its 0.07 s life.
    Fixed to the HUD (muzzle is the weapon icon) — unaffected by shake."""
    if state.shot:
        return
    for muzzle, target, age, width_px in state.tracers:
        if age >= TRACER_LIFE_S:
            continue
        fade = 1.0 - age / TRACER_LIFE_S
        mx, my = muzzle
        tx, ty = target
        dx, dy = tx - mx, ty - my
        length = math.hypot(dx, dy) or 1.0
        nx, ny = -dy / length, dx / length
        for col, fraction in ((TRACER_EDGE_COL, 1.0), (TRACER_CORE_COL,
                                                        TRACER_CORE_FRACTION)):
            hw_m = width_px * 0.5 * fraction
            hw_t = TRACER_TARGET_WIDTH_PX * 0.5 * fraction
            corners = ((mx + nx * hw_m, my + ny * hw_m),
                      (tx + nx * hw_t, ty + ny * hw_t),
                      (tx - nx * hw_t, ty - ny * hw_t),
                      (mx - nx * hw_m, my - ny * hw_m))
            _flat_quad(painter, corners, rgba(col, fade))


def _draw_hit_markers(painter, state) -> None:
    """TASK_game_feel_1118 P4: four ticks at 45° around the crosshair on a
    hit, outlined, fading over 0.18 s — bigger and ``#E8676B`` on a crit.

    TASK_game_feel_fix_1118 Fix 5: a hit marker is spawned at the
    crosshair's position AT THE TIME of the shot — same scene-anchored
    treatment as shot marks (``_scope_adjust``) once the crosshair has
    since moved while still scoped.
    """
    if state.shot:
        return
    for x, y, age, crit in state.hit_markers:
        if age >= HITMARK_LIFE_S:
            continue
        x, y, visible = _scope_adjust(x, y, state)
        if not visible:
            continue
        fade = 1.0 - age / HITMARK_LIFE_S
        scale = HITMARK_CRIT_SCALE if crit else 1.0
        inner, outer = HITMARK_INNER_PX * scale, HITMARK_OUTER_PX * scale
        width, outline = HITMARK_WIDTH_PX * scale, HITMARK_OUTLINE_PX * scale
        core_col = HITMARK_CRIT_COL if crit else HITMARK_COL
        for col, half_w in ((HITMARK_OUTLINE_COL, (width + 2.0 * outline) * 0.5),
                            (core_col, width * 0.5)):
            colour = rgba(col, fade)
            for angle_deg in HITMARK_ANGLES_DEG:
                angle = math.radians(angle_deg)
                dx, dy = math.cos(angle), math.sin(angle)
                nx, ny = -dy, dx
                x0, y0 = x + dx * inner, y + dy * inner
                x1, y1 = x + dx * outer, y + dy * outer
                corners = ((x0 + nx * half_w, y0 + ny * half_w),
                          (x1 + nx * half_w, y1 + ny * half_w),
                          (x1 - nx * half_w, y1 - ny * half_w),
                          (x0 - nx * half_w, y0 - ny * half_w))
                _flat_quad(painter, corners, colour)


def _draw_crit_impact(painter, state) -> None:
    """TASK_game_feel_1118 P4: 0.05 s full-scene flash on a crit — placed
    (by ``draw_screen``'s call order) under the HUD and under the
    ``hit_crit`` Sakuga burst, so the burst pops against it."""
    if state.shot or state.crit_impact_alpha <= 0.0:
        return
    painter.fill((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1]),
                rgba(CRIT_IMPACT_COL, state.crit_impact_alpha))


def _scope_vignette_texture(painter):
    """A square radial-gradient RGBA texture — transparent inside
    ``VIGNETTE_RADIUS_PX``, full ``VIGNETTE_COL``/``VIGNETTE_ALPHA``
    outside it, with a soft edge in between. Built once per framebuffer
    scale and cached (same ``pil`` cache ``rounded_frame`` uses) — the
    crosshair it's centred on moves every frame, but the gradient itself
    never changes shape, so only the DRAW rect needs to move."""
    scale = painter.scale
    half_px = max(8, int(round((VIGNETTE_RADIUS_PX + VIGNETTE_SOFT_PX) * scale)))
    size_px = half_px * 2
    radius_px = VIGNETTE_RADIUS_PX * scale
    soft_px = max(1.0, VIGNETTE_SOFT_PX * scale)
    key = ("scope_vignette", size_px, round(radius_px, 1), round(soft_px, 1))

    def build():
        yy, xx = np.mgrid[0:size_px, 0:size_px].astype(np.float64)
        dist = np.hypot(xx - half_px + 0.5, yy - half_px + 0.5)
        falloff = np.clip((dist - radius_px) / soft_px, 0.0, 1.0)
        r, g, b, _a = rgba(VIGNETTE_COL)
        image = np.empty((size_px, size_px, 4), dtype=np.uint8)
        image[..., 0] = int(round(r * 255.0))
        image[..., 1] = int(round(g * 255.0))
        image[..., 2] = int(round(b * 255.0))
        image[..., 3] = (falloff * VIGNETTE_ALPHA * 255.0).astype(np.uint8)
        return image, None

    return painter.pil.get(key, build)


def _draw_scope_vignette(painter, state) -> None:
    """TASK_game_feel_1118 P6: outside a 288 design-px circle centred on
    the crosshair, an ``#0F1C2B`` wash at alpha 0.94 with a soft 6 px
    edge — placed (by ``draw_screen``'s call order) above the scene, under
    the HUD, same slot as the crit impact frame."""
    if (state.shot or state.aim is None
            or state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS):
        return
    cx, cy = float(state.aim[0]), float(state.aim[1])
    half = VIGNETTE_RADIUS_PX + VIGNETTE_SOFT_PX
    design_w, design_h = DESIGN_SIZE
    col = rgba(VIGNETTE_COL, VIGNETTE_ALPHA)
    top = max(0.0, cy - half)
    bottom = min(design_h, cy + half)
    if top > 0.0:
        painter.fill((0.0, 0.0, design_w, top), col)
    if bottom < design_h:
        painter.fill((0.0, bottom, design_w, design_h - bottom), col)
    left = max(0.0, cx - half)
    right = min(design_w, cx + half)
    band_h = bottom - top
    if left > 0.0:
        painter.fill((0.0, top, left, band_h), col)
    if right < design_w:
        painter.fill((right, top, design_w - right, band_h), col)
    entry = _scope_vignette_texture(painter)
    if entry is not None:
        size = half * 2.0
        painter.quad((cx - half, cy - half, size, size), entry[0])


def _draw_scope_reticle(painter, state) -> None:
    """TASK_game_feel_1118 P6: a thin 1.5 px cross with a centre gap, in
    ``#24415F`` — the scope's OWN reticle, drawn instead of the normal
    crosshair while scoped (``_draw_aim`` skips itself in that case)."""
    if (state.shot or state.aim is None
            or state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS):
        return
    x, y = float(state.aim[0]), float(state.aim[1])
    half = RETICLE_THICK_PX * 0.5
    colour = rgba(RETICLE_COL)
    painter.fill((x - RETICLE_ARM_PX, y - half,
                 RETICLE_ARM_PX - RETICLE_GAP_PX, RETICLE_THICK_PX), colour)
    painter.fill((x + RETICLE_GAP_PX, y - half,
                 RETICLE_ARM_PX - RETICLE_GAP_PX, RETICLE_THICK_PX), colour)
    painter.fill((x - half, y - RETICLE_ARM_PX,
                 RETICLE_THICK_PX, RETICLE_ARM_PX - RETICLE_GAP_PX), colour)
    painter.fill((x - half, y + RETICLE_GAP_PX,
                 RETICLE_THICK_PX, RETICLE_ARM_PX - RETICLE_GAP_PX), colour)


def _draw_rings(painter, manifest, screen, record, state) -> None:
    """Vòng hover và vòng focus — lớp cuối, sau mọi thứ (README mục 9).

    TASK_ui_sakuga_1117 Phase 5: chỉ còn một lệnh gọi mỏng vào
    ``v3_hover.HoverFx.draw`` — spring, màu/alpha theo hình thật
    (``rounded``/``pill``/``circle``/``underline``/``thumb``/``none``),
    con trỏ OS, và vòng focus bàn phím đều ở đó. Nhập trong hàm (không ở
    đầu file) để tránh vòng nhập: ``v3_hover`` đọc ngược lại các hằng/hàm
    của module này.
    """
    if state.shot or state.hover_fx is None:
        return
    state.hover_fx.draw(painter, manifest, screen, record, state)


def hit_test(manifest: V3Manifest, screen: str, design_x: float,
             design_y: float) -> str | None:
    """Tên hotspot dưới con trỏ; ``None`` nếu không có."""
    for hotspot in manifest.screens[screen].get("hotspots", ()):
        x, y, w, h = hotspot_rect(screen, hotspot)
        if x <= design_x <= x + w and y <= design_y <= y + h:
            return hotspot["n"]
    return None
