"""
Hotspot geometry and hit-testing for the v3 UI (TASK_arch_refactor_1120 R4:
split out of ``src.ui_album.v3``, which had grown to 2479 lines).

Hover/focus ring shape per hotspot name, corner-radius measurement helpers
(used offline by ``tools/`` to measure real art), the ``LINK_HIT_RECTS``
override table, and ``hit_test`` itself (screen + point -> hotspot name).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from src.ui_album.v3 import V3Manifest

#: Design-space canvas size (``src.ui_album.v3``'s own value, duplicated
#: here — same already-established pattern as ``fx/library.py``'s
#: DESIGN_SIZE from TASK_arch_refactor_1120 R2 — so this module does not
#: need to import anything back from ``v3.py``).
DESIGN_SIZE = (1280.0, 720.0)

#: TASK_bug_art_0530: bán kính bo góc của field_user/field_pass, đo trực tiếp
#: trên ``v3_01a_sign_in/base.png`` (quét cột đầu mép bo ở mỗi hàng, bán kính
#: = khoảng cách từ mép tới điểm offset về 0 — ra ~22 px ở ảnh ×1,5, tức 15 px
#: không gian thiết kế). Không có trường bán kính nào trong manifest.
FIELD_CORNER_RADIUS_PX = 15.0

BTN_CORNER_RADIUS_PX = 12.0
ROW_CORNER_RADIUS_PX = 18.0

#: Số đo (thiết kế px, làm tròn — báo cáo P2 có số thật): xem lịch sử của
#: ``src/ui_album/v3.py`` cho chú thích đầy đủ từng con số.
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


def hit_test(manifest: V3Manifest, screen: str, design_x: float,
             design_y: float) -> str | None:
    """Tên hotspot dưới con trỏ; ``None`` nếu không có."""
    for hotspot in manifest.screens[screen].get("hotspots", ()):
        x, y, w, h = hotspot_rect(screen, hotspot)
        if x <= design_x <= x + w and y <= design_y <= y + h:
            return hotspot["n"]
    return None
