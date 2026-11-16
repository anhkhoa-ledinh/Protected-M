"""
Vẽ trọn một màn album từ bản kê: nền → ảnh → lớp phủ → ô động → khung chọn.

Ô động trong manifest có bốn kiểu: ``x`` chữ, ``r`` hình chữ nhật, ``e`` bầu
dục, ``b`` nhóm (vẽ ``children``). Kiểu ``svg`` là chỗ dành cho bọ — **trò chơi
tự vẽ**, ở đây bỏ qua.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ScreenState:
    """Mọi thứ thay đổi theo thời gian của một màn."""

    theme: str = "ngay"
    photo: Path | None = None
    values: dict = field(default_factory=dict)
    hidden: set = field(default_factory=set)
    selected: set = field(default_factory=set)
    hover: str | None = None
    disabled: set = field(default_factory=set)
    crosshair: tuple | None = None
    #: ``(x, y, tuổi giây, trúng?)`` của phát bắn gần nhất — vẽ chớp + vệt.
    shot: tuple | None = None
    #: Dòng báo cho người dùng, vẽ nổi ở giữa dưới.
    message: str = ""
    #: v2: ``<khoá>_track`` là rãnh 100 %; phần đầy rộng = track.w × tỉ lệ.
    tracks: dict = field(default_factory=dict)
    viewport_draw = None


def _style(record):
    return (record.get("ff") or "Inter", record.get("fw") or 400,
            record.get("fs") or 16)


def draw_record(painter, manifest, record, state, offset=(0.0, 0.0)) -> None:
    """Vẽ một ô động (hoặc một phần con của nó)."""
    kind = record.get("t")
    rect = list(record.get("rect", (0, 0, 0, 0)))
    rect[0] += offset[0]
    rect[1] += offset[1]
    theme = state.theme
    key = record.get("key", "")
    opacity = float(record.get("op", 1.0) or 1.0)
    if kind == "x":
        value = state.values.get(key, record.get("c"))
        # Màu cũng có thể sống: "● Đầu: tốt" xanh, "● Mất dấu đầu" đỏ.
        token = state.values.get(f"{key}:col", record.get("col"))
        colour = manifest.colour(token, theme)
        colour = (colour[0], colour[1], colour[2], colour[3] * opacity)
        painter.text(rect, value, _style(record), colour,
                     record.get("al") or "left")
    elif kind == "r":
        fraction = state.values.get(f"{key}:frac")
        if fraction is not None:
            # Bề rộng lấy từ RÃNH, không phải bề rộng mẫu của thanh: mẫu vẽ
            # sẵn ở 72 % nên "MÁU 100" trước đây chỉ đầy 72 %.
            full = state.tracks.get(key[:-4] if key.endswith("_bar") else key,
                                    rect[2])
            rect[2] = max(0.0, float(full) * float(fraction))
        if record.get("f"):
            fill = manifest.colour(record["f"], theme)
            painter.fill(rect, (fill[0], fill[1], fill[2], fill[3] * opacity))
        if record.get("s") and float(record.get("sw", 0) or 0) > 0:
            painter.frame(rect, manifest.colour(record["s"], theme),
                          float(record["sw"]))
    elif kind == "e":
        fill = manifest.colour(record["f"], theme) if record.get("f") else None
        stroke = manifest.colour(record["s"], theme) if record.get("s") else None
        painter.ellipse(rect, fill, stroke, float(record.get("sw", 2) or 2))
    elif kind == "b":
        if record.get("f"):
            fill = manifest.colour(record["f"], theme)
            painter.fill(rect, (fill[0], fill[1], fill[2], fill[3] * opacity))
        if record.get("s") and float(record.get("sw", 0) or 0) > 0:
            painter.frame(rect, manifest.colour(record["s"], theme),
                          float(record["sw"]))
        # Mỗi phần con có khoá riêng ``<khoá cha>#<số thứ tự>``: nhóm hai dòng
        # (vd. photo_caption) nếu dùng chung khoá cha sẽ in CÙNG một giá trị
        # lên cả hai dòng — thấy rõ ở ảnh chụp đầu tiên của màn chơi.
        for index, child in enumerate(record.get("children", [])):
            child = dict(child)
            child["key"] = f"{key}#{index}"
            if child["key"] in state.hidden:
                continue
            draw_record(painter, manifest, child, state, offset)


def track_widths(screen) -> dict:
    """``{tên thanh: bề rộng rãnh}`` lấy từ các ô ``<khoá>_track`` (v2)."""
    return {record["key"][:-6]: record["rect"][2]
            for record in screen.get("dyn", [])
            if str(record.get("key", "")).endswith("_track")}


def draw_dyn(painter, manifest, screen, state) -> None:
    """Mọi ô động của màn, trừ những khoá đang ẩn."""
    groups: dict[str, list] = {}
    for record in screen.get("dyn", []):
        groups.setdefault(record.get("key", ""), []).append(record)
    for key, records in groups.items():
        if key in state.hidden or key == "bugs":
            continue
        offset = (0.0, 0.0)
        if key == "crosshair":
            if state.crosshair is None:
                continue
            xs = [r["rect"][0] + r["rect"][2] * 0.5 for r in records]
            ys = [r["rect"][1] + r["rect"][3] * 0.5 for r in records]
            centre = (sum(xs) / len(xs), sum(ys) / len(ys))
            offset = (state.crosshair[0] - centre[0],
                      state.crosshair[1] - centre[1])
        for record in records:
            draw_record(painter, manifest, record, state, offset)


def draw_screen(painter, manifest, name: str, state: ScreenState) -> None:
    """Vẽ màn ``name`` theo đúng thứ tự lớp của README."""
    screen = manifest.screens[name]
    if not state.tracks:
        state.tracks = track_widths(screen)
    design = manifest.size_of(name)
    full = (0.0, 0.0, float(design[0]), float(design[1]))
    painter.fill(full, manifest.colour("bg", state.theme))
    painter.image(full, manifest.layer_path(name, "base", state.theme))
    for photo in screen.get("photos", []):
        rect = photo["rect"]
        if photo.get("kind") == "viewport":
            if state.viewport_draw is not None:
                state.viewport_draw(rect)
            continue
        source = state.photo or manifest.image(manifest.data.get("sample_photo"))
        if source is not None and Path(source).exists():
            painter.cover(rect, source)
    painter.image(full, manifest.layer_path(name, "top", state.theme))
    draw_dyn(painter, manifest, screen, state)
    for hotspot in screen.get("hotspots", []):
        if hotspot["n"] in state.disabled:
            dim = manifest.colour("bg", state.theme)
            painter.fill(hotspot["rect"], (dim[0], dim[1], dim[2], 0.5))
    draw_shot(painter, manifest, state)
    if state.message:
        box = (240.0, 620.0, 800.0, 48.0)
        tint = manifest.colour("surface", state.theme)
        painter.fill(box, (tint[0], tint[1], tint[2], 0.96))
        painter.frame(box, manifest.colour("danger", state.theme), 2.0)
        painter.text((box[0] + 16, box[1] + 12, box[2] - 32, 24.0),
                     state.message, ("Inter", 600, 16),
                     manifest.colour("text", state.theme))
    primary = manifest.colour("primary", state.theme)
    for hotspot in screen.get("hotspots", []):
        if hotspot["n"] in state.selected:
            painter.frame(hotspot["rect"], primary)
        elif hotspot["n"] == state.hover:
            painter.frame(hotspot["rect"],
                          (primary[0], primary[1], primary[2], 0.45))


#: Chớp bắn sống ngần này giây.
SHOT_FLASH_S = 0.18


def draw_shot(painter, manifest, state) -> None:
    """
    Chớp đầu nòng + vòng loang ở chỗ vừa bắn.

    Người dùng 06/11: "thiếu cảm giác bắn viên đạn". Vòng loang to dần và mờ
    dần trong 0,18 s, màu ``success`` khi trúng và ``danger`` khi hụt.
    """
    if state.shot is None:
        return
    x, y, age, hit = state.shot
    if age > SHOT_FLASH_S:
        return
    k = 1.0 - age / SHOT_FLASH_S
    colour = manifest.colour("success" if hit else "danger", state.theme)
    ring = 18.0 + 46.0 * (1.0 - k)
    painter.ellipse((x - ring * 0.5, y - ring * 0.5, ring, ring), None,
                    (colour[0], colour[1], colour[2], k), 3.0)
    core = 10.0 * k
    painter.ellipse((x - core * 0.5, y - core * 0.5, core, core),
                    (colour[0], colour[1], colour[2], k), None, 1.0)


def hit_test(manifest, name: str, state, design_x: float, design_y: float):
    """Tên hotspot (hoặc ``text:<chuỗi>``) dưới con trỏ; ``None`` nếu không."""
    screen = manifest.screens[name]
    for hotspot in screen.get("hotspots", []):
        x, y, w, h = hotspot["rect"]
        if x <= design_x <= x + w and y <= design_y <= y + h:
            return hotspot["n"]
    for text in screen.get("texts", []):
        x, y, w, h = text["rect"]
        if x <= design_x <= x + w and y <= design_y <= y + h:
            return f"text:{text['c']}"
    return None
