"""
Trò chơi tìm vật ẩn — PB-015 phần 2, phần KHÔNG dính OpenGL.

Bản ghi phẳng + hàm thuần (RULES §4: không class dùng một lần). Mọi hình học
lớp đi vào đây là ``LayerPlane`` đọc từ đường vẽ thật của renderer, nên module
này không tự dựng lại công thức overscan hay relief shift nào.

Mắt nghỉ KHÔNG còn là hằng số: ``G`` đo tại chỗ (xem ``prop_rest_eye``) rồi
truyền vào đây, nên ``start_game`` nhận cả ``z_rest``.
"""

from __future__ import annotations

import csv
import time
from dataclasses import dataclass, field
from pathlib import Path

from src.content.hidden_props import _nearer, is_visible
from src.content.hidden_props_search import place_props

#: Số vật mỗi ván. Giữ là hằng module, chưa thành khoá config (ngoài phạm vi C2).
PROP_COUNT = 3

#: Thấy liên tục ngần này giây thì tính là tìm ra.
FOUND_HOLD_S = 0.5

#: Màu vật lúc chưa tìm thấy và lúc đã tìm thấy.
COLOR_HIDDEN = (0.95, 0.72, 0.18)
COLOR_FOUND = (0.25, 0.9, 0.45)

CSV_HEADER = ("timestamp", "image", "found", "total", "seconds", "tracking",
              "dd_mm", "rest_x_mm", "rest_y_mm", "rest_z_mm",
              "spread_x_mm", "spread_y_mm", "end_reason")


@dataclass
class GameState:
    """Một ván chơi. Bản ghi phẳng; mọi phép biến đổi nằm ở hàm module."""

    image: str
    props: list[dict]
    tracking_on: bool
    started_at_s: float
    found: list[bool] = field(default_factory=list)
    visible_since_s: list[float | None] = field(default_factory=list)
    finished_at_s: float | None = None

    def __post_init__(self) -> None:
        # Hai danh sách song song với ``props``; để ở đây nên mọi đường tạo
        # state đều có chúng đúng độ dài, kể cả test tự dựng.
        if not self.found:
            self.found = [False] * len(self.props)
        if not self.visible_since_s:
            self.visible_since_s = [None] * len(self.props)


def start_game(planes, radius_mm, rest_eye_mm, z_rest_mm, now_s, image,
               *, tracking_on=True, count=PROP_COUNT):
    """
    Đặt vật rồi mở một ván. Trả state với ``k ≤ count`` vật, có thể rỗng.

    Không bao giờ lỗi với ảnh không giấu được vật nào — người gọi xem
    ``len(state.props)`` để quyết định toast nào (C1 "any image").

    ``count`` mặc định bằng ``PROP_COUNT`` nên mọi chỗ gọi cũ không đổi gì;
    giao diện album truyền số bọ của từng cấp (TASK_album_ui_1106).
    """
    props = place_props(planes, int(count), float(radius_mm), tuple(rest_eye_mm),
                        z_rest=float(z_rest_mm))
    return GameState(image=str(image), props=list(props),
                     tracking_on=bool(tracking_on), started_at_s=float(now_s))


def update(state, eye_mm, planes, now_s, radius_mm, auto_find: bool = True):
    """
    Cập nhật đồng hồ "thấy liên tục"; trả chỉ số các vật VỪA tìm ra khung này.

    ``auto_find=False``: vẫn theo dõi vật nào đang lộ, nhưng **không** tự tính
    là tìm ra. Giao diện album dùng chế độ này — ở đó phải **bắn** mới diệt
    được (người dùng 06/11: "chỉ cần nghiêng đầu bọ tự bị bắt").

    Đây là việc duy nhất chạy mỗi khung. ``is_visible`` kiểm mặt phẳng phẳng ở
    z đã shift và bỏ qua relief, nên có thể báo "thấy" sớm hơn mắt người —
    mỗi lần hẹn giờ xong có một dòng console để chủ sở hữu đối chiếu.
    """
    newly = []
    for i, prop in enumerate(state.props):
        if state.found[i]:
            continue
        near = _nearer(planes, prop["pos_mm"][2] / 1000.0)
        if not is_visible(eye_mm, prop["pos_mm"], float(radius_mm), near):
            state.visible_since_s[i] = None
            continue
        if state.visible_since_s[i] is None:
            state.visible_since_s[i] = float(now_s)
        elif auto_find and now_s - state.visible_since_s[i] >= FOUND_HOLD_S:
            state.found[i] = True
            newly.append(i)
    if state.props and all(state.found) and state.finished_at_s is None:
        state.finished_at_s = float(now_s)
    return newly


def is_exposed(state, index: int) -> bool:
    """Vật thứ ``index`` có đang lộ (nhìn thấy được) không."""
    return (0 <= index < len(state.props)
            and not state.found[index]
            and state.visible_since_s[index] is not None)


def csv_row(state, now_s, dd_mm, rest_eye_mm, spread_xy_mm, end_reason):
    """Một dòng kết quả; ``end_reason``: finished / G / new_image / geometry / exit."""
    end = state.finished_at_s if state.finished_at_s is not None else float(now_s)
    return (time.strftime("%Y-%m-%dT%H:%M:%S"), state.image,
            sum(state.found), len(state.props), f"{end - state.started_at_s:.1f}",
            "ON" if state.tracking_on else "OFF", f"{float(dd_mm):.0f}",
            f"{rest_eye_mm[0]:.0f}", f"{rest_eye_mm[1]:.0f}", f"{rest_eye_mm[2]:.0f}",
            f"{spread_xy_mm[0]:.0f}", f"{spread_xy_mm[1]:.0f}", str(end_reason))


def append_csv(path, row):
    """Thêm một dòng, tự tạo thư mục và dòng tiêu đề nếu file chưa có."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    new = not target.exists()
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new:
            writer.writerow(CSV_HEADER)
        writer.writerow(row)
    return target


def prop_items(state, radius_mm):
    """
    Vật dưới dạng renderer cần: ``(x_m, y_m, z_m, radius_m, color)``.

    ``place_props`` trả vị trí bằng mm trên hệ mặt kính (z âm = sau kính), cùng
    hệ mm với world vì ``screen.json`` đặt ``screen_center = [0, 0, 0]``. Khoá
    ``color`` chỉ có ở vật thử chẩn đoán; vật thật dùng màu theo trạng thái.
    """
    r_m = float(radius_mm) / 1000.0
    out = []
    for prop, found in zip(state.props, state.found):
        x, y, z = prop["pos_mm"]
        color = prop.get("color") or (COLOR_FOUND if found else COLOR_HIDDEN)
        out.append((x / 1000.0, y / 1000.0, z / 1000.0, r_m, color))
    return out


def status_line(state):
    """Dòng log một ván vừa mở."""
    if not state.props:
        return f"Prop game [{state.image}]: 0 vat — anh nay khong giau duoc vat nao"
    parts = []
    for i, prop in enumerate(state.props):
        x, y, z = prop["pos_mm"]
        parts.append(
            f"#{i} ({x:+.0f},{y:+.0f},{z:+.0f})mm truc={prop['reveal_axis']} "
            f"lo={prop['reveal_450_mm']:+.0f}mm"
        )
    return (f"Prop game [{state.image}]: {len(state.props)} vat, "
            f"tracking={'ON' if state.tracking_on else 'OFF'} | " + "  ".join(parts))
