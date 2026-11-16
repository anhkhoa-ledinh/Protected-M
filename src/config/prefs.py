"""
Tuỳ chọn riêng từng máy: ``assets/config/user_prefs.json`` (D1).

Chỉ hai khoá, và cả hai đều là chuyện của **cái màn hình đang ngồi trước**:
chế độ hiển thị, và khung cửa sổ lần trước. Vì vậy file này nằm trong
``.gitignore`` — commit lên thì người khác clone về sẽ mở cửa sổ ở toạ độ của
máy mình, có khi lọt hẳn ra ngoài màn hình của họ (đúng lý do ``user.json``
cũng bị loại).

Đọc lỗi thì trả mặc định và ghi log; **không** ném lỗi: một file JSON gãy
không được phép làm app không mở lên được.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

PREFS_NAME = "user_prefs.json"

FULLSCREEN, WINDOW = "fullscreen", "window"

#: Cửa sổ mặc định và trần dưới (D1.4).
DEFAULT_WINDOW = (1280, 720)
MIN_WINDOW = (960, 540)
ASPECT = (16, 9)

#: Màn hình nhỏ hơn cửa sổ mặc định thì lấy 80 % bề rộng màn, giữ 16:9.
SMALL_SCREEN_RATIO = 0.8


def prefs_path(root: Path) -> Path:
    return Path(root) / "assets" / "config" / PREFS_NAME


def load(root: Path) -> dict:
    """``{"display": ..., "window_rect": [x, y, w, h] | None}``."""
    path = prefs_path(root)
    out = {"display": FULLSCREEN, "window_rect": None}
    if not path.exists():
        return out
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as err:
        logger.warning("user_prefs.json khong doc duoc (%s) — dung mac dinh", err)
        return out
    if not isinstance(payload, dict):
        return out
    mode = str(payload.get("display", FULLSCREEN))
    out["display"] = mode if mode in (FULLSCREEN, WINDOW) else FULLSCREEN
    rect = payload.get("window_rect")
    if isinstance(rect, (list, tuple)) and len(rect) == 4:
        try:
            out["window_rect"] = [int(v) for v in rect]
        except (TypeError, ValueError):
            out["window_rect"] = None
    return out


def save(root: Path, display: str, window_rect=None) -> None:
    """Ghi lại tuỳ chọn. Lỗi ghi chỉ log, không làm hỏng lượt chạy."""
    path = prefs_path(root)
    payload = {"display": display if display in (FULLSCREEN, WINDOW) else FULLSCREEN,
               "window_rect": list(window_rect) if window_rect else None}
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    except OSError as err:
        logger.warning("Khong ghi duoc %s (%s)", path, err)


def default_rect(monitor_pos, monitor_size) -> list[int]:
    """
    Khung cửa sổ mặc định: 1280×720 canh giữa màn.

    Màn hẹp hơn thì lấy 80 % bề rộng **và giữ đúng 16:9** — tính chiều cao từ
    bề rộng chứ không co riêng từng chiều, vì toàn bộ không gian thiết kế là
    16:9 và lệch tỉ lệ sẽ sinh hai dải letterbox ngay trong cửa sổ.
    """
    mx, my = int(monitor_pos[0]), int(monitor_pos[1])
    mw, mh = int(monitor_size[0]), int(monitor_size[1])
    width, height = DEFAULT_WINDOW
    if mw < width or mh < height:
        width = max(MIN_WINDOW[0], int(mw * SMALL_SCREEN_RATIO))
        height = int(round(width * ASPECT[1] / ASPECT[0]))
        if height > mh * SMALL_SCREEN_RATIO:
            height = max(MIN_WINDOW[1], int(mh * SMALL_SCREEN_RATIO))
            width = int(round(height * ASPECT[0] / ASPECT[1]))
    return [mx + (mw - width) // 2, my + (mh - height) // 2, width, height]
