"""
Nạp và kiểm `assets/ui_album/manifest.json` — hợp đồng của giao diện album.

Mọi toạ độ trong manifest là **không gian thiết kế** 1280×720 (bảng thành phần
1280×960). Hàm ``letterbox`` cho phép đổi sang pixel cửa sổ: giữ tỉ lệ, canh
giữa, hai dải thừa tô bằng token ``bg``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Thư mục asset, tính từ gốc project.
ASSET_DIR = Path("assets") / "ui_album"

#: Màn tham chiếu, không phải màn chạy thật (README).
REFERENCE_SCREEN = "moi_00_thanh_phan"


@dataclass(frozen=True)
class Letterbox:
    """Phép biến đổi thiết kế → pixel cửa sổ: nhân ``scale``, cộng gốc."""

    scale: float
    ox: float
    oy: float

    def rect(self, r) -> tuple[float, float, float, float]:
        """``[x, y, w, h]`` thiết kế → pixel cửa sổ."""
        return (self.ox + r[0] * self.scale, self.oy + r[1] * self.scale,
                r[2] * self.scale, r[3] * self.scale)

    def unmap(self, x: float, y: float) -> tuple[float, float]:
        """Pixel cửa sổ → toạ độ thiết kế (để bắt chuột)."""
        return ((x - self.ox) / self.scale, (y - self.oy) / self.scale)

    def map(self, x: float, y: float) -> tuple[float, float]:
        """Toạ độ thiết kế → pixel cửa sổ — nghịch đảo của ``unmap``
        (TASK_game_feel_1118 P6: dời con trỏ HĐH về tâm ngắm lúc hạ ống)."""
        return (self.ox + x * self.scale, self.oy + y * self.scale)


def letterbox(design, window) -> Letterbox:
    """Khung thiết kế lọt giữa cửa sổ, giữ tỉ lệ."""
    scale = min(window[0] / float(design[0]), window[1] / float(design[1]))
    return Letterbox(scale, (window[0] - design[0] * scale) * 0.5,
                     (window[1] - design[1] * scale) * 0.5)


class AlbumManifest:
    """Bản kê đã nạp; truy cập màn, token và đường dẫn file."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / ASSET_DIR
        path = self.root / "manifest.json"
        self.data = json.loads(path.read_text(encoding="utf-8"))
        self.design_size = tuple(self.data.get("design_size", (1280, 720)))
        self.screens = self.data["screens"]
        self.tokens = self.data["tokens"]
        self.fonts = self.data["fonts"]
        self.sprites = self.data.get("sprites", {})
        missing = self.validate()
        logger.info("Album manifest: %d man, %d token, thieu %d file",
                    len(self.screens), len(self.tokens.get("ngay", {})),
                    len(missing))
        if missing:
            logger.warning("Album: thieu file %s", missing[:5])

    def screen_names(self, include_reference: bool = False) -> list[str]:
        """Tên mọi màn chạy thật, theo thứ tự trong manifest."""
        return [n for n in self.screens
                if include_reference or n != REFERENCE_SCREEN]

    def size_of(self, screen: str) -> tuple[int, int]:
        return tuple(self.screens[screen].get("size", self.design_size))

    def colour(self, token: str | None, theme: str, fallback="#FF00FF"):
        """Token màu → ``(r, g, b, a)`` 0–1. Token lạ thì trả màu báo lỗi."""
        table = self.tokens.get(theme, {})
        hexa = table.get(str(token), fallback)
        hexa = hexa.lstrip("#")
        if len(hexa) == 6:
            hexa += "FF"
        return tuple(int(hexa[i:i + 2], 16) / 255.0 for i in (0, 2, 4, 6))

    def image(self, relative: str | None) -> Path | None:
        """Đường dẫn tuyệt đối của một file trong bộ asset."""
        if not relative:
            return None
        return self.root / relative

    def layer_path(self, screen: str, layer: str, theme: str) -> Path | None:
        """File ``base``/``top`` của một màn theo theme."""
        record = self.screens[screen].get(layer) or {}
        return self.image(record.get(theme) or record.get("ngay"))

    def validate(self) -> list[str]:
        """Danh sách file PNG khai trong manifest mà không có trên đĩa."""
        missing = []
        for name in self.screens:
            for layer in ("base", "top"):
                for theme in ("ngay", "dem"):
                    path = self.layer_path(name, layer, theme)
                    if path is not None and not path.exists():
                        missing.append(str(path.relative_to(self.root)))
        for sprite in self.sprites.values():
            path = self.image(sprite.get("file"))
            if path is not None and not path.exists():
                missing.append(sprite.get("file"))
        return missing
