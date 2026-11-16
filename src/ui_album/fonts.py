"""
Chữ cho giao diện album: PIL vẽ ra ảnh RGBA, ``textures.py`` đưa lên GPU.

Vì sao PIL chứ không phải ImGui: venv **không có** `imgui` và luật của kịch
bản cấm cài thêm. PIL đã có sẵn, dựng chữ tiếng Việt đầy đủ dấu (không cần
khai dải glyph như ImGui) và cho ra đúng một texture RGBA — vừa khớp đường vẽ
quad đang dùng cho HUD/toast.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger(__name__)

#: Cỡ chữ nhỏ cho phần UI tự vẽ (README: thêm 15 và 18 ngoài bảng kiểu).
EXTRA_SIZES = (15, 18)


class FontBook:
    """Kho `ImageFont` theo (họ, độ đậm, cỡ); nạp một lần, dùng lại."""

    def __init__(self, manifest) -> None:
        self._root = manifest.root
        self._files = manifest.fonts.get("files", {})
        self._cache: dict[tuple[str, int, int], ImageFont.FreeTypeFont] = {}
        self._fallback = ImageFont.load_default()
        styles = manifest.fonts.get("dynamic_text_styles", [])
        for style in styles:
            self.get(style["family"], style["weight"], style["size"])
        for size in EXTRA_SIZES:
            self.get("Inter", 500, size)
        logger.info("Album fonts: nap san %d kieu", len(self._cache))

    def _file_for(self, family: str, weight: int) -> Path | None:
        table = self._files.get(family) or self._files.get("Inter") or {}
        if not table:
            return None
        key = str(int(weight))
        if key not in table:
            # Gần nhất theo số, để kiểu lạ vẫn ra đúng họ chữ.
            key = min(table, key=lambda k: abs(int(k) - int(weight)))
        return self._root / table[key]

    def get(self, family, weight, size) -> ImageFont.FreeTypeFont:
        """Font đã nạp; thiếu file thì trả font mặc định của PIL."""
        family = str(family or "Inter")
        weight = int(weight or 400)
        size = max(6, int(round(size or 16)))
        key = (family, weight, size)
        if key in self._cache:
            return self._cache[key]
        path = self._file_for(family, weight)
        try:
            font = ImageFont.truetype(str(path), size)
        except (OSError, TypeError) as err:
            logger.warning("Album: khong nap duoc font %s %s (%s)",
                           family, weight, err)
            font = self._fallback
        self._cache[key] = font
        return font


def render_text(book: FontBook, text: str, style, rgba) -> np.ndarray | None:
    """
    Vẽ ``text`` (có thể nhiều dòng) ra mảng RGBA. ``None`` khi chuỗi rỗng.

    ``style`` = ``(họ, độ đậm, cỡ)``; ``rgba`` là 4 số 0–1.
    """
    text = "" if text is None else str(text)
    if not text:
        return None
    font = book.get(*style)
    lines = text.splitlines() or [text]
    ascent, descent = font.getmetrics() if hasattr(font, "getmetrics") else (
        int(style[2]), int(style[2] * 0.25))
    line_h = ascent + descent
    spacing = max(2, int(style[2] * 0.18))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    widths = [probe.textlength(line or " ", font=font) for line in lines]
    width = max(1, int(round(max(widths))) + 4)
    height = max(1, line_h * len(lines) + spacing * (len(lines) - 1))
    image = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    colour = tuple(int(round(c * 255)) for c in rgba)
    for index, line in enumerate(lines):
        # neo "la" = mép TRÊN của chữ, khớp với rect trong manifest (bbox chữ).
        draw.text((2, index * (line_h + spacing)), line, font=font,
                  fill=colour, anchor="la")
    return np.asarray(image, dtype=np.uint8)
