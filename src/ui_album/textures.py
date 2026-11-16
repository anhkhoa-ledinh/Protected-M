"""
Texture cho giao diện album: PIL/numpy → OpenGL, có giới hạn bộ nhớ.

Ảnh nền một màn là 1920×1080 RGBA ≈ 8 MB trên GPU, nên bộ đệm ảnh giữ **tối
đa 6 texture** (README) theo lối LRU. Chữ và hình khối nhỏ hơn nhiều nên có
bộ đệm riêng, rộng hơn.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import numpy as np
from OpenGL import GL
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_CACHE_MAX = 6
SMALL_CACHE_MAX = 256


def upload_rgba(array: np.ndarray) -> int:
    """Mảng RGBA uint8 → tên texture GL (lọc tuyến tính, kẹp biên)."""
    height, width = array.shape[:2]
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                    np.ascontiguousarray(array))
    return int(texture)


#: Cạnh dài tối đa khi nạp ảnh nền. Ảnh xuất ở 1920×1080 nhưng cửa sổ chỉ
#: 1366×768, nên thu về trước khi lên GPU: RAM mỗi texture 8,3 → 4,2 MB và
#: bớt cả fillrate. Một lượt chụp đã **chết vì MemoryError** đúng ở chỗ đọc
#: ảnh này khi máy hết commit (22:30 ngày 06/11).
MAX_IMAGE_WIDTH = 1366


def load_rgba(path, max_width: int = MAX_IMAGE_WIDTH) -> np.ndarray | None:
    """Đọc PNG thành mảng RGBA (thu về ``max_width``); ``None`` nếu hỏng."""
    try:
        with Image.open(path) as image:
            if max_width and image.width > max_width:
                height = max(1, round(image.height * max_width / image.width))
                image = image.resize((max_width, height), Image.LANCZOS)
            return np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (OSError, ValueError, MemoryError) as err:
        logger.warning("Album: khong doc duoc %s (%s)", path, err)
        return None


class TextureCache:
    """LRU hai tầng: ảnh nền (ít, nặng) và hình nhỏ/chữ (nhiều, nhẹ)."""

    def __init__(self) -> None:
        self._images: OrderedDict[str, tuple[int, int, int]] = OrderedDict()
        self._small: OrderedDict[tuple, tuple[int, int, int]] = OrderedDict()
        self.loads = 0

    def _evict(self, store: OrderedDict, limit: int) -> None:
        while len(store) > limit:
            _key, (texture, _w, _h) = store.popitem(last=False)
            GL.glDeleteTextures(1, [texture])

    def image(self, path) -> tuple[int, int, int] | None:
        """``(texture, w, h)`` của một file PNG; nạp khi cần."""
        if path is None:
            return None
        key = str(path)
        if key in self._images:
            self._images.move_to_end(key)
            return self._images[key]
        array = load_rgba(path)
        if array is None:
            return None
        entry = (upload_rgba(array), array.shape[1], array.shape[0])
        self._images[key] = entry
        self.loads += 1
        self._evict(self._images, IMAGE_CACHE_MAX)
        return entry

    def small(self, key: tuple, build) -> tuple[int, int, int] | None:
        """``(texture, w, h)`` dựng từ ``build()`` (mảng RGBA), có đệm."""
        if key in self._small:
            self._small.move_to_end(key)
            return self._small[key]
        array = build()
        if array is None:
            return None
        entry = (upload_rgba(array), array.shape[1], array.shape[0])
        self._small[key] = entry
        self._evict(self._small, SMALL_CACHE_MAX)
        return entry

    def white(self) -> tuple[int, int, int]:
        """Texture 1×1 trắng — nền cho mọi hình tô màu phẳng."""
        return self.small(("white",),
                          lambda: np.full((1, 1, 4), 255, np.uint8))

    def release(self) -> None:
        for store in (self._images, self._small):
            for texture, _w, _h in store.values():
                GL.glDeleteTextures(1, [texture])
            store.clear()
