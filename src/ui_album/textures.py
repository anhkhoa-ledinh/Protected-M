"""
Texture cho giao diện album: PIL/numpy → OpenGL, có giới hạn bộ nhớ.

Ảnh nền một màn là 1920×1080 RGBA ≈ 8 MB trên GPU, nên bộ đệm ảnh giữ **tối
đa 6 texture** (README) theo lối LRU. Chữ và hình khối nhỏ hơn nhiều nên có
bộ đệm riêng, rộng hơn.
"""

from __future__ import annotations

import logging
from collections import OrderedDict

import cv2
import numpy as np
from OpenGL import GL
from PIL import Image

logger = logging.getLogger(__name__)

IMAGE_CACHE_MAX = 6
SMALL_CACHE_MAX = 256

#: TASK_game_feel_1118 P2: số lần lặp box-blur của ``bleed_transparent`` —
#: gần giống Gauss (rẻ hơn, ``cv2.blur`` là box-filter thuần), loang màu
#: ra xa dần theo số lần.
BLEED_ITERATIONS = 6
BLEED_KERNEL = 3


def bleed_transparent(array: np.ndarray, iterations: int = BLEED_ITERATIONS,
                      ksize: int = BLEED_KERNEL) -> np.ndarray:
    """
    Loang màu vào các điểm TRONG SUỐT HOÀN TOÀN (alpha=0) trước khi lên
    mipmap.

    PNG alpha thẳng (không premultiply) thường giữ RGB=đen ở chỗ alpha=0
    (không ai nhìn thấy màu đó — cho tới khi mipmap TRUNG BÌNH CỘNG nó
    vào viền, ra quầng tối quanh hình). Cách: premultiply (``RGB·a``),
    làm mờ CẢ ``RGB·a`` LẪN ``a`` bằng cùng một nhân hộp lặp
    ``iterations`` lần, rồi ở những điểm ``a=0`` gán
    ``RGB = blur(RGB·a) / max(blur(a), eps)`` — màu trung bình có trọng
    số của các điểm KHÔNG trong suốt gần đó. Alpha giữ NGUYÊN — chỉ RGB
    của điểm alpha=0 đổi.
    """
    if array.ndim != 3 or array.shape[2] != 4:
        return array
    alpha = array[..., 3]
    transparent = alpha == 0
    if not np.any(transparent):
        return array           # đặc hoàn toàn (hoặc không điểm nào a=0): không có gì để loang
    # TASK_game_feel_1118 P3: ``dst=`` dùng LẠI đúng bộ đệm mỗi lần lặp
    # (``cv2.blur`` mặc định cấp một mảng MỚI mỗi lần gọi) — atlas lớn
    # nhất (hit_kill, 1987x3548) tốn ~80 MB một bản float32 RGB; không
    # dùng lại thì N lần lặp = N bản sống cùng lúc, dễ ``MemoryError``
    # khi vài atlas lớn giải mã song song (đo được khi chạy test).
    a = alpha.astype(np.float32)
    a *= 1.0 / 255.0
    rgb_pm = array[..., :3].astype(np.float32)
    rgb_pm *= a[..., None]
    for _ in range(max(1, iterations)):
        rgb_pm = cv2.blur(rgb_pm, (ksize, ksize), dst=rgb_pm)
        a = cv2.blur(a, (ksize, ksize), dst=a)
    np.divide(rgb_pm, np.maximum(a[..., None], 1e-6), out=rgb_pm)
    np.clip(rgb_pm, 0.0, 255.0, out=rgb_pm)
    out = array.copy()
    # Index BEFORE converting dtype — ``rgb_pm[transparent]`` is only as
    # big as the transparent REGION; ``rgb_pm.astype(uint8)[transparent]``
    # would convert and hold the WHOLE image first (a 1920x1080 UI
    # background mostly-opaque-but-not-entirely blew well past 15 MiB
    # just for that one throwaway copy, observed under full-suite memory
    # pressure).
    out[..., :3][transparent] = rgb_pm[transparent].astype(np.uint8)
    return out


def upload_rgba(array: np.ndarray, *, mipmaps: bool = True,
                max_level: int | None = None) -> int:
    """
    Mảng RGBA uint8 → tên texture GL (kẹp biên).

    TASK_game_feel_1118 P2: mặc định BẬT mipmap
    (``GL_LINEAR_MIPMAP_LINEAR`` + ``glGenerateMipmap``) — không texture
    nào trong app trước đây dùng mipmap, và ảnh gốc luôn lớn hơn cỡ hiển
    thị thật (UI 1366px thu 1,35×, sprite/fx 1,5× thu ~1,9× ở cửa sổ
    1012px), nên viền mỏng/góc bo alias rõ. ``max_level`` giới hạn số bậc
    mip (cho atlas fx: khung cách nhau 2px, bậc cao hơn sẽ loang màu
    SANG khung bên cạnh).
    """
    height, width = array.shape[:2]
    texture = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
    min_filter = GL.GL_LINEAR_MIPMAP_LINEAR if mipmaps else GL.GL_LINEAR
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, min_filter)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
    GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
    if max_level is not None:
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAX_LEVEL, int(max_level))
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, width, height, 0,
                    GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                    np.ascontiguousarray(array))
    if mipmaps:
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
    return int(texture)


#: Cạnh dài tối đa khi nạp ảnh nền. Ảnh xuất ở 1920×1080 nhưng cửa sổ chỉ
#: 1366×768, nên thu về trước khi lên GPU: RAM mỗi texture 8,3 → 4,2 MB và
#: bớt cả fillrate. Một lượt chụp đã **chết vì MemoryError** đúng ở chỗ đọc
#: ảnh này khi máy hết commit (22:30 ngày 06/11).
MAX_IMAGE_WIDTH = 1366


def load_rgba(path, max_width: int = MAX_IMAGE_WIDTH) -> np.ndarray | None:
    """
    Đọc PNG thành mảng RGBA (thu về ``max_width``); ``None`` nếu hỏng.

    TASK_game_feel_1118 P2: loang màu (``bleed_transparent``) TRƯỚC khi
    thu nhỏ, không phải sau — ``Image.resize`` không premultiply alpha,
    nên tự nó đã có đúng lỗi quầng đen y hệt mipmap nếu điểm RGB=đen còn
    nguyên lúc resize.
    """
    try:
        with Image.open(path) as image:
            array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (OSError, ValueError, MemoryError) as err:
        logger.warning("Album: khong doc duoc %s (%s)", path, err)
        return None
    array = bleed_transparent(array)
    if max_width and array.shape[1] > max_width:
        height = max(1, round(array.shape[0] * max_width / array.shape[1]))
        resized = Image.fromarray(array).resize((max_width, height), Image.LANCZOS)
        array = np.asarray(resized, dtype=np.uint8)
    return array


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
