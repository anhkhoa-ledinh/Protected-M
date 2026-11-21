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

#: TASK_perf_textures_1119 Fix 2: band half-width (px, in FULL-res terms) for
#: the fast runtime bleed — matches the offline ``tools/prebleed.py``'s 4 px
#: reach.
BLEED_BAND_PX = 4

#: Downsample factor cap for the low-res blur below. Capped by image size
#: (``_bleed_band_scale``) so small sprites don't collapse to a 1x1 pixel.
BLEED_BAND_SCALE_CAP = 8


def _bleed_band_scale(width: int, height: int) -> int:
    return max(1, min(BLEED_BAND_SCALE_CAP, width // 32, height // 32))


def bleed_transparent_band(array: np.ndarray, band_px: int = BLEED_BAND_PX) -> np.ndarray:
    """
    Nhanh hơn ``bleed_transparent``: thay vì 6 lượt box-blur float32 TRÊN
    TOÀN BỘ ảnh gốc, thu ảnh về một bản nhỏ (``_bleed_band_scale`` — các
    sprite bé không thu, ảnh UI 1920x1080 thu 8×), loang màu trên bản nhỏ
    đó (rẻ — ít điểm hơn nhiều lần), rồi phóng to lại đúng kích cỡ và chỉ
    ghi vào những điểm alpha=0 (``cv2.copyTo`` có mặt nạ).

    Trên bản nhỏ: điểm nào còn "thấy" một điểm alpha>0 trong bán kính
    ``band_px`` (sau khi quy đổi theo tỉ lệ thu) nhận màu lân cận THẬT
    (``cv2.blur`` kernel nhỏ, trọng số theo alpha); điểm nào không thấy gì
    (mẫu số blur ~0 — xa mọi điểm alpha>0) nhận MỘT màu phẳng, trung bình
    trọng số của toàn bộ điểm alpha>0 trên bản nhỏ — không ai phân biệt
    được sự khác nhau giữa các màu lân cận ở đó nữa (mipmap bậc 0-1 chỉ
    đọc 1-2px qua viền — xa viền thì không ai nhìn tới).

    4 kênh được xử lý CÙNG LÚC qua ``cv2``/``cv2.copyTo`` — tách riêng
    kênh RGB khỏi mảng RGBA bằng cách cắt lát (``array[..., :3]``) rồi
    ghi lại là một mảng KHÔNG liên tục trong bộ nhớ, đo được chậm hơn
    nhiều so với ``cv2.split``/``cv2.merge`` hay giữ nguyên 4 kênh trên
    máy này.

    Dùng cho file CHƯA pre-bled (không có ``rgb_bled=1``) — xem
    ``load_rgba``. Mục tiêu đo được: ≤ 20 ms / megapixel trên máy này
    (tests/test_aa_mipmaps.py đo lại bằng ``time.perf_counter``).
    """
    if array.ndim != 3 or array.shape[2] != 4:
        return array
    alpha = array[..., 3]
    if not np.any(alpha == 0):
        return array
    opaque_u8 = (alpha > 0).astype(np.uint8)
    if not np.any(opaque_u8):
        return array           # không điểm nào alpha>0: không có màu để loang

    height, width = alpha.shape
    scale = _bleed_band_scale(width, height)
    small_w, small_h = max(1, width // scale), max(1, height // scale)
    small = cv2.resize(array, (small_w, small_h), interpolation=cv2.INTER_AREA)

    small_alpha = small[..., 3].astype(np.float32) * (1.0 / 255.0)
    small_rgb = small[..., :3].astype(np.float32)
    small_num = small_rgb * small_alpha[..., None]
    reach = max(1, band_px // scale)
    ksize = 2 * reach + 1
    blurred_num = cv2.blur(small_num, (ksize, ksize))
    blurred_den = cv2.blur(small_alpha, (ksize, ksize))

    den_total = float(small_alpha.sum())
    flat = (small_num.sum(axis=(0, 1)) / den_total) if den_total > 1e-6 else np.zeros(3, np.float32)

    fill_small = np.empty_like(small_rgb, dtype=np.uint8)
    local = blurred_den > 0.05
    fill_small[~local] = np.clip(flat, 0.0, 255.0).astype(np.uint8)
    local_fill = np.clip(blurred_num / np.maximum(blurred_den[..., None], 1e-6), 0.0, 255.0)
    fill_small[local] = local_fill[local].astype(np.uint8)

    # 4th channel stays 0 — harmless, ``copyTo`` below only ever writes it
    # where alpha was ALREADY 0 (that's the mask), so alpha is unchanged.
    zero_channel = np.zeros((small_h, small_w), np.uint8)
    fill_small_rgba = cv2.merge((fill_small[..., 0], fill_small[..., 1],
                                 fill_small[..., 2], zero_channel))
    fill_full_rgba = cv2.resize(fill_small_rgba, (width, height), interpolation=cv2.INTER_LINEAR)

    transparent_u8 = (alpha == 0).astype(np.uint8)
    out = array.copy()
    cv2.copyTo(fill_full_rgba, transparent_u8, out)
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

    TASK_perf_textures_1119 Fix 2: file đã được ``tools/prebleed.py``
    loang màu OFFLINE (``rgb_bled=1`` trong PNG text chunk — đọc được
    ngay từ ``image.info`` sau ``Image.open()``, không cần giải mã điểm
    ảnh) thì KHÔNG loang lại. File chưa pre-bled nhưng không có điểm nào
    alpha=0 thì cũng không có gì để loang. Còn lại (file lạ, chưa pre-bled,
    có điểm alpha=0) mới chạy ``bleed_transparent_band`` — bản rẻ, chỉ
    tính màu lân cận THẬT trong dải 4px, phần còn lại nhận một màu phẳng.
    """
    try:
        with Image.open(path) as image:
            already_bled = image.info.get("rgb_bled") == "1"
            array = np.asarray(image.convert("RGBA"), dtype=np.uint8)
    except (OSError, ValueError, MemoryError) as err:
        logger.warning("Album: khong doc duoc %s (%s)", path, err)
        return None
    if not already_bled and array.shape[2] == 4 and np.any(array[..., 3] == 0):
        array = bleed_transparent_band(array)
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
