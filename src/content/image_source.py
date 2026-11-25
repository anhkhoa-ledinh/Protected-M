"""Static image source for the content plane."""

from __future__ import annotations

import logging

import cv2
import numpy as np
import numpy.typing as npt

from src.content import media_gate
from src.content.media_gate import CleanPhoto

logger = logging.getLogger(__name__)

#: Cạnh dài tối đa của ảnh sau khi nạp (PB-050). Model depth chỉ nhận cạnh ngắn
#: 518 px: ảnh 5823×2700 và 2048×950 cho CÙNG đầu vào 1120×518, nên chặn ở đây
#: không mất chút depth nào. 2048 vẫn ≥ 1366 px ngang màn cho lớp gần (k = 1) và
#: là cỡ lớn nhất đã chạy ổn trong app (t1.png 2048×1542, run8/run9 ngày 16/10).
#: Không chặn thì ảnh 5823×2700 làm app chết 3/3 lần khi dựng lớp (MemoryError
#: 180 MiB trong ``stage_layer_color``). Đổi lại, lớp xa mềm hơn với ảnh lớn hơn
#: 2048: ảnh 5823×2700 ở D 450 còn 0,74 / 0,81 / 0,94 texel mỗi pixel màn ở
#: L0 / L1 / L2 (trước khi chặn 2,1–2,7), vẫn hơn t22 (0,64 / 0,71).
MAX_IMAGE_SIDE_PX = media_gate.PIPELINE_MAX_SIDE_PX


def _cap_size(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Thu nhỏ giữ tỉ lệ (INTER_AREA) khi cạnh dài vượt ``MAX_IMAGE_SIDE_PX``."""
    h, w = image.shape[:2]
    scale = MAX_IMAGE_SIDE_PX / float(max(h, w))
    if scale >= 1.0:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class ImageSource:
    """Load a single image and expose it as a static RGB frame stream.

    Only a :class:`CleanPhoto` is accepted: the file already went through
    ``media_gate.import_photo`` (checked, re-encoded, capped)."""

    def __init__(self, photo: CleanPhoto) -> None:
        if not isinstance(photo, CleanPhoto):
            raise TypeError("ImageSource needs a CleanPhoto from media_gate.import_photo")
        path = photo.path
        self._path = path
        self._rgb: npt.NDArray[np.uint8] | None = None
        self._width = 1
        self._height = 1
        self._aspect = 16.0 / 9.0
        self._load(photo)

    @property
    def aspect_ratio(self) -> float:
        return self._aspect

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    def close(self) -> None:
        self._rgb = None

    def tick(self, _dt_s: float) -> npt.NDArray[np.uint8] | None:
        return self._rgb

    def _load(self, photo: CleanPhoto) -> None:
        path = photo.path
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        rgb = media_gate.load_rgb(photo)
        src_h, src_w = rgb.shape[:2]
        # The gate already capped the long side; kept as a safety net.
        self._rgb = _cap_size(rgb)
        self._height, self._width = self._rgb.shape[:2]
        if (src_w, src_h) != (self._width, self._height):
            logger.info(
                "Image capped (PB-050): %s %dx%d -> %dx%d (MAX_IMAGE_SIDE_PX=%d)",
                path.name, src_w, src_h, self._width, self._height, MAX_IMAGE_SIDE_PX,
            )
        self._aspect = self._width / float(self._height)
        logger.info(
            "Image loaded: %s (%dx%d, aspect %.3f)",
            path.name,
            self._width,
            self._height,
            self._aspect,
        )
