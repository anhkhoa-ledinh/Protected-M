"""Static image source for the content plane."""

from __future__ import annotations

import logging
from pathlib import Path

from src.content.cv_io import imread_bgr
import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: Cạnh dài tối đa của ảnh sau khi nạp (PB-050). Model depth chỉ nhận cạnh ngắn
#: 518 px: ảnh 5823×2700 và 2048×950 cho CÙNG đầu vào 1120×518, nên chặn ở đây
#: không mất chút depth nào. 2048 vẫn ≥ 1366 px ngang màn cho lớp gần (k = 1) và
#: là cỡ lớn nhất đã chạy ổn trong app (t1.png 2048×1542, run8/run9 ngày 16/10).
#: Không chặn thì ảnh 5823×2700 làm app chết 3/3 lần khi dựng lớp (MemoryError
#: 180 MiB trong ``stage_layer_color``). Đổi lại, lớp xa mềm hơn với ảnh lớn hơn
#: 2048: ảnh 5823×2700 ở D 450 còn 0,74 / 0,81 / 0,94 texel mỗi pixel màn ở
#: L0 / L1 / L2 (trước khi chặn 2,1–2,7), vẫn hơn t22 (0,64 / 0,71).
MAX_IMAGE_SIDE_PX = 2048


def _cap_size(image: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    """Thu nhỏ giữ tỉ lệ (INTER_AREA) khi cạnh dài vượt ``MAX_IMAGE_SIDE_PX``."""
    h, w = image.shape[:2]
    scale = MAX_IMAGE_SIDE_PX / float(max(h, w))
    if scale >= 1.0:
        return image
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


class ImageSource:
    """Load a single image and expose it as a static RGB frame stream."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._rgb: npt.NDArray[np.uint8] | None = None
        self._width = 1
        self._height = 1
        self._aspect = 16.0 / 9.0
        self._load(path)

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

    def _load(self, path: Path) -> None:
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        bgr = imread_bgr(path)
        if bgr is None:
            raise ValueError(
                f"Failed to decode image: {path.name} "
                f"(file exists but OpenCV could not read it — "
                f"try moving it to a simple ASCII path like D:\\Pictures\\photo.png)"
            )
        src_h, src_w = bgr.shape[:2]
        # Chặn trên BGR, trước cvtColor: khỏi giữ thêm một bản RGB cỡ gốc.
        self._rgb = cv2.cvtColor(_cap_size(bgr), cv2.COLOR_BGR2RGB)
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
