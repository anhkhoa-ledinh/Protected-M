"""
Chụp màn cho luận văn: đọc framebuffer rồi ghi PNG.

``glReadPixels`` trả hàng từ **dưới lên**, nên phải lật dọc trước khi lưu.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from OpenGL import GL
from PIL import Image

logger = logging.getLogger(__name__)


def save_framebuffer(path: Path, width: int, height: int) -> Path:
    """Lưu vùng vẽ hiện tại ra ``path`` (tự tạo thư mục). Trả đường dẫn."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    raw = GL.glReadPixels(0, 0, int(width), int(height), GL.GL_RGB,
                          GL.GL_UNSIGNED_BYTE)
    array = np.frombuffer(raw, dtype=np.uint8).reshape(int(height), int(width), 3)
    Image.fromarray(array[::-1]).save(path)
    logger.info("Album shot: %s (%dx%d)", path, width, height)
    return path
