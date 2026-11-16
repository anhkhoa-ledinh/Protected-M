r"""
PB-070b: bản đồ vùng P10 (RGB-D Felzenszwalb) cho bước bầu của SPIKE-15.

XUẤT XỨ: ghép phần P10 từ ``scripts/seg_regions.py`` + ``seg_lab/regions.py`` +
``seg_lab/regions_c.py`` của nhánh ``spike11b-seg`` @ ``f834a41`` và
``D:\seg_lab`` @ ``2058b3b``. **Copy, KHÔNG import** — app không bao giờ phụ
thuộc ``scripts/`` hay seg_lab. Chỉ lấy đúng đường P10; các đường khác
(SLIC/P1, ``region_constant_depth``, P9, P11) **không** copy vì bước bầu không
cần: nó chỉ cần mask của từng vùng, không cần depth hằng theo vùng.

λ = 3.0 và bộ ``SegParams`` dưới đây **đóng băng** đúng giá trị SPIKE-13/15 đã
chấm; đổi là mất hiệu lực của số trên 765 ảnh.
"""
from __future__ import annotations

from typing import NamedTuple

import cv2
import numpy as np
import numpy.typing as npt

from src.content.region_merge import merge_regions, relabel

#: Cạnh dài nhất của ảnh làm việc; vùng tính ở cỡ này rồi phóng lên cỡ gốc.
WORK_MAX_DIM = 640
#: Trọng số kênh depth trong ``[L, a, b, λ·D]`` của P10. TUNING.md §P10.
P10_LAMBDA = 3.0


class SegParams(NamedTuple):
    """
    Tham số; **copy y nguyên** giá trị đóng băng của ``scripts/seg_regions.py``
    @ ``f834a41`` (dò trên t24 ở SPIKE-11). Chỉ giữ các trường đường P10 cần.
    """

    gs_sigma: float = 0.6      # Felzenszwalb
    gs_k: float = 220.0
    gs_min_size: int = 180
    alpha: float = 1.0         # trọng số |Δ depth|, depth trong [0, 1]
    beta: float = 0.6          # trọng số ‖Δ Lab‖, Lab đã chia 255
    gamma: float = 0.8         # trọng số biên Sobel, đã cắt về [0, 1]
    # Ngưỡng gộp của P2/P10: biên Felzenszwalb vốn nằm sẵn trên cạnh thật nên
    # số hạng γE ở đó lớn hơn hẳn SLIC; một τ chung thì P2 không gộp được cặp
    # nào. 0,35 là giá trị đã dò trên t24, không được chỉnh lại.
    tau_p2: float = 0.35


FROZEN = SegParams()


def work_image(rgb: npt.NDArray[np.uint8], max_dim: int = WORK_MAX_DIM):
    """Thu ảnh về cạnh dài ≤ ``max_dim`` để phân vùng cho nhanh."""
    h, w = rgb.shape[:2]
    s = min(1.0, float(max_dim) / float(max(h, w)))
    if s >= 1.0:
        return np.ascontiguousarray(rgb)
    size = (max(1, int(round(w * s))), max(1, int(round(h * s))))
    return np.ascontiguousarray(cv2.resize(rgb, size,
                                           interpolation=cv2.INTER_AREA))


def small_pair(rgb, depth):
    """``(rgb thu nhỏ, depth thu nhỏ cùng cỡ)``."""
    small = work_image(rgb)
    h, w = small.shape[:2]
    return small, cv2.resize(depth, (w, h), interpolation=cv2.INTER_AREA)


def _stack4(small, dep, lam=P10_LAMBDA):
    """Ảnh 4 kênh float32 ``[L, a, b, λ·D·255]`` cho ximgproc."""
    lab = cv2.cvtColor(small, cv2.COLOR_RGB2Lab).astype(np.float32)
    band = (float(lam) * dep * 255.0).astype(np.float32)
    return np.ascontiguousarray(np.dstack([lab, band]))


def p10_labels(small, dep):
    """P10: Felzenszwalb trên ``[L, a, b, λ·D]``, sigma/k/min_size của P2."""
    gs = cv2.ximgproc.segmentation.createGraphSegmentation(
        sigma=float(FROZEN.gs_sigma), k=float(FROZEN.gs_k),
        min_size=int(FROZEN.gs_min_size))
    return gs.processImage(_stack4(small, dep))


def region_masks(rgb, depth):
    """
    Mask của từng vùng P10 ở cỡ ảnh gốc.

    Phóng bản đồ nhãn bằng INTER_NEAREST — cùng phép phóng mà
    ``region_constant_depth`` dùng cho depth ở seg_lab, nên vùng ở đây trùng
    vùng P10 đã được chấm trên 765 ảnh.
    """
    small, dep = small_pair(rgb, depth)
    base = p10_labels(small, dep)
    merged = relabel(merge_regions(base, dep, small, FROZEN, FROZEN.tau_p2))
    full = cv2.resize(merged.astype(np.int32),
                      (int(depth.shape[1]), int(depth.shape[0])),
                      interpolation=cv2.INTER_NEAREST)
    return [full == i for i in range(int(full.max()) + 1)]
