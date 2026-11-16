r"""
PB-069: phép đo sương cho ``measure_fog_ab.py`` (tách ra để file dưới 150 dòng).

Định nghĩa dựng lại từ `MEMORY.md` mục 5 dòng 27/10: lúc đó Claude chat đo tay
trên PNG của seg_lab (`seg_lab\output\t25\layers_P5\layer0-2.png`) và ghi
"tương phản 46%, bão hoà 39%" cho lớp xa của t25, chưa có script nào trong
repo. Mức A trên t25 phải ra xấp xỉ hai số đó, đó là mốc tự kiểm.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.content.layer_clustering import LayerStagingParams

#: Ba mức sương của PB-069. Thứ tự ba số: ``atmosphere_blend_max`` /
#: ``far_desaturation`` / ``far_contrast_reduce``.
LEVELS = {
    "A": {"atmospheric_enabled": True, "atmosphere_blend_max": 0.35,
          "far_desaturation": 0.6, "far_contrast_reduce": 0.75},
    "B": {"atmospheric_enabled": True, "atmosphere_blend_max": 0.175,
          "far_desaturation": 0.8, "far_contrast_reduce": 0.875},
    "C": {"atmospheric_enabled": False},
}


def staging_for(cs, level):
    """``LayerStagingParams`` của một mức; mức C tắt hẳn staging."""
    over = LEVELS[level]
    return LayerStagingParams(
        enabled=bool(over["atmospheric_enabled"]),
        atmosphere_blend_max=float(
            over.get("atmosphere_blend_max", cs.atmosphere_blend_max)),
        far_desaturation=float(over.get("far_desaturation", cs.far_desaturation)),
        near_contrast_boost=float(cs.near_contrast_boost),
        far_contrast_reduce=float(
            over.get("far_contrast_reduce", cs.far_contrast_reduce)),
    )


def far_only_mask(layers):
    """Điểm **chỉ** thuộc lớp xa (lớp 0), trừ hết mask của các lớp gần hơn."""
    far = layers[0].mask.astype(bool)
    for layer in layers[1:]:
        far = far & ~layer.mask.astype(bool)
    return far


def nearer_mask(layers):
    """Hợp mask của mọi lớp gần hơn lớp xa — đầu vào của ``rings_dl``."""
    out = np.zeros(layers[0].mask.shape, dtype=bool)
    for layer in layers[1:]:
        out |= layer.mask.astype(bool)
    return out


def ratios(layer, original, mask):
    """
    ``(contrast_pct, saturation_pct)`` của lớp so với ảnh gốc, cùng tập pixel.

    So tỷ lệ chứ không so giá trị tuyệt đối: câu hỏi của PB-069 là "sương làm
    MẤT bao nhiêu phần tương phản/bão hoà của chính ảnh đó", nên mẫu số phải
    là ảnh gốc trên đúng những pixel ấy.
    """
    if not bool(mask.any()):
        return float("nan"), float("nan")
    lay = np.ascontiguousarray(layer.rgba[..., :3], dtype=np.uint8)
    src = np.ascontiguousarray(original, dtype=np.uint8)
    lum_a = cv2.cvtColor(lay, cv2.COLOR_RGB2GRAY).astype(np.float64)[mask]
    lum_b = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY).astype(np.float64)[mask]
    sat_a = cv2.cvtColor(lay, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float64)[mask]
    sat_b = cv2.cvtColor(src, cv2.COLOR_RGB2HSV)[..., 1].astype(np.float64)[mask]
    contrast = 100.0 * float(lum_a.std()) / max(float(lum_b.std()), 1e-9)
    saturation = 100.0 * float(sat_a.mean()) / max(float(sat_b.mean()), 1e-9)
    return contrast, saturation
