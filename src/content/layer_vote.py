r"""
PB-070b: "lớp đệm" cho k-means — bầu theo từng vùng (SPIKE-15, P18b thắng).

XUẤT XỨ: copy từ ``D:\seg_lab\seg_lab\vote.py`` @ ``2058b3b``. **Copy, KHÔNG
import.** Chỉ lấy đường P18b (mask = vùng P10); đường P18 (mask FastSAM) không
copy vì nó kéo theo ultralytics + giấy phép AGPL vào app.

Vì sao nó tồn tại: P2/P5/P10 **thay depth** cả vùng bằng một hằng rồi **cụm
lại**, nên tâm cụm dịch và vùng có thể rơi vào lớp mà ít pixel của nó từng ở —
mất 4–6 điểm thứ tự che khuất trên 765 ảnh. Ở đây **không bao giờ cụm lại**:
nhãn và tâm cụm vẫn đúng của k-means, chỉ pixel thiểu số trong mỗi vùng bị kéo
về nhãn đa số của chính vùng đó (Δocc chỉ −0,97 điểm).

Hằng số đóng băng theo SPIKE-15; đổi là mất hiệu lực của số trên 765 ảnh.
"""
from __future__ import annotations

import logging

import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

#: Tỷ lệ tối thiểu của nhãn đa số trong một vùng để được dồn cả vùng về nó.
TAU_F = 0.6
#: Vùng nhỏ hơn số điểm này bị bỏ qua.
MIN_PX = 64


def centroid_gaps(centroids) -> npt.NDArray[np.float64]:
    """Với mỗi tâm cụm, khoảng cách tới tâm cụm GẦN NHẤT khác nó."""
    c = np.asarray(centroids, dtype=np.float64).ravel()
    if c.size < 2:
        # K = 1: không có lớp cạnh tranh nên cửa ``s <= g`` không được chặn;
        # để 0 thì nó chặn sạch mọi vùng.
        return np.full(c.size, np.inf)
    diff = np.abs(c[:, None] - c[None, :])
    np.fill_diagonal(diff, np.inf)
    return diff.min(axis=1)


def vote_labels(labels, centroids, depth, masks, tau_f=TAU_F, min_px=MIN_PX):
    """
    ``(labels mới, relabel_frac)``. Tâm cụm KHÔNG đổi.

    Xét vùng từ LỚN tới NHỎ để vùng nhỏ nằm trong vùng lớn được quyết sau (nó
    ghi đè). ``ℓ*`` luôn tính trên nhãn k-means **gốc**, không phải nhãn đã bị
    vùng trước sửa — nếu không thì chính thứ tự xét sẽ đổi kết quả.
    """
    labels = np.asarray(labels)
    out = labels.copy()
    n_lab = int(np.asarray(centroids).ravel().size)
    if n_lab == 0 or not masks:
        return out, 0.0
    gaps = centroid_gaps(centroids)
    for mask in sorted(masks, key=lambda m: -int(np.count_nonzero(m))):
        mask = np.asarray(mask, dtype=bool)
        area = int(np.count_nonzero(mask))
        if area < int(min_px):
            continue
        counts = np.bincount(labels[mask].ravel(), minlength=n_lab)
        star = int(counts.argmax())
        share = float(counts[star]) / float(area)
        band = depth[mask]
        spread = float(np.percentile(band, 90) - np.percentile(band, 10))
        if share >= float(tau_f) and spread <= float(gaps[star]):
            out[mask] = star
    relabel_frac = float(np.count_nonzero(out != labels)) / float(labels.size)
    return out, relabel_frac


def vote_on_regions(labels, centroids, depth, rgb):
    """
    Bầu trên vùng P10 (P18b). Lỗi ở bước vùng thì giữ nguyên nhãn k-means.

    Không để một lỗi phân vùng làm chết cả lượt nạp ảnh: app vẫn phải dựng
    được lớp, chỉ là không có bước bầu.
    """
    from src.content.layer_regions import region_masks

    try:
        masks = region_masks(rgb, depth)
    except Exception as exc:
        logger.warning("Layer vote skipped (region step failed: %s)", exc)
        return np.asarray(labels), 0.0
    voted, frac = vote_labels(labels, centroids, depth, masks)
    logger.info("Layer vote: %d regions, %.2f%% pixels relabelled",
                len(masks), 100.0 * frac)
    return voted, frac
