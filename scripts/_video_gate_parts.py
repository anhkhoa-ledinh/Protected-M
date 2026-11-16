r"""
SPIKE-12: các mảnh của ``video_gate.py`` (tách ra cho đủ giới hạn 150 dòng).

Kích thước depth khung khoá = **266**: theo ``artifacts/depth_ab/depth_size.csv``
đó là cỡ NHỎ NHẤT còn cho **cùng K = 3** như 518 trên t22 (518 → K 3 ở
2081,4 ms; 364 → K 3 ở 866,9 ms; 266 → K 3 ở 418,9 ms).
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from src.content.depth_estimator import normalize_depth_map

WORK_LONG_SIDE = 640
DEPTH_SHORTEST = 266
MODEL_ID = "LiheYoung/depth-anything-small-hf"
_CACHE = {}


def load_depth(threads=2):
    """Hàm depth dùng processor cỡ 266; trả ``rgb -> depth [0,1]``."""
    import torch
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    key = f"depth{threads}"
    if key not in _CACHE:
        torch.set_num_threads(int(threads))
        # DPT muốn {"height","width"}; nó TỰ giữ tỉ lệ (keep_aspect_ratio) và
        # làm tròn về bội 14, nên 266 ra tensor 266x350 đúng như depth_size.csv.
        proc = AutoImageProcessor.from_pretrained(
            MODEL_ID, size={"height": DEPTH_SHORTEST,
                            "width": DEPTH_SHORTEST})
        model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).eval()
        _CACHE[key] = (proc, model)
    proc, model = _CACHE[key]

    def run(rgb):
        import torch as _t

        with _t.no_grad():
            out = model(**proc(images=rgb, return_tensors="pt")).predicted_depth
        raw = out.squeeze(0).numpy().astype(np.float32)
        return normalize_depth_map(raw, rgb.shape[0], rgb.shape[1])

    return run


def work_resize(bgr):
    """Thu về cạnh dài ``WORK_LONG_SIDE``, trả RGB."""
    h, w = bgr.shape[:2]
    s = WORK_LONG_SIDE / float(max(h, w))
    if s < 1.0:
        bgr = cv2.resize(bgr, (max(1, int(round(w * s))),
                               max(1, int(round(h * s)))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def dis(preset):
    """``DISOpticalFlow`` theo preset ULTRAFAST / FAST."""
    table = {"ULTRAFAST": cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
             "FAST": cv2.DISOPTICAL_FLOW_PRESET_FAST}
    return cv2.DISOpticalFlow_create(table[preset])


def warp_by_flow(src, flow):
    """Kéo ``src`` theo ``flow`` bằng remap (dùng cho depth khung khoá)."""
    h, w = flow.shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    return cv2.remap(src, gx + flow[..., 0], gy + flow[..., 1],
                     interpolation=cv2.INTER_LINEAR,
                     borderMode=cv2.BORDER_REPLICATE)


def labels_from_centroids(depth, centroids):
    """Gán mỗi điểm về tâm cụm GẦN NHẤT — ngưỡng cố định, không cụm lại."""
    c = np.asarray(centroids, dtype=np.float32).ravel()
    return np.argmin(np.abs(depth[..., None] - c[None, None, :]),
                     axis=2).astype(np.int32)


def light_layering(rgb, labels, n_lab, feather_px=6):
    """
    Tầng lớp nhẹ: mask → ``feather_mask_alpha`` → RGBA, tấm nền alpha 255.

    Trả ``(danh sách RGBA, thời gian ms)``. Không Telea, không relief — đúng
    thiết kế video của TASK_video25d.
    """
    from src.content.layer_clustering import feather_mask_alpha

    t0 = time.perf_counter()
    out = []
    for i in range(int(n_lab)):
        mask = labels == i
        alpha = (feather_mask_alpha(mask, feather_px, feather_px)
                 if mask.any() else np.zeros(mask.shape, np.uint8))
        if i == 0:
            alpha = np.full(mask.shape, 255, np.uint8)
        rgba = np.dstack([rgb, alpha])
        out.append(np.ascontiguousarray(rgba))
    return out, (time.perf_counter() - t0) * 1000.0


def flip_rate(prev_labels, labels):
    """% điểm đổi nhãn lớp giữa hai khung nội dung liền nhau."""
    if prev_labels is None:
        return float("nan")
    return 100.0 * float(np.count_nonzero(prev_labels != labels)) \
        / float(labels.size)


def amortised_p(p_base_s, keyframe_depth_ms, k):
    """
    ``p`` mỗi khung nội dung KHI chia đều depth khung khoá cho ``k`` khung.

    ``p_base_s`` là đường mỗi khung (decode + flow + tầng lớp). Depth chỉ chạy
    ở khung khoá thứ ``k``, nhưng nó VẪN là chi phí của producer nên phải nằm
    trong ``p``: bỏ nó ra là báo 65 fps cho một đường thật ra chạy 13 fps.
    """
    return float(p_base_s) + float(keyframe_depth_ms) / (1000.0 * max(1, int(k)))


def wait_table(p_seconds, lengths=(30.0, 60.0), fps=(5.0, 8.0, 10.0)):
    """``W = L·(p·f − 1)`` khi ``p·f > 1``, không thì 0 (không bao giờ nghẽn)."""
    rows = []
    for f in fps:
        pf = float(p_seconds) * float(f)
        for L in lengths:
            rows.append({"f_fps": f, "clip_s": L, "p_s": round(p_seconds, 4),
                         "p_times_f": round(pf, 4),
                         "W_s": round(max(0.0, L * (pf - 1.0)), 2)})
    return rows
