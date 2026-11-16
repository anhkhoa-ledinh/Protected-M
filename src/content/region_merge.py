r"""
Gộp vùng cho SPIKE-11: đồ thị kề giữa các vùng, rồi gộp tham lam cặp rẻ nhất
theo ``C = α|Δd| + β‖Δc‖ + γE``. Tách khỏi ``seg_regions.py`` cho đủ giới hạn
dòng của RULES.md §4. Chỉ đọc.

XUAT XU: copy y nguyen tu ``scripts/_seg_merge.py`` cua nhanh
``spike11b-seg`` @ ``f834a41`` (SPIKE-11, khong merge vao master). PB-070b can
ban do vung P10 cho buoc bau, ma code nay chi nam o nhanh do va o ``D:\seg_lab``
(``vendor/_seg_merge.py``). Copy, KHONG import — app khong bao gio phu thuoc
``scripts/`` hay seg_lab.
"""
from __future__ import annotations

import heapq

import cv2
import numpy as np


def relabel(labels):
    """Đánh số lại 0..R-1 liên tục."""
    _, flat = np.unique(labels.reshape(-1), return_inverse=True)
    return flat.reshape(labels.shape).astype(np.int32)


def region_pixels(labels, n):
    """Danh sách chỉ số điểm ảnh (đã làm phẳng) của từng vùng."""
    flat = labels.reshape(-1)
    order = np.argsort(flat, kind="stable")
    bounds = np.searchsorted(flat[order], np.arange(n + 1))
    return [order[bounds[i]:bounds[i + 1]] for i in range(n)]


def _grad(rgb_small):
    """Độ lớn Sobel của ảnh xám, cắt về [0, 1]."""
    g = cv2.cvtColor(rgb_small, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    return np.clip(cv2.magnitude(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3),
                                 cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)), 0.0, 1.0)


def _borders(labels, grad):
    """``{(i, j): [tổng grad, số điểm]}`` trên biên chung, 4 lân cận."""
    n = int(labels.max()) + 1
    lo_all, hi_all, g_all = [], [], []
    for a, b, ga, gb in ((labels[:, :-1], labels[:, 1:], grad[:, :-1], grad[:, 1:]),
                         (labels[:-1, :], labels[1:, :], grad[:-1, :], grad[1:, :])):
        sel = a != b
        if not sel.any():
            continue
        i, j = a[sel].ravel(), b[sel].ravel()
        lo_all.append(np.minimum(i, j).astype(np.int64))
        hi_all.append(np.maximum(i, j).astype(np.int64))
        g_all.append(0.5 * (ga[sel].ravel() + gb[sel].ravel()))
    if not lo_all:
        return {}
    key = np.concatenate(lo_all) * n + np.concatenate(hi_all)
    s = np.bincount(key, weights=np.concatenate(g_all).astype(np.float64),
                    minlength=n * n)
    c = np.bincount(key, minlength=n * n)
    return {(int(k // n), int(k % n)): [float(s[k]), int(c[k])]
            for k in np.nonzero(c)[0]}


def _state(labels, depth_small, rgb_small, n):
    """Trạng thái gộp: trung vị depth, Lab trung bình, biên chung, đồ thị kề."""
    d = depth_small.reshape(-1)
    c = (cv2.cvtColor(rgb_small, cv2.COLOR_RGB2Lab).astype(np.float32)
         / 255.0).reshape(-1, 3)
    pix = region_pixels(labels, n)
    adj = _borders(labels, _grad(rgb_small))
    nbr = {i: set() for i in range(n)}
    for i, j in adj:
        nbr[i].add(j)
        nbr[j].add(i)
    return {"d": d, "c": c, "pix": pix, "adj": adj, "nbr": nbr,
            "dep": [float(np.median(d[q])) if q.size else 0.0 for q in pix],
            "col": [c[q].mean(axis=0) if q.size else np.zeros(3, np.float32)
                    for q in pix],
            "parent": list(range(n)), "ver": [0] * n}


def _find(parent, x):
    """Tìm gốc union-find, nén đường đi."""
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def _cost(S, p, i, j):
    """``α|Δd| + β‖Δc‖ + γE`` của một cặp vùng kề nhau."""
    s, cnt = S["adj"][(min(i, j), max(i, j))]
    return (p.alpha * abs(S["dep"][i] - S["dep"][j])
            + p.beta * float(np.linalg.norm(S["col"][i] - S["col"][j]))
            + p.gamma * s / max(cnt, 1))


def _absorb(S, p, heap, i, j):
    """Gộp ``j`` vào ``i``: cập nhật thống kê, biên chung, rồi đẩy cặp mới."""
    parent, nbr, adj = S["parent"], S["nbr"], S["adj"]
    parent[j] = i
    S["pix"][i] = np.concatenate((S["pix"][i], S["pix"][j]))
    S["dep"][i] = float(np.median(S["d"][S["pix"][i]]))
    S["col"][i] = S["c"][S["pix"][i]].mean(axis=0)
    S["ver"][i] += 1
    for k in list(nbr[j]):
        if k == i or _find(parent, k) != k:
            continue
        a = adj.pop((min(j, k), max(j, k)), [0.0, 0])
        key = (min(i, k), max(i, k))
        b = adj.get(key, [0.0, 0])
        adj[key] = [b[0] + a[0], b[1] + a[1]]
        nbr[i].add(k)
        nbr[k].add(i)
        nbr[k].discard(j)
    adj.pop((min(i, j), max(i, j)), None)
    nbr[i].discard(j)
    nbr[j] = set()
    for k in nbr[i]:
        if _find(parent, k) == k:
            heapq.heappush(heap, (_cost(S, p, i, k), i, k, S["ver"][i], S["ver"][k]))


def merge_regions(labels, depth_small, rgb_small, p, tau):
    """
    Gộp tham lam vùng kề nhau khi ``C < tau``, luôn lấy cặp rẻ nhất trước.

    ``tau`` truyền riêng (không đọc từ ``p``) vì ngưỡng khác nhau theo nguồn
    vùng — xem ``SegParams.tau_for``. Mục nào trong hàng đợi đã cũ (một đầu đã
    bị gộp, hoặc thống kê đã đổi) thì bỏ qua; vì là hàng đợi nhỏ nhất, gặp cặp
    hợp lệ đầu tiên có ``C ≥ tau`` là dừng được.
    """
    n = int(labels.max()) + 1
    S = _state(labels, depth_small, rgb_small, n)
    parent, ver = S["parent"], S["ver"]
    heap = [(_cost(S, p, i, j), i, j, 0, 0) for i, j in S["adj"]]
    heapq.heapify(heap)
    while heap:
        cij, i, j, vi, vj = heapq.heappop(heap)
        if _find(parent, i) != i or _find(parent, j) != j:
            continue
        if ver[i] != vi or ver[j] != vj:
            continue
        if cij >= tau:
            break
        _absorb(S, p, heap, i, j)
    root = np.array([_find(parent, i) for i in range(n)], dtype=np.int32)
    return relabel(root[labels])
