r"""
Nạp NYU Depth V2 labeled cho SPIKE-11b. Chỉ đọc.

File ``nyu_depth_v2_labeled.mat`` là HDF5 v7.3, mảng lưu **chuyển vị** so với
MATLAB: ``images`` là ``(N, 3, 640, 480)`` tức ``(N, C, W, H)``, nên một ảnh
phải ``transpose(2, 1, 0)`` mới ra ``(480, 640, 3)`` = (H, W, C) như OpenCV.

Một **vật** của ground truth là cặp ``(label, instance)``: ``instances`` đánh số
lại từ 1 trong từng lớp ngữ nghĩa, nên riêng ``instances`` không phân biệt được
hai vật khác loại. ``label == 0`` là điểm chưa gán nhãn.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import scipy.io as sio

ROOT = Path(__file__).resolve().parents[1]
MAT = ROOT / "nyu_depth_v2_labeled.mat"
SPLITS = ROOT / "splits.mat"


def test_indices():
    """654 chỉ số ảnh test chính thức, đổi từ 1-based của MATLAB sang 0-based."""
    raw = sio.loadmat(str(SPLITS))["testNdxs"].ravel().astype(np.int64)
    return np.sort(raw - 1)


def sample_indices(n=100, seed=0):
    """``n`` ảnh lấy từ tập test, seed cố định; trả theo thứ tự tăng dần."""
    rng = np.random.default_rng(int(seed))
    pool = test_indices()
    take = min(int(n), pool.size)
    return np.sort(rng.choice(pool, size=take, replace=False))


def object_ids(labels, instances):
    """
    Bản đồ id vật duy nhất từ cặp ``(label, instance)``; 0 = không phải vật.

    Ghép bằng ``label * 256 + instance`` rồi đánh số lại cho liên tục — chỉ cần
    duy nhất, không cần mang nghĩa.
    """
    lab = labels.astype(np.int64)
    ins = instances.astype(np.int64)
    pair = np.where((lab > 0) & (ins > 0), lab * 256 + ins, 0)
    uniq = np.unique(pair)
    out = np.zeros_like(pair, dtype=np.int32)
    next_id = 0
    for value in uniq:
        if value == 0:
            continue
        next_id += 1
        out[pair == value] = next_id
    return out


class NyuReader:
    """
    Mở file .mat một lần rồi đọc từng ảnh theo chỉ số.

    Có trạng thái (handle HDF5) và dùng ở nhiều script, nên để là class thay vì
    mở lại file 100 lần — file nặng 2,77 GB.
    """

    def __init__(self, path=MAT):
        self._f = h5py.File(str(path), "r")

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        self._f.close()

    def rgb(self, idx):
        """Ảnh RGB ``(480, 640, 3)`` uint8."""
        return np.ascontiguousarray(
            self._f["images"][int(idx)].transpose(2, 1, 0), dtype=np.uint8)

    def depth_gt(self, idx):
        """Depth ground truth ``(480, 640)`` mét, đã nội suy sẵn trong bộ dữ liệu."""
        return np.ascontiguousarray(
            self._f["depths"][int(idx)].T, dtype=np.float32)

    def objects(self, idx):
        """Bản đồ id vật ``(480, 640)`` int32; 0 = không thuộc vật nào."""
        labels = self._f["labels"][int(idx)].T
        instances = self._f["instances"][int(idx)].T
        return object_ids(labels, instances)
