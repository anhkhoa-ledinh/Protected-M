r"""
PB-060: do mot backend depth (chay rieng mot tien trinh de peak RSS sach).

    .\.venv\Scripts\python.exe scripts\onnx_compare_depth.py torch_fp32

Ghi ``artifacts/pb060_onnx/rows_<backend>.csv`` (moi dong mot anh) va
``meta_<backend>.csv`` (model MB, peak RSS). Backend ``torch_fp32`` con luu
depth da chuan hoa vao ``ref/`` lam moc do sai lech cho hai backend ORT.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from src.content.depth_estimator import normalize_depth_map  # noqa: E402
import onnx_depth_common as C  # noqa: E402
from _peak_ram import peak_working_set_mb  # noqa: E402
from nyu_loader import NyuReader, sample_indices  # noqa: E402

IMG_DIR = Path(r"D:\PycharmProjects\test_images")
APP_IMGS = ["t21", "t22", "t23", "t24", "t25", "t26", "t11", "t4"]
OUT = ROOT / "artifacts" / "pb060_onnx"
REF = OUT / "ref"
NYU_N = 100
FIELDS = ["backend", "kind", "image", "img_w", "img_h", "tensor_shape",
          "t_pre_ms", "t_infer_ms", "max_abs_diff", "mean_abs_diff",
          "absrel", "delta1", "scale", "shift"]


def timed_runs(processor, run, rgb, repeats=3):
    """1 lan warm-up + ``repeats`` lan do; tra ve (trung vi pre, trung vi infer, depth)."""
    pre, infer = [], []
    depth = None
    for i in range(repeats + 1):
        t0 = time.perf_counter()
        pixel_values = C.preprocess(processor, rgb)
        t1 = time.perf_counter()
        raw = run(pixel_values)
        t2 = time.perf_counter()
        if i == 0:  # bo lan warm-up
            continue
        pre.append((t1 - t0) * 1000.0)
        infer.append((t2 - t1) * 1000.0)
        depth = raw
    return st.median(pre), st.median(infer), depth, pixel_values.shape


def app_rows(backend, processor, run):
    """8 anh cua app: thoi gian tach pre/infer + sai lech so voi torch_fp32."""
    rows = []
    for name in APP_IMGS:
        bgr = cv2.imread(str(IMG_DIR / f"{name}.png"))
        if bgr is None:
            raise FileNotFoundError(IMG_DIR / f"{name}.png")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        t_pre, t_inf, raw, shape = timed_runs(processor, run, rgb)
        depth = normalize_depth_map(raw, h, w)
        ref_path = REF / f"{name}.npy"
        if backend == "torch_fp32":
            REF.mkdir(parents=True, exist_ok=True)
            np.save(ref_path, depth)
            max_d = mean_d = 0.0
        else:
            diff = np.abs(depth - np.load(ref_path))
            max_d, mean_d = float(diff.max()), float(diff.mean())
        rows.append({
            "backend": backend, "kind": "app", "image": name,
            "img_w": w, "img_h": h,
            "tensor_shape": "x".join(str(v) for v in shape),
            "t_pre_ms": round(t_pre, 1), "t_infer_ms": round(t_inf, 1),
            "max_abs_diff": f"{max_d:.8f}", "mean_abs_diff": f"{mean_d:.8f}",
            "absrel": "", "delta1": "", "scale": "", "shift": "",
        })
        print(f"  {name} {w}x{h} pre={t_pre:.1f}ms infer={t_inf:.1f}ms "
              f"max_diff={max_d:.6f}")
    return rows


def nyu_rows(backend, processor, run):
    """100 anh test NYU (seed 0): AbsRel va delta1 sau khi khop scale+shift."""
    rows = []
    indices = sample_indices(NYU_N, seed=0)
    with NyuReader() as reader:
        for pos, idx in enumerate(indices):
            rgb = reader.rgb(idx)
            gt = reader.depth_gt(idx)
            raw = run(C.preprocess(processor, rgb))
            pred = cv2.resize(raw, (gt.shape[1], gt.shape[0]),
                              interpolation=cv2.INTER_LINEAR)
            depth, scale, shift = C.align_disparity(pred, gt)
            absrel, delta1 = C.absrel_delta1(depth, gt)
            rows.append({
                "backend": backend, "kind": "nyu", "image": f"nyu_{int(idx):04d}",
                "img_w": gt.shape[1], "img_h": gt.shape[0], "tensor_shape": "",
                "t_pre_ms": "", "t_infer_ms": "",
                "max_abs_diff": "", "mean_abs_diff": "",
                "absrel": f"{absrel:.6f}", "delta1": f"{delta1:.6f}",
                "scale": f"{scale:.6g}", "shift": f"{shift:.6g}",
            })
            if (pos + 1) % 25 == 0:
                print(f"  NYU {pos + 1}/{len(indices)}")
    return rows


def main(backend):
    if backend not in C.BACKENDS:
        print(f"backend phai thuoc {C.BACKENDS}")
        return 2
    OUT.mkdir(parents=True, exist_ok=True)
    processor = C.load_processor()
    run, model_mb = C.make_backend(backend)
    print(f"== {backend} (threads={C.THREADS}, model_mb={model_mb})")
    rows = app_rows(backend, processor, run) + nyu_rows(backend, processor, run)
    with open(OUT / f"rows_{backend}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    peak = peak_working_set_mb()
    with open(OUT / f"meta_{backend}.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["backend", "model_mb", "peak_rss_mb"])
        writer.writeheader()
        writer.writerow({"backend": backend, "model_mb": model_mb,
                         "peak_rss_mb": peak})
    print(f"peak_rss_mb={peak}  rows={len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else ""))
