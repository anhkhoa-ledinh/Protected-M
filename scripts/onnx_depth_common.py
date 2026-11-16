r"""
PB-060: ba backend depth (torch FP32, ORT FP32, ORT INT8) va cac thuoc do.

Moi backend dung **cung** processor cua HF nhu ``src/content/depth_estimator.py``
nen chi khac nhau o buoc inference. Moi backend chay 2 thread de so sanh cong
bang voi cau hinh app (PB-057).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MODEL_ID = "LiheYoung/depth-anything-small-hf"
ROOT = Path(__file__).resolve().parents[1]
MODELS = ROOT / "artifacts" / "pb060_onnx" / "models"
ONNX_PATHS = {
    "ort_fp32": MODELS / "da_v1_small_fp32.onnx",
    "ort_int8": MODELS / "da_v1_small_int8.onnx",
}
BACKENDS = ("torch_fp32", "ort_fp32", "ort_int8")
THREADS = 2


def load_processor():
    """Processor cua HF, doc tu cache (khong tai mang)."""
    from transformers import AutoImageProcessor

    return AutoImageProcessor.from_pretrained(MODEL_ID, local_files_only=True)


def preprocess(processor, rgb):
    """RGB uint8 -> ``pixel_values`` float32 NCHW (numpy)."""
    out = processor(images=rgb, return_tensors="np")["pixel_values"]
    return np.ascontiguousarray(out, dtype=np.float32)


def make_torch_backend():
    """Mo hinh HF tren CPU, 2 thread; tra ve ham nhan NCHW -> depth HxW."""
    import torch
    from transformers import AutoModelForDepthEstimation

    torch.set_num_threads(THREADS)
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID, local_files_only=True)
    model.eval()

    def run(pixel_values):
        with torch.no_grad():
            out = model(pixel_values=torch.from_numpy(pixel_values)).predicted_depth
        return out.squeeze(0).numpy().astype(np.float32)

    return run


def make_ort_backend(name):
    """Session ORT CPU-only cho ``ort_fp32`` / ``ort_int8``, 2 intra / 1 inter."""
    import onnxruntime as ort

    path = ONNX_PATHS[name]
    if not path.exists():
        raise FileNotFoundError(f"Thieu file ONNX: {path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = THREADS
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(path), opts, providers=["CPUExecutionProvider"])

    def run(pixel_values):
        out = session.run(["predicted_depth"], {"pixel_values": pixel_values})[0]
        return np.asarray(out[0], dtype=np.float32)

    return run


def make_backend(name):
    """Chon backend theo ten; tra ve ``(run, model_mb)``."""
    if name == "torch_fp32":
        return make_torch_backend(), float("nan")
    size_mb = ONNX_PATHS[name].stat().st_size / 1e6
    return make_ort_backend(name), round(size_mb, 2)


def align_disparity(pred, gt):
    """
    Khop ``pred`` vao GT bang least-squares scale+shift **trong khong gian
    disparity**: tim ``s, t`` cho ``s*pred + t ~ 1/gt`` tren pixel GT > 0.

    Depth Anything tra ve gia tri kieu disparity (lon = gan), nen khop o day
    moi dung; doi sang depth bang ``1/(s*pred + t)``.
    """
    valid = gt > 0
    x = pred[valid].astype(np.float64)
    y = 1.0 / gt[valid].astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    scale, shift = np.linalg.lstsq(design, y, rcond=None)[0]
    disp = scale * pred.astype(np.float64) + shift
    depth = 1.0 / np.clip(disp, 1e-6, None)
    return depth, float(scale), float(shift)


def absrel_delta1(depth, gt):
    """AbsRel va delta1 (ty le pixel co max(d/gt, gt/d) < 1.25) tren GT > 0."""
    valid = gt > 0
    d = depth[valid]
    g = gt[valid].astype(np.float64)
    absrel = float(np.mean(np.abs(d - g) / g))
    ratio = np.maximum(d / g, g / d)
    return absrel, float(np.mean(ratio < 1.25))


def bootstrap_ci(values, n_boot=1000, seed=0):
    """Khoang tin cay 95% cua trung binh bang bootstrap percentile."""
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, arr.size, size=(int(n_boot), arr.size))
    means = arr[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))
