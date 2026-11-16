"""Monocular depth estimation for content-plane displacement."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

# File ONNX xuất ở PB-060 Phase 1b (bản FP32 — bản INT8 chậm gấp ~2,6 lần nên
# không dùng trong app). Đường dẫn tương đối so với gốc repo; file nằm trong
# artifacts/ nên gitignored, vì vậy backend phải chịu được việc nó không có.
ONNX_DEPTH_RELPATH = Path("artifacts") / "pb060_onnx" / "models" / "da_v1_small_fp32.onnx"

# Số thread cho ORT, lấy đúng cấu hình đã đo ở PB-057/PB-060 (2 intra, 1 inter).
_ORT_INTRA_THREADS = 2
_ORT_INTER_THREADS = 1


class _DepthBackend(Protocol):
    def estimate(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]: ...


def normalize_depth_map(raw: npt.NDArray[np.float32], target_h: int, target_w: int) -> npt.NDArray[np.float32]:
    """
    Normalize depth to 0-1 (1 = near / toward viewer) and resize to target size.

    Uses percentile clipping so a few outlier pixels do not crush contrast.
    """
    depth = raw.astype(np.float32)
    lo, hi = float(np.percentile(depth, 2)), float(np.percentile(depth, 98))
    depth = np.clip(depth, lo, hi)
    depth = (depth - lo) / max(hi - lo, 1e-6)
    if depth.shape[0] != target_h or depth.shape[1] != target_w:
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return np.ascontiguousarray(depth, dtype=np.float32)


class DepthEstimator:
    """
    Lazy-loaded monocular depth backend.

    Supports MiDaS small (torch.hub) and Depth Anything small (transformers).

    ``backend`` chọn engine chạy inference cho Depth Anything: ``"torch"``
    (mặc định) hoặc ``"onnxruntime"``. Hai engine dùng **cùng** processor của
    HF nên chỉ khác ở bước inference; ORT không dùng được thì rơi về torch.
    """

    def __init__(
        self, model_name: str = "depth-anything-small", backend: str = "torch"
    ) -> None:
        self._model_name = model_name.lower().strip()
        self._backend_name = backend.lower().strip()
        self._backend: _DepthBackend | None = None
        self._load_error: str | None = None

    @property
    def available(self) -> bool:
        return self._ensure_loaded()

    @property
    def load_error(self) -> str | None:
        return self._load_error

    def estimate(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """Return depth map HxW float32 in [0, 1], 1 = near."""
        if not self._ensure_loaded():
            raise RuntimeError(self._load_error or "Depth model unavailable")
        assert self._backend is not None
        h, w = rgb.shape[:2]
        raw = self._backend.estimate(rgb)
        return normalize_depth_map(raw, h, w)

    def _ensure_loaded(self) -> bool:
        if self._backend is not None:
            return True
        if self._load_error is not None:
            return False
        try:
            self._backend = self._create_backend()
            logger.info("Depth model loaded: %s", self._model_name)
            return True
        except Exception as exc:
            self._load_error = str(exc)
            logger.warning(
                "Depth model '%s' unavailable (%s) — flat content plane will be used",
                self._model_name,
                exc,
            )
            return False

    def _create_backend(self) -> _DepthBackend:
        if self._model_name in {"midas-small", "midas_small", "midas"}:
            return _MiDaSBackend("MiDaS_small")
        if self._model_name in {"depth-anything-small", "depth_anything_small", "depth-anything"}:
            model_id = "LiheYoung/depth-anything-small-hf"
            if self._backend_name in {"onnxruntime", "onnx", "ort"}:
                onnx_path = Path(__file__).resolve().parents[2] / ONNX_DEPTH_RELPATH
                try:
                    return _OnnxRuntimeBackend(model_id, onnx_path)
                except Exception as exc:
                    logger.warning(
                        "ONNX Runtime unavailable (%s), falling back to torch", exc
                    )
            try:
                return _DepthAnythingBackend(model_id)
            except Exception as exc:
                logger.warning("Depth Anything failed (%s), falling back to MiDaS small", exc)
                return _MiDaSBackend("MiDaS_small")
        raise ValueError(f"Unknown depth model: {self._model_name}")


class _MiDaSBackend:
    def __init__(self, variant: str) -> None:
        import torch

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = torch.hub.load("intel-isl/MiDaS", variant, pretrained=True)
        self._model.to(self._device).eval()
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        self._transform = transforms.small_transform
        logger.info("Depth backend: torch (%s)", variant)

    def estimate(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        import torch

        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        tensor = self._transform(bgr).to(self._device)
        with torch.no_grad():
            prediction = self._model(tensor)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        return depth


class _DepthAnythingBackend:
    def __init__(self, model_id: str) -> None:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        self._model = AutoModelForDepthEstimation.from_pretrained(model_id)
        self._model.to(self._device).eval()
        logger.info("Depth backend: torch (%s)", model_id)

    def estimate(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        import torch

        inputs = self._processor(images=rgb, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
            prediction = getattr(outputs, "predicted_depth", None)
            if prediction is None:
                raise RuntimeError(
                    "Depth model output has no 'predicted_depth' field"
                )
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
        depth = prediction.cpu().numpy().astype(np.float32)
        return depth


class _OnnxRuntimeBackend:
    """
    Depth Anything Small chạy bằng ONNX Runtime (CPU), file .onnx của PB-060.

    Dùng **đúng** processor của HF như ``_DepthAnythingBackend`` nên chỉ khác ở
    bước inference; nhờ vậy sai khác so với torch chỉ đến từ engine.

    Lưu ý: backend này **vẫn nạp torch**. ``onnxruntime`` thì không, nhưng
    ``from transformers import AutoImageProcessor`` kéo torch vào
    ``sys.modules`` (đo ngày 2026-10-29 trong process sạch: False trước khi
    dựng backend, True sau khi dựng). Vậy chỉ có inference là không đi qua
    torch; bỏ torch khỏi dependency hay tiết kiệm RAM thì không được — muốn
    thế phải tự viết lại processor, ngoài phạm vi PB-060.
    """

    def __init__(self, model_id: str, onnx_path: Path) -> None:
        import onnxruntime as ort
        from transformers import AutoImageProcessor

        if not onnx_path.exists():
            raise FileNotFoundError(f"Thieu file ONNX: {onnx_path}")
        self._processor = AutoImageProcessor.from_pretrained(model_id)
        options = ort.SessionOptions()
        options.intra_op_num_threads = _ORT_INTRA_THREADS
        options.inter_op_num_threads = _ORT_INTER_THREADS
        # Chỉ định CPU rõ ràng: wheel còn liệt kê AzureExecutionProvider.
        self._session = ort.InferenceSession(
            str(onnx_path), options, providers=["CPUExecutionProvider"]
        )
        logger.info("Depth backend: onnxruntime (%s)", onnx_path)

    def estimate(self, rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        pixel_values = np.ascontiguousarray(
            self._processor(images=rgb, return_tensors="np")["pixel_values"],
            dtype=np.float32,
        )
        raw = self._session.run(["predicted_depth"], {"pixel_values": pixel_values})[0]
        prediction = np.asarray(raw[0], dtype=np.float32)
        target_h, target_w = rgb.shape[:2]
        if prediction.shape != (target_h, target_w):
            # Thay cho torch bicubic interpolate của _DepthAnythingBackend. Hai
            # kernel bicubic không cho kết quả giống từng bit, nhưng phải resize
            # ở ĐÂY (không để normalize_depth_map làm) để phân vị 2–98% được
            # tính trên cùng độ phân giải như đường torch.
            prediction = cv2.resize(
                prediction, (target_w, target_h), interpolation=cv2.INTER_CUBIC
            )
        return np.ascontiguousarray(prediction, dtype=np.float32)
