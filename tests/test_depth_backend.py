"""PB-060 Phase 2: chọn engine chạy depth bằng key ``depth_backend``."""

from __future__ import annotations

import unittest
from pathlib import Path

import cv2
import numpy as np
import pytest

from src.config.schema import ContentSettings
from src.content.depth_estimator import (
    ONNX_DEPTH_RELPATH,
    DepthEstimator,
    _OnnxRuntimeBackend,
)

ROOT = Path(__file__).resolve().parents[1]
ONNX_PATH = ROOT / ONNX_DEPTH_RELPATH
FIXTURE = ROOT / "assets" / "images" / "test_pattern.jpg"

# Ngưỡng chốt trước ở PB-060 Phase 1 (TASK_onnx.md), không chỉnh sau khi đo.
MAX_ABS_DIFF_THRESHOLD = 1e-3


class TestDepthBackendSelection(unittest.TestCase):
    """Mặc định phải là torch — đổi engine chỉ khi config nói rõ."""

    def test_default_config_is_torch(self) -> None:
        self.assertEqual(ContentSettings().depth_backend, "torch")
        self.assertEqual(ContentSettings.from_dict({}).depth_backend, "torch")

    def test_estimator_defaults_to_torch(self) -> None:
        # Không nạp model: chỉ kiểm tên engine đã chọn (DepthEstimator lazy).
        self.assertEqual(DepthEstimator("depth-anything-small")._backend_name, "torch")

    def test_config_selects_onnxruntime(self) -> None:
        loaded = ContentSettings.from_dict({"depth_backend": "onnxruntime"})
        self.assertEqual(loaded.depth_backend, "onnxruntime")
        estimator = DepthEstimator(loaded.depth_model, loaded.depth_backend)
        self.assertEqual(estimator._backend_name, "onnxruntime")

    def test_unknown_backend_name_still_loads_config(self) -> None:
        # Tên lạ không được làm chết app: _create_backend sẽ rơi về torch.
        loaded = ContentSettings.from_dict({"depth_backend": "tensorrt"})
        self.assertEqual(loaded.depth_backend, "tensorrt")

    def test_missing_onnx_file_raises_before_loading_anything(self) -> None:
        # Cửa cho nhánh rơi về torch trong _create_backend: thiếu file thì
        # ném FileNotFoundError ngay, chưa nạp processor hay tạo session.
        with self.assertRaises(FileNotFoundError):
            _OnnxRuntimeBackend(
                "LiheYoung/depth-anything-small-hf", ROOT / "khong-ton-tai.onnx"
            )


@unittest.skipUnless(ONNX_PATH.exists(), f"Thieu file ONNX: {ONNX_PATH}")
class TestOnnxMatchesTorch(unittest.TestCase):
    """
    ORT và torch phải cho cùng depth map trong ngưỡng Phase 1.

    Chạy cả hai engine thật nên test này nặng (nạp 2 model + 2 lượt inference);
    bỏ qua khi không có file .onnx (file nằm trong ``artifacts/``, gitignored).

    TASK_ci_1116: cả hai backend đều gọi ``AutoImageProcessor.from_pretrained``
    cho CÙNG model id (xem ``_OnnxRuntimeBackend``'s docstring) — thiếu file
    .onnx không phải lý do duy nhất để bỏ qua; model HF chưa có trong cache
    cục bộ (CI không bao giờ được tải nó) cũng phải bỏ qua, không phải lỗi.
    """

    @pytest.fixture(autouse=True)
    def _inject_model_files(self, model_files):
        self.model_files = model_files

    def test_onnx_matches_torch_on_fixture_image(self) -> None:
        self.model_files.require_depth_anything()
        bgr = cv2.imread(str(FIXTURE), cv2.IMREAD_COLOR)
        self.assertIsNotNone(bgr, f"Khong doc duoc anh fixture: {FIXTURE}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        torch_depth = DepthEstimator("depth-anything-small", "torch").estimate(rgb)
        onnx_depth = DepthEstimator("depth-anything-small", "onnxruntime").estimate(rgb)

        self.assertEqual(onnx_depth.shape, torch_depth.shape)
        self.assertEqual(onnx_depth.shape, rgb.shape[:2])
        max_abs_diff = float(np.max(np.abs(onnx_depth - torch_depth)))
        self.assertLessEqual(max_abs_diff, MAX_ABS_DIFF_THRESHOLD, f"max_abs_diff={max_abs_diff:.8f}")


if __name__ == "__main__":
    unittest.main()
