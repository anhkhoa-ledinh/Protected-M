#!/usr/bin/env python3
"""Phát lại chuỗi landmark thật qua từng biến thể solvePnP và so số.

Đo trước khi sửa. Cả bốn biến thể chạy trên CÙNG một file .npz nên khác biệt
không thể đổ cho phiên quay khác nhau.

LƯU Ý đã kiểm chứng: useExtrinsicGuess (khởi tạo bằng nghiệm khung trước) chỉ
có tác dụng với SOLVEPNP_ITERATIVE. EPNP và SQPNP là solver toàn cục (global
solver), đưa seed sai bét vào vẫn ra đúng kết quả cũ — nên "SQPNP + seed thời
gian" không phải một lựa chọn, phải chọn một trong hai.

QUY TẮC của bộ đo này: in CẢ HÀNG chỉ số cho mọi biến thể, không in mỗi cột
mình quan tâm. Lý do nằm ở docs/PORT_TO_TEST_TRUE3D.md — một phiên trước đã
cải thiện rò rỉ pitch 17.4 → 3.7 mm rồi trả giá bằng tỉ lệ tz<0 đi từ 0% lên
73.9%, thứ nó không hề đo.

Nguồn: kooima_mesh_test/compare_solvers.py. Chỉ đổi phần đường dẫn (script ở
đây nằm trong scripts/ nên lùi một cấp bằng parents[1]). Logic đo giữ nguyên.

Cách chạy:
    .\\.venv\\Scripts\\python.exe scripts\\compare_solvers.py
"""

from __future__ import annotations

import statistics as stats
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Chèn gốc dự án vào sys.path trước khi import src.* — cùng quy ước với
# scripts/report_gvf_table.py. parents[1] vì file này nằm trong scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.calibration import CameraIntrinsics  # noqa: E402
from src.pose.head_pose import DIST_SANE_MM, _reprojection_rmse_px  # noqa: E402
from src.tracking.landmarks import FACE_MODEL_MM  # noqa: E402

NPZ_PATH = PROJECT_ROOT / "artifacts" / "landmarks.npz"

# DIST_SANE_MM import từ head_pose — MỘT nguồn sự thật, không giữ bản sao.
# Bản gốc của script này (kooima_mesh_test) tự định nghĩa (200, 1500); cận dưới
# đã hạ xuống 120 vì 200 bắt nhầm khung người dùng ghé sát thật. Lý do đầy đủ
# nằm ở chỗ khai báo trong src/pose/head_pose.py.

# (tên, flag OpenCV, có seed nghiệm khung trước hay không)
VARIANTS = (
    ("EPNP (dang dung)", cv2.SOLVEPNP_EPNP, False),
    ("ITERATIVE", cv2.SOLVEPNP_ITERATIVE, False),
    ("ITERATIVE + seed", cv2.SOLVEPNP_ITERATIVE, True),
    ("SQPNP", cv2.SOLVEPNP_SQPNP, False),
)


def replay(points: np.ndarray, intrinsics: CameraIntrinsics, flag: int, seed: bool):
    """
    Chạy cả chuỗi qua một biến thể solver.

    points: (N, 6, 2) landmark. flag: cờ solvePnP. seed: dùng nghiệm khung
    trước làm khởi tạo hay không.
    Trả về dict các chỉ số đã tổng hợp.
    """
    last_r = np.zeros((3, 1))
    last_t = np.array([[0.0], [0.0], [600.0]])
    translations: list[np.ndarray | None] = []
    rmses: list[float] = []
    negatives = 0
    insane = 0
    elapsed = 0.0

    for image_points in points:
        start = time.perf_counter()
        ok, rvec, tvec = cv2.solvePnP(
            FACE_MODEL_MM, image_points, intrinsics.matrix, intrinsics.dist_coeffs,
            last_r.copy(), last_t.copy(), useExtrinsicGuess=seed, flags=flag,
        )
        elapsed += time.perf_counter() - start
        if not ok:
            translations.append(None)
            continue
        translation = tvec.ravel()
        if translation[2] <= 0.0:
            negatives += 1
            translations.append(None)
            continue
        if not DIST_SANE_MM[0] <= float(np.linalg.norm(translation)) <= DIST_SANE_MM[1]:
            # Nghiệm RÁC: tz dương nên lọt qua cổng tz<0, nhưng khoảng cách vô
            # lý. Đếm riêng — đây chính là chỉ số trước đây KHÔNG ai đo, nên
            # "0% tz<0" từng bị hiểu nhầm thành "hết lỗi".
            insane += 1
        rmses.append(_reprojection_rmse_px(
            FACE_MODEL_MM, image_points, intrinsics, rvec, tvec))
        translations.append(translation)
        if seed:
            # Chỉ nạp lại seed từ nghiệm HỢP LỆ; seed bằng nghiệm lật thì
            # ràng buộc thời gian sẽ khoá luôn vào nhánh gương.
            last_r, last_t = rvec.copy(), tvec.copy()

    return _summarise(negatives, insane, len(points), rmses, translations, elapsed)


def _summarise(negatives, insane, total, rmses, translations, elapsed) -> dict:
    """Gom cac chi so cua mot bien the thanh dict de in bang."""
    return {
        "neg": negatives, "insane": insane, "n": total,
        "rmse": stats.median(rmses) if rmses else float("nan"),
        "jitter": _median_step(translations),
        "ms": 1000.0 * elapsed / total,
    }


def _median_step(translations: list[np.ndarray | None]) -> float:
    """Trung vị bước nhảy tịnh tiến giữa hai khung hợp lệ liên tiếp, mm."""
    steps = [
        float(np.linalg.norm(b - a))
        for a, b in zip(translations, translations[1:])
        if a is not None and b is not None
    ]
    return stats.median(steps) if steps else float("nan")


def main() -> int:
    """So bốn biến thể trên cùng chuỗi landmark. Trả về exit code."""
    if not NPZ_PATH.exists():
        raise SystemExit(f"Chua co {NPZ_PATH} — chay capture_landmarks.py truoc")
    data = np.load(NPZ_PATH)
    points = data["image_points"]
    intrinsics = CameraIntrinsics.from_image_size(
        int(data["width"]), int(data["height"])
    )
    print(f"{len(points)} khung thuc tu {NPZ_PATH.name}\n")
    print(f"{'bien the':24s} {'tz<0':>12} {'dist rac':>12} {'reproj med':>11} "
          f"{'jitter med':>11} {'ms/khung':>9}")
    print("-" * 86)
    for name, flag, seed in VARIANTS:
        result = replay(points, intrinsics, flag, seed)
        rate = 100.0 * result["neg"] / result["n"]
        bad = 100.0 * result["insane"] / result["n"]
        print(f"{name:24s} {result['neg']:5d} ({rate:4.1f}%) "
              f"{result['insane']:5d} ({bad:4.1f}%) {result['rmse']:10.2f}px "
              f"{result['jitter']:9.1f}mm {result['ms']:8.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
