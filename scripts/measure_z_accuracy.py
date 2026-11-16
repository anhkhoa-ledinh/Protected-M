#!/usr/bin/env python3
"""Đo độ chính xác của Z ở một khoảng cách đã biết (nửa sau của Phase D).

Ngồi yên ở khoảng cách đã đo bằng thước, chạy script này, nó thu N giây rồi
báo trung bình / độ lệch chuẩn / sai số tuyệt đối / sai số phần trăm — đúng bốn
cột mà bảng nghiệm thu Phase D yêu cầu.

Dùng ĐÚNG đường mà app dùng: CameraIntrinsics.from_image_size ->
FaceTracker -> HeadPoseEstimator (tự nạp face_scale từ user.json nếu có) ->
EyePositionEstimator. Không chạy bộ lọc thời gian: bộ lọc thêm độ trễ chứ
không sửa sai lệch, mà ở đây đang đo SAI LỆCH, không đo độ mượt.

Đường Z được báo là translation_mm[2] của eye_pose — tức toạ độ world tính từ
tâm màn hình, đã cộng offset webcam. Đây chính là số đi vào frustum Kooima.

Cách chạy:
    .\\.venv\\Scripts\\python.exe scripts\\measure_z_accuracy.py --true-distance 400
    .\\.venv\\Scripts\\python.exe scripts\\measure_z_accuracy.py --true-distance 600 --seconds 8
"""

from __future__ import annotations

import argparse
import statistics as stats
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.capture import CameraCapture, CaptureConfig  # noqa: E402
from src.config.loader import merge_user_config  # noqa: E402
from src.pose.calibration import (  # noqa: E402
    CameraIntrinsics,
    load_face_calibration,
)
from src.pose.eye_position import EyePositionEstimator  # noqa: E402
from src.pose.head_pose import HeadPoseEstimator  # noqa: E402
from src.tracking.face_tracker import FaceTracker  # noqa: E402

DEFAULT_CONFIG = PROJECT_ROOT / "assets" / "config" / "default.json"
USER_CONFIG = PROJECT_ROOT / "assets" / "config" / "user.json"
SCREEN_JSON = PROJECT_ROOT / "assets" / "config" / "screen.json"


def _webcam_offset_mm() -> np.ndarray:
    """Offset webcam theo mm world. screen.json thắng, giống app.py."""
    import json

    if SCREEN_JSON.exists():
        data = json.loads(SCREEN_JSON.read_text(encoding="utf-8"))
        offset_m = data.get("camera_offset")
        if offset_m is not None:
            return np.asarray(offset_m, dtype=np.float64) * 1000.0
    return np.zeros(3, dtype=np.float64)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh. Trả về Namespace."""
    parser = argparse.ArgumentParser(description="Đo sai số Z ở khoảng cách đã biết")
    parser.add_argument(
        "--true-distance", type=float, required=True,
        help="khoảng cách mắt -> mặt kính màn hình đã đo bằng thước (mm)",
    )
    parser.add_argument("--seconds", type=float, default=5.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Thu Z trong N giây rồi in bảng. Trả về exit code."""
    args = parse_args(argv)

    config = merge_user_config(DEFAULT_CONFIG, USER_CONFIG)
    calibration = load_face_calibration(USER_CONFIG)
    face_scale = calibration.face_scale if calibration is not None else None
    if face_scale is not None:
        print(f"Z path: CALIBRATED (face_scale={face_scale:.4f})")
    else:
        print("Z path: LEGACY (_monocular_scale) — chua chay wizard")

    cam = config.camera
    intrinsics = CameraIntrinsics.from_image_size(cam.width, cam.height)
    offset = _webcam_offset_mm()
    head_pose = HeadPoseEstimator(
        intrinsics=intrinsics,
        reference_distance_mm=config.user.viewing_distance_mm,
        webcam_offset_mm=offset,
        fast_solve_pnp=config.optimization.fast_solve_pnp,
        face_scale=face_scale,
    )
    eye = EyePositionEstimator(
        ipd_mm=config.user.ipd_mm,
        fx=intrinsics.fx, cx=intrinsics.cx, cy=intrinsics.cy,
    )

    capture = CameraCapture(
        CaptureConfig(device_id=cam.device_id, width=cam.width,
                      height=cam.height, fps=cam.fps)
    )
    tracker = FaceTracker(model_path=config.tracking.model_path)

    samples: list[float] = []
    missed = 0
    print(f"\nNgoi yen o {args.true_distance:.0f} mm, thu {args.seconds:.0f} giay...")
    capture.open()
    try:
        deadline = time.perf_counter() + args.seconds
        while time.perf_counter() < deadline:
            packet = capture.read()
            if packet is None:
                continue
            result = tracker.process(packet.image)
            if not result.detected or result.landmarks is None:
                missed += 1
                continue
            head = head_pose.estimate(result.landmarks, time.perf_counter())
            if not head.detected:
                # solvePnP hong — dem rieng, khong tron vao mau.
                missed += 1
                continue
            pose = eye.estimate(head, result.landmarks, time.perf_counter())
            samples.append(float(pose.translation_mm[2]))
    finally:
        tracker.close()
        capture.close()

    if len(samples) < 10:
        print(f"Chi thu duoc {len(samples)} mau — khong du de ket luan.")
        return 1

    mean = stats.mean(samples)
    err = mean - args.true_distance
    print(f"\n{'khoang cach that':>18} {'Z do duoc':>11} {'std':>9} "
          f"{'sai so':>10} {'% sai so':>10} {'n':>6}")
    print("-" * 70)
    print(f"{args.true_distance:15.0f}mm {mean:9.1f}mm {stats.pstdev(samples):7.1f}mm "
          f"{err:+8.1f}mm {100.0 * err / args.true_distance:+9.2f}% {len(samples):6d}")
    print(f"\nkhung bo qua (mat mat hoac solvePnP hong): {missed}")
    print(f"dat neu |sai so| < 50 mm  ->  {'DAT' if abs(err) < 50.0 else 'KHONG DAT'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
