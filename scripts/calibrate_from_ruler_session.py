#!/usr/bin/env python3
"""Sinh lại bảng hiệu chỉnh face_scale từ một phiên quay thước (Phase D).

CHỈ ĐỌC. Script này KHÔNG ghi user.json. Nó in ra con số đề xuất; việc ghi
vẫn thuộc về save_face_calibration(). Tách đo khỏi ghi nên chạy lại bao nhiêu
lần cũng không đụng cấu hình.

Vì sao tồn tại: face_scale = 0.9556 đang dùng trong assets/config/user.json
được suy ra từ phiên quay thước của dự án anh em, và user.json nằm trong
.gitignore. Không có script này thì con số đó không ai tái lập được — với một
kết quả sẽ đưa vào luận văn (sai số 1.2 mm trên dải 200 mm) thì đó là lỗ hổng.

CẢNH BÁO QUAN TRỌNG — đọc trước khi so số với dự án anh em:
    tracking_benchmark giải PnP bằng fx = max(w, h) = 1280 px.
    test_true3D giải PnP bằng fx = 1044.4 px (suy từ giả định FOV 63 độ).
    Chênh 22.6%. tvec tỉ lệ THUẬN với fx, nên |tz| của hai dự án lệch đúng
    chừng đó mà cả hai con số đều trông hợp lý. KHÔNG tái dùng |tz| hay
    face_scale của bên kia. Script này luôn giải lại từ video bằng intrinsics
    của DỰ ÁN NÀY.

Cách chạy:
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_from_ruler_session.py
    .\\.venv\\Scripts\\python.exe scripts\\calibrate_from_ruler_session.py --session <duong_dan.json>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pose.calibration import (  # noqa: E402
    FACE_SCALE_MAX,
    FACE_SCALE_MIN,
    CameraIntrinsics,
)
from src.pose.head_pose import HeadPoseEstimator  # noqa: E402
from src.pose.solve_pnp import solve_head_pose  # noqa: E402
from src.tracking.face_tracker import FaceTracker  # noqa: E402
from src.tracking.landmarks import FACE_MODEL_MM  # noqa: E402

DEFAULT_SESSION = Path(
    r"d:\tracking_benchmark\data\ruler_session\session_20260714T140310Z.json"
)
AXIAL = ("ax_500", "ax_600", "ax_700")
LATERAL = ("lat_m200", "lat_m100", "lat_0a", "lat_0b", "lat_p100", "lat_p200")
REFERENCE_MARK = "ax_600"
# Offset webcam theo world mm, lấy từ assets/config/screen.json.
WEBCAM_OFFSET_MM = np.array([0.0, 103.0, 5.0])


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh. Trả về Namespace."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--session", type=Path, default=DEFAULT_SESSION)
    parser.add_argument("--model", type=Path, default=None, help="face_landmarker.task")
    return parser.parse_args(argv)


def _solve_mark(cap, tracker, intrinsics, mark, model):
    """Giải cả cửa sổ giữ của một mốc. Trả về (tvecs Nx3, số khung trượt)."""
    cap.set(cv2.CAP_PROP_POS_FRAMES, mark["frame_start"])
    tvecs: list[np.ndarray] = []
    missed = 0
    for _ in range(mark["frame_start"], mark["frame_end"] + 1):
        ok, frame = cap.read()
        if not ok:
            break
        result = tracker.process(frame)
        if not result.detected or result.landmarks is None:
            missed += 1
            continue
        solved, _rvec, tvec = solve_head_pose(
            model, result.landmarks.image_points, intrinsics, fast=True
        )
        if not solved:
            missed += 1
            continue
        tvecs.append(np.asarray(tvec, dtype=np.float64))
    return (np.stack(tvecs) if tvecs else np.empty((0, 3))), missed


def main(argv: list[str] | None = None) -> int:
    """In bảng trục Z, bảng trục ngang, và face_scale đề xuất."""
    args = parse_args(argv)
    if not args.session.exists():
        raise SystemExit(f"Khong thay {args.session}")
    meta = json.loads(args.session.read_text(encoding="utf-8"))
    marks = {m["label"]: m for m in meta["marks"]}
    video = args.session.with_suffix(".mp4")
    if not video.exists():
        raise SystemExit(f"Khong thay video {video}")

    cap = cv2.VideoCapture(str(video))
    intrinsics = CameraIntrinsics.from_image_size(meta["width"], meta["height"])
    model_path = args.model or (
        PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"
    )
    tracker = FaceTracker(model_path=str(model_path))

    print(f"phien : {args.session.name}")
    print(f"video : {meta['width']}x{meta['height']} @ {meta['fps']:.0f} fps")
    print(f"fx    : {intrinsics.fx:.1f} px  <- CUA DU AN NAY, khong phai cua benchmark\n")

    # --- Trục Z: ba mốc đo thước, model GỐC (chưa nhân face_scale) ---
    print("=== A) Truc Z, model goc (chua hieu chinh) ===")
    print(f"{'moc':8s} {'do thuoc':>10} {'|tz|':>10} {'std':>7} {'n':>5} {'truot':>7} {'face_scale':>12}")
    tz_med, phys = [], []
    for label in AXIAL:
        tvecs, missed = _solve_mark(cap, tracker, intrinsics, marks[label], FACE_MODEL_MM)
        tz = np.abs(tvecs[:, 2])
        z_true = marks[label]["physical_mm"]["z"]
        tz_med.append(float(np.median(tz)))
        phys.append(float(z_true))
        print(f"{label:8s} {z_true:9.0f}mm {np.median(tz):9.1f}mm {tz.std():6.1f} "
              f"{len(tz):5d} {missed:7d} {z_true / np.median(tz):12.4f}")

    tz_arr, ph_arr = np.asarray(tz_med), np.asarray(phys)
    face_scale = float(np.median(ph_arr / tz_arr))

    # --- Tỉ lệ thuần hay phụ thuộc khoảng cách? Đây là câu hỏi của Phase D ---
    print("\n=== B) Sai so la TI LE THUAN hay PHU THUOC KHOANG CACH? ===")
    pct = 100.0 * (tz_arr - ph_arr) / ph_arr
    print(f"  sai so truoc hieu chinh: {np.round(pct, 1)} %")
    spread = float(pct.max() - pct.min())
    print(f"  tan cua phan tram = {spread:.2f} diem")
    print("  -> " + ("TI LE THUAN: mot vo huong sua duoc."
                     if spread < 1.0 else
                     "PHU THUOC KHOANG CACH: mot vo huong KHONG sua duoc, ghi la han che."))
    slope, intercept = np.polyfit(tz_arr, ph_arr, 1)
    print(f"  khop 2 tham so z_phys = a*tz + b: a={slope:.4f}  b={intercept:+.1f} mm")
    print(f"  ep b=0 (tuc face_scale)         : {face_scale:.4f}"
          f"   {'trong cong' if FACE_SCALE_MIN <= face_scale <= FACE_SCALE_MAX else 'NGOAI CONG'}"
          f" [{FACE_SCALE_MIN}, {FACE_SCALE_MAX}]")
    residual = face_scale * tz_arr - ph_arr
    print(f"  Z sau hieu chinh = {np.round(face_scale * tz_arr, 1)} mm")
    print(f"  du               = {np.round(residual, 1)} mm  (max {np.abs(residual).max():.1f} mm)")

    # --- Trục ngang: KHÔNG dùng để khớp, chỉ để kiểm chéo ---
    print("\n=== C) Kiem cheo truc NGANG (khong dung de khop) ===")
    scaled_model = FACE_MODEL_MM * face_scale
    xs, xp = {}, {}
    for label in LATERAL:
        tvecs, _ = _solve_mark(cap, tracker, intrinsics, marks[label], scaled_model)
        xs[label] = float(np.median(tvecs[:, 0]))
        xp[label] = float(marks[label]["physical_mm"]["x"])
    origin = np.mean([xs["lat_0a"], xs["lat_0b"]])
    print(f"{'moc':10s} {'x that':>9} {'x do duoc':>11} {'sai so':>9}")
    errs = []
    for label in LATERAL:
        d = xs[label] - origin
        errs.append(abs(d - xp[label]))
        print(f"{label:10s} {xp[label]:8.0f}mm {d:10.1f}mm {d - xp[label]:+8.1f}mm")
    errs = np.asarray(errs)
    inner = [i for i, l in enumerate(LATERAL) if "200" not in l]
    print(f"  MAE ngang toan bo    = {errs.mean():.1f} mm")
    print(f"  MAE bo hai moc +-200 = {errs[inner].mean():.1f} mm")

    # --- Đầu cuối qua chính HeadPoseEstimator, để lộ offset webcam ---
    print("\n=== D) Dau-cuoi qua HeadPoseEstimator that (co cong offset webcam) ===")
    estimator = HeadPoseEstimator(
        intrinsics=intrinsics, reference_distance_mm=600.0,
        webcam_offset_mm=WEBCAM_OFFSET_MM, fast_solve_pnp=True, face_scale=face_scale,
    )
    print(f"{'moc':8s} {'do thuoc':>10} {'Z he bao ra':>13} {'sai so':>9}")
    e2e = []
    for label in AXIAL:
        mark = marks[label]
        cap.set(cv2.CAP_PROP_POS_FRAMES, mark["frame_start"])
        zs = []
        for i in range(mark["frame_start"], mark["frame_end"] + 1):
            ok, frame = cap.read()
            if not ok:
                break
            result = tracker.process(frame)
            if not result.detected or result.landmarks is None:
                continue
            head = estimator.estimate(result.landmarks, i / float(meta["fps"]))
            if head.detected:
                zs.append(float(head.translation_mm[2]))
        z = float(np.median(zs))
        z_true = mark["physical_mm"]["z"]
        e2e.append(z - z_true)
        print(f"{label:8s} {z_true:9.0f}mm {z:12.1f}mm {z - z_true:+8.1f}mm")
    tracker.close()
    cap.release()

    bias = float(np.mean(e2e))
    print(f"\n  do lech trung binh = {bias:+.1f} mm")
    print(f"  offset Z cua webcam trong screen.json = {WEBCAM_OFFSET_MM[2]:+.1f} mm")
    print("  Hai so nay trung nhau la MO HO GOC TOA DO, khong phai loi:")
    print("    thuoc do toi CAMERA  -> lech nay la DUNG, giu face_scale nhu tren")
    print(f"    thuoc do toi MAN HINH -> face_scale phai la "
          f"{face_scale * (600.0 - WEBCAM_OFFSET_MM[2]) / 600.0:.4f}")
    print("  ruler_test.py khong ghi goc toa do o bat ky dau. Chenh lech 0.8%,")
    print("  nho hon do chinh xac cua chinh phep do thuoc.")

    print(f"\n>>> face_scale de xuat = {face_scale:.4f}")
    print(">>> Script nay KHONG ghi user.json. Ghi bang save_face_calibration().")
    return 0


if __name__ == "__main__":
    sys.exit(main())
