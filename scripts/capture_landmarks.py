#!/usr/bin/env python3
"""Ghi chuỗi landmark (điểm mốc khuôn mặt) thật ra .npz để phát lại offline.

Có bản ghi này thì mọi đề xuất đổi solver (bộ giải) đều so được trên CÙNG dữ
liệu, thay vì mỗi lần chạy lại một phiên khác nhau rồi cãi nhau về con số.

KHÔNG sửa file nào trong src/: dùng thẳng FaceTracker và CameraCapture như thư
viện. Hãy cử động trong lúc ghi — kể cả ghé sát dưới 350 mm, nơi phần lớn
thất bại nằm ở đó.

Nguồn: kooima_mesh_test/capture_landmarks.py. Chỉ đổi phần đường dẫn, vì bên
đó script nằm ở gốc dự án (Path(__file__).parent = gốc) còn ở đây nó nằm trong
scripts/ nên phải lùi một cấp — nếu không, artifacts/ sẽ rơi vào
scripts/artifacts/ thay vì artifacts/ của dự án. Logic đo giữ nguyên từng dòng.

Cách chạy:
    .\\.venv\\Scripts\\python.exe scripts\\capture_landmarks.py --frames 1200
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np

# Chèn gốc dự án vào sys.path trước khi import src.* — cùng quy ước với
# scripts/report_gvf_table.py. parents[1] vì file này nằm trong scripts/.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.camera.capture import CameraCapture, CaptureConfig  # noqa: E402
from src.tracking.face_tracker import FaceTracker  # noqa: E402

OUT_PATH = PROJECT_ROOT / "artifacts" / "landmarks.npz"
MODEL_PATH = PROJECT_ROOT / "assets" / "models" / "face_landmarker.task"

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh. Trả về Namespace."""
    parser = argparse.ArgumentParser(description="Ghi landmark thật ra .npz")
    parser.add_argument("--frames", type=int, default=1200, help="số khung CÓ MẶT")
    parser.add_argument("--out", type=Path, default=OUT_PATH)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Mở webcam, ghi image_points của từng khung có mặt. Trả về exit code."""
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    camera = CameraCapture(CaptureConfig(device_id=0, width=1280, height=720, fps=60))
    camera.open()
    tracker = FaceTracker(model_path=str(MODEL_PATH))

    points: list[np.ndarray] = []
    confidences: list[float] = []
    seen = 0
    started = time.perf_counter()
    try:
        while len(points) < args.frames:
            packet = camera.read()
            if packet is None:
                continue
            seen += 1
            result = tracker.process(packet.image)
            if not result.detected or result.landmarks is None:
                continue
            points.append(result.landmarks.image_points.astype(np.float64).copy())
            confidences.append(float(result.confidence))
            if len(points) % 200 == 0:
                logger.info("da ghi %d/%d khung", len(points), args.frames)
    finally:
        tracker.close()
        camera.close()

    _save(args.out, points, confidences, seen, time.perf_counter() - started)
    return 0


def _save(out: Path, points, confidences, seen: int, elapsed: float) -> None:
    """Ghi .npz va bao so lieu. points: list (6,2); seen: tong khung da doc."""
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out, image_points=np.stack(points), confidence=np.asarray(confidences),
        width=1280, height=720,
    )
    logger.info("%d khung co mat / %d khung doc, %.1f s -> %s",
                len(points), seen, elapsed, out)


if __name__ == "__main__":
    sys.exit(main())
