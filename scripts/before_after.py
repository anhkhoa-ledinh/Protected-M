r"""
TASK_game_1110 L — ghép sheet ``baseline`` và ``game`` của cùng một ảnh thành
MỘT tấm trước/sau, cho báo cáo L (8 ảnh: 3 người dùng + 5 t).

Thuần stitching: không mở GL, không nướng lại — chỉ đọc PNG đã có sẵn ở
``results/layers_1110/baseline/`` và ``results/layers_1110/game/``.

    .\.venv\Scripts\python.exe scripts\before_after.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layer_review import EIGHT_SET  # noqa: E402

OUT_ROOT = ROOT / "results" / "layers_1110"


def main() -> int:
    missing = []
    for name in EIGHT_SET:
        before = OUT_ROOT / "baseline" / f"sheet_{name}_baseline.png"
        after = OUT_ROOT / "game" / f"sheet_{name}_game.png"
        if not before.exists() or not after.exists():
            missing.append(name)
            continue
        top = cv2.imread(str(before))
        bottom = cv2.imread(str(after))
        width = max(top.shape[1], bottom.shape[1])
        if top.shape[1] != width:
            top = cv2.resize(top, (width, int(top.shape[0] * width / top.shape[1])))
        if bottom.shape[1] != width:
            bottom = cv2.resize(
                bottom, (width, int(bottom.shape[0] * width / bottom.shape[1])))
        label_top = np.zeros((26, width, 3), np.uint8)
        label_bottom = np.zeros((26, width, 3), np.uint8)
        # HERSHEY khong co em-dash: giu ASCII thuan, cung ky luat voi toast
        # (xem test_toast_is_ascii_only).
        cv2.putText(label_top, f"{name} - BASELINE (demo profile)", (8, 19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(label_bottom, f"{name} - GAME (L2 profile)", (8, 19),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1, cv2.LINE_AA)
        canvas = np.vstack([label_top, top, label_bottom, bottom])
        out = OUT_ROOT / f"before_after_{name}.png"
        cv2.imwrite(str(out), canvas)
        print(f"-> {out}")
    if missing:
        print(f"THIEU sheet cho: {', '.join(missing)}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
