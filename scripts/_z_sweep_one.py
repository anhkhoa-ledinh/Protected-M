r"""
TASK_game_1110 L4 — một ảnh, quét ``layer_z_far_m`` qua 4 giá trị, một phiên GL.

Bốn giá trị dùng CHUNG một lượt nướng (``layer_z_far_m`` không nằm trong
``bake.BAKE_KEYS``): mở ``OffscreenApp`` một lần, chờ lớp xong, rồi với mỗi
z chỉ sửa ``layer_z_far_m`` trên ``ContentSettings`` đang chạy và gọi
``ContentPlaneRenderer.refresh_layers()`` — dựng lại MESH từ bộ lớp đã nướng
sẵn, không tách lớp lại, không mở tiến trình con nào thêm.

Ra:
    - ``sheet_<ảnh>.png`` — 4 cột (z) × 3 hàng (ba vị trí mắt toàn màn);
    - ``<ảnh>.json`` — một hàng mỗi z: ``placed``, ``requested``, ``ratio``,
      ``outside``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _layer_review_one import content_for  # noqa: E402
from src.config.loader import merge_user_config  # noqa: E402

#: Cùng quy ước mắt với L0: toàn màn hình, ba vị trí ngang, z = 600 mm.
EYES = (-120.0, 0.0, 120.0)
EYE_Y_MM, EYE_Z_MM = 0.0, 600.0
#: L4: biên độ thị sai cần chấm.
Z_VALUES = (-0.60, -0.35, -0.22, -0.14)
#: Mức khó NHẤT (12 con) — chỗ phép đặt chỗ phải chịu áp lực lớn nhất.
REQUEST_COUNT = 12
CELL_W = 420


def main(argv=None):
    ap = argparse.ArgumentParser(description="L4: quet z tren mot anh")
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--base", default="assets/config/content.game.json")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)

    from _render_eyes_gl import OffscreenApp
    from src.content.hidden_props import layer_planes
    from src.content.prop_game import start_game
    from src.content.prop_rest_eye import z_rest_mm
    from src.engine.app import HcpdApplication
    from src.game.viewport_filter import VIEWPORT_RECT, filter_props

    config = merge_user_config(ROOT / "assets/config/default.json")
    holder = type("ConfigHolder", (), {"_config": config})()
    screen_cal = HcpdApplication._load_screen_calibration(holder, ROOT)
    content = content_for(args.image, {}, args.base)

    app = OffscreenApp(config, content, screen_cal)
    columns: dict[float, dict[str, object]] = {}
    try:
        t0 = time.perf_counter()
        if app.wait_ready((0.0, EYE_Y_MM, EYE_Z_MM)) is None:
            raise RuntimeError("het gio cho tach lop")
        bake_s = time.perf_counter() - t0
        plane = app._renderer._video_plane
        design_eye = tuple(float(v) for v in content.overscan_design_eye_mm)
        rest = (design_eye[0], design_eye[1], 450.0)
        radius_mm = float(content.game_prop_radius_mm)
        glass_mm = (screen_cal["screen_width"] * 1000.0,
                   screen_cal["screen_height"] * 1000.0)

        for z in Z_VALUES:
            plane._content.layer_z_far_m = float(z)
            if not plane.refresh_layers():
                raise RuntimeError(f"refresh_layers that bai o z={z}")
            planes = layer_planes(plane.layer_records())
            game = start_game(planes, radius_mm, rest[:2], z_rest_mm(rest[2]),
                              time.perf_counter(), args.name, count=REQUEST_COUNT * 3)
            kept, outside = filter_props(game.props, rest, REQUEST_COUNT,
                                         glass_mm, VIEWPORT_RECT, (0.0, 0.0))
            frames = {}
            for eye_x in EYES:
                app.draw((eye_x, EYE_Y_MM, EYE_Z_MM))
                frames[eye_x] = app.draw((eye_x, EYE_Y_MM, EYE_Z_MM))
            columns[z] = {
                "placed": len(kept), "requested": REQUEST_COUNT,
                "ratio": round(len(kept) / REQUEST_COUNT, 3),
                "outside": outside, "raw_candidates": len(game.props),
                "frames": frames,
            }
    finally:
        app.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for z in Z_VALUES:
        col = columns[z]
        rows.append({"image": args.name, "layer_z_far_m": z,
                    "placed": col["placed"], "requested": col["requested"],
                    "ratio": col["ratio"], "outside_viewport": col["outside"],
                    "raw_candidates": col["raw_candidates"],
                    "bake_s": round(bake_s, 2)})
        for eye_x, frame in col["frames"].items():
            cv2.imwrite(
                str(out_dir / f"{args.name}_z{z:+.2f}_x{int(eye_x):+d}.png"),
                cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    (out_dir / f"{args.name}.json").write_text(json.dumps(rows, indent=1),
                                               encoding="utf-8")
    sheet(out_dir / f"sheet_{args.name}.png", columns)
    print(json.dumps(rows), flush=True)
    return 0


def sheet(path, columns) -> None:
    """Bảng 4 cột (z) × 3 hàng (mắt) + nhãn tỉ lệ đặt chỗ mỗi cột."""
    tiles = []
    for z in Z_VALUES:
        col = columns[z]
        imgs = []
        for eye_x in EYES:
            frame = col["frames"][eye_x]
            h, w = frame.shape[:2]
            small = cv2.resize(frame, (CELL_W, max(1, int(round(h * CELL_W / w)))),
                               interpolation=cv2.INTER_AREA)
            imgs.append(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        column_img = np.vstack(imgs)
        label = np.zeros((26, CELL_W, 3), np.uint8)
        text = f"z={z:+.2f}  {col['placed']}/{col['requested']}"
        cv2.putText(label, text, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                   (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(np.vstack([label, column_img]))
    height = max(t.shape[0] for t in tiles)
    canvas = np.zeros((height, CELL_W * len(tiles), 3), np.uint8)
    for i, tile in enumerate(tiles):
        canvas[:tile.shape[0], i * CELL_W:(i + 1) * CELL_W] = tile
    cv2.imwrite(str(path), canvas)


if __name__ == "__main__":
    raise SystemExit(main())
