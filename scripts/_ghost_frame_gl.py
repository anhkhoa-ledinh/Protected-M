r"""
TASK_ghost_ab_1111 Phase 1 — frame_px qua đúng cơ chế đánh dấu của ``void_pct``
(so hai lượt dựng, một lượt tô riêng node ``window_frame``), vì so màu trực
tiếp với ``frame_color`` ăn lẫn vào các vùng tối của chính ảnh (đo thử: hàng
chục nghìn điểm dương giả trên ảnh có tóc/áo đen).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _layer_review_one import content_for, viewport_slice  # noqa: E402
from src.config.loader import merge_user_config  # noqa: E402

MARK = (1.0, 0.0, 1.0)
MARK_MIN = 8
EYE_Y_MM, EYE_Z_MM = 0.0, 600.0
EYES = (-120.0, 0.0, 120.0)
WINDOW_SIZE = (1280, 720)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--variant", required=True)
    ap.add_argument("--keys", default="{}")
    args = ap.parse_args(argv)

    from _render_eyes_gl import OffscreenApp
    from src.engine.app import HcpdApplication
    from src.renderer.window_plane import window_plane

    config = merge_user_config(ROOT / "assets/config/default.json")
    holder = type("ConfigHolder", (), {"_config": config})()
    screen_cal = HcpdApplication._load_screen_calibration(holder, ROOT)
    content = content_for(args.image, json.loads(args.keys),
                          "assets/config/content.game.json")

    app = OffscreenApp(config, content, screen_cal)
    try:
        if app.wait_ready((0.0, EYE_Y_MM, EYE_Z_MM)) is None:
            raise RuntimeError("het gio cho tach lop")
        mon_pos, mon_size, _hz = app._window.monitor_geometry(None)
        win_pos = (mon_pos[0], mon_pos[1] + (mon_size[1] - WINDOW_SIZE[1]) // 2)
        win_plane = window_plane(
            screen_cal["screen_width"], screen_cal["screen_height"],
            screen_cal["screen_center"], mon_pos, mon_size, win_pos, WINDOW_SIZE)
        full_plane = (screen_cal["screen_width"], screen_cal["screen_height"],
                      screen_cal["screen_center"])
        views = [(tag, (x, EYE_Y_MM, EYE_Z_MM), full_plane)
                 for tag, x in zip(("x-120", "x+0", "x+120"), EYES)]
        views.append(("win0", (0.0, EYE_Y_MM, EYE_Z_MM), win_plane))

        def render_all():
            out = {}
            for tag, eye, plane in views:
                app._camera.set_screen(plane[0], plane[1], plane[2])
                app.draw(eye)
                out[tag] = app.draw(eye)
            return out

        plain = render_all()

        # Chi doi mau rieng node "window_frame" — khac _mark_scene (L0) vi no
        # to MOI node de do void toan phan; o day chi can tach dung khung vien.
        found = False
        stack = [app._renderer._scene_root]
        while stack:
            node = stack.pop()
            stack.extend(node.children)
            if node.name == "window_frame":
                node.material.color = MARK
                found = True
        if not found:
            raise RuntimeError("khong thay node window_frame")
        marked = render_all()
    finally:
        app.close()

    row = {"image": args.name, "variant": args.variant}
    for tag in plain:
        a = viewport_slice(plain[tag]).astype(np.int16)
        b = viewport_slice(marked[tag]).astype(np.int16)
        changed = np.abs(a - b).max(axis=2) >= MARK_MIN
        row[f"frame_px_{tag}"] = int(changed.sum())
    row["frame_px_max"] = max(row[f"frame_px_{t}"] for t in plain)
    print(json.dumps(row), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
