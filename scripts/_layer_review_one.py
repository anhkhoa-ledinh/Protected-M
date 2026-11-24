r"""
TASK_game_1110 L0 — chấm MỘT ảnh với MỘT profile, trong một tiến trình riêng.

Gọi qua ``scripts/layer_review.py``. Ở đây mới mở ngữ cảnh GL, nên một tiến
trình chỉ có đúng một ngữ cảnh — cùng kỷ luật của ``_render_eyes_one.py``.

Bốn vị trí mắt, tất cả ở z = 600 mm:

- ``x-120`` / ``x+0`` / ``x+120`` với tấm kính = **cả màn hình** (hiệu chỉnh
  ``screen.json``);
- ``win0`` với tấm kính = một cửa sổ 1280×720 **kê sát mép trái** màn chính,
  tính bằng đúng ``window_plane`` mà app dùng ở chế độ cửa sổ (D2).

Mỗi lượt dựng hai lần trong **cùng một phiên GL**: lượt thường, rồi lượt tô
"void" (màu nền, backdrop, hai tấm giữa, khung viền đều đổi sang đỏ tươi).
Điểm nào ĐỔI giữa hai lượt là điểm có đóng góp của thứ **không phải ảnh** —
đó là định nghĩa của ``void_pct``. Hai lượt dùng chung một bộ lớp đã dựng, nên
chỉ tốn đúng MỘT lần tách lớp mỗi (ảnh × profile).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.config.loader import merge_user_config  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.renderer.window_plane import window_plane  # noqa: E402

#: z của mắt theo kịch bản L0; y giữ mắt thiết kế của demo.
EYE_Z_MM = 600.0
EYE_Y_MM = 0.0
#: Ba vị trí toàn màn + một lượt cửa sổ. Tên này cũng là tên cột/ảnh.
FULL_EYES = (-120.0, 0.0, 120.0)
WINDOW_SIZE = (1280, 720)
#: Màu tô "không phải ảnh". Đỏ tươi, xa mọi màu của cảnh thường.
MARK = (1.0, 0.0, 1.0)
#: Chênh lệch tối thiểu (0–255) để coi một điểm là đã đổi giữa hai lượt.
MARK_MIN = 8
#: Ô viewport của màn chơi v3 trong toạ độ thiết kế 1280×720 (xem
#: ``app._viewport_rect``). ΔE và void đều chỉ tính trong ô này.
VIEWPORT_RECT = (28.0, 28.0, 1224.0, 664.0)
DESIGN_SIZE = (1280.0, 720.0)
#: ΔE đo ở cỡ này (kịch bản L0: "cả hai thu về rộng 320 px").
DELTAE_WIDTH = 320


def viewport_slice(frame):
    """Cắt ô viewport ra khỏi một khung đã dựng (tỉ lệ theo cỡ khung thật)."""
    h, w = frame.shape[:2]
    vx, vy, vw, vh = VIEWPORT_RECT
    sx, sy = w / DESIGN_SIZE[0], h / DESIGN_SIZE[1]
    x0, y0 = int(round(vx * sx)), int(round(vy * sy))
    x1, y1 = int(round((vx + vw) * sx)), int(round((vy + vh) * sy))
    return frame[y0:y1, x0:x1]


def void_pct(plain, marked, *, whole=False) -> float:
    """
    % điểm có đóng góp của thứ **không phải ảnh**.

    Mặc định chỉ tính trong ô viewport — đó là tất cả những gì người chơi thấy,
    nên nó là thước đo của cổng L5. ``whole=True`` tính cả khung, kể cả vành
    28 px mà giao diện v3 che đi: giữ cả hai số để một chỗ hở ngoài ô không
    biến mất khỏi báo cáo.
    """
    a, b = (plain, marked) if whole else (viewport_slice(plain),
                                          viewport_slice(marked))
    changed = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return 100.0 * float((changed >= MARK_MIN).mean())


def delta_e_mean(render, original) -> float:
    """
    ΔE76 trung bình giữa khung dựng ở mắt giữa và ảnh gốc.

    Cả hai thu về rộng ``DELTAE_WIDTH`` rồi đổi sang Lab. Ảnh gốc được cắt theo
    **tỉ lệ của ô viewport** chứ không dùng nguyên tấm: ô chỉ hở một phần ảnh.
    """
    view = viewport_slice(render)
    vh, vw = view.shape[:2]
    scale = DELTAE_WIDTH / float(vw)
    size = (DELTAE_WIDTH, max(1, int(round(vh * scale))))
    a = cv2.resize(view, size, interpolation=cv2.INTER_AREA)
    b = cv2.resize(original, size, interpolation=cv2.INTER_AREA)
    lab_a = cv2.cvtColor(a, cv2.COLOR_RGB2Lab).astype(np.float32)
    lab_b = cv2.cvtColor(b, cv2.COLOR_RGB2Lab).astype(np.float32)
    # OpenCV 8-bit Lab: L ở 0–255 (thang 0–100 × 2,55), a/b lệch +128. Quy về
    # thang CIE trước khi lấy chuẩn, nếu không L nặng gấp 2,55 lần đúng mức.
    lab_a[..., 0] *= 100.0 / 255.0
    lab_b[..., 0] *= 100.0 / 255.0
    return float(np.sqrt(((lab_a - lab_b) ** 2).sum(axis=2)).mean())


def edge_step_px(frame) -> float:
    """
    Bậc thang trên mép: độ dài trung bình của đoạn biên **thẳng đứng/ngang**.

    Mép tách lớp bị phóng bằng ``INTER_NEAREST`` từ bản đồ nhãn 640 px để lên
    cỡ gốc, nên đường bao của nó gãy thành bậc. Đo bằng tỉ lệ điểm biên Canny
    nằm trên một đoạn thẳng trục dài ≥ 3 px: càng cao càng giống bậc thang.
    Số thuần trên khung đã dựng, không cần mặt nạ nào.
    """
    gray = cv2.cvtColor(viewport_slice(frame), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160) > 0
    if not bool(edges.any()):
        return 0.0
    run_h = cv2.erode(edges.astype(np.uint8),
                      np.ones((1, 3), np.uint8)) > 0
    run_v = cv2.erode(edges.astype(np.uint8),
                      np.ones((3, 1), np.uint8)) > 0
    return 100.0 * float((run_h | run_v).sum()) / float(edges.sum())


def content_for(image_path, profile_keys, base_json):
    """``ContentSettings`` của ``base_json``, trỏ vào ảnh, cộng khoá profile."""
    data = json.loads((ROOT / base_json).read_text(encoding="utf-8"))
    data["content_path"] = str(image_path)
    data.update(profile_keys)
    return ContentSettings.from_dict(data)


def _mark_scene(renderer, content):
    """Tô mọi thứ KHÔNG phải ảnh bằng ``MARK`` (màu nền + 4 nút cảnh)."""
    renderer._content = replace(content, void_clear_color=MARK)
    stack = [renderer._scene_root]
    while stack:
        node = stack.pop()
        stack.extend(node.children)
        if node.material is not None and node.name != "video_void":
            node.material.color = MARK


def main(argv=None):
    ap = argparse.ArgumentParser(description="L0: mot anh, mot profile")
    ap.add_argument("--image", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--profile", default="baseline")
    ap.add_argument("--base", default="assets/config/content.demo.json")
    ap.add_argument("--keys", default="{}", help="JSON khoa cua profile")
    ap.add_argument("--eye-max", type=float, default=120.0,
                    help="|x| cua hai vi tri mat ben (L5 doi sang 150)")
    ap.add_argument("--win-eye-max", type=float, default=0.0,
                    help="L5: them hai luot cua so o mat +-x nay (0 = tat, "
                         "chi giu dung 4 luot goc cua L0)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    eyes = (-args.eye_max, 0.0, args.eye_max)

    from _render_eyes_gl import OffscreenApp
    from src.engine.app import HcpdApplication

    config = merge_user_config(ROOT / "assets/config/default.json")
    holder = type("ConfigHolder", (), {"_config": config})()
    screen_cal = HcpdApplication._load_screen_calibration(holder, ROOT)
    content = content_for(args.image, json.loads(args.keys), args.base)

    app = OffscreenApp(config, content, screen_cal)
    try:
        t0 = time.perf_counter()
        if app.wait_ready((0.0, EYE_Y_MM, EYE_Z_MM)) is None:
            raise RuntimeError("het gio cho tach lop")
        bake_s = time.perf_counter() - t0
        holes = tuple(app._renderer.layer_hole_stats)
        k = app.layer_count()

        # Tấm kính của lượt cửa sổ: 1280×720 kê sát mép trái màn chính, giữa
        # theo chiều dọc. Tính bằng chính ``window_plane`` của app.
        mon_pos, mon_size, _hz = app._window.monitor_geometry(None)
        win_pos = (mon_pos[0], mon_pos[1] + (mon_size[1] - WINDOW_SIZE[1]) // 2)
        win_plane = window_plane(
            screen_cal["screen_width"], screen_cal["screen_height"],
            screen_cal["screen_center"], mon_pos, mon_size, win_pos, WINDOW_SIZE)
        full_plane = (screen_cal["screen_width"], screen_cal["screen_height"],
                      screen_cal["screen_center"])

        views = [(tag, (x, EYE_Y_MM, EYE_Z_MM), full_plane)
                 for tag, x in zip(("x-120", "x+0", "x+120"), eyes)]
        views.append(("win0", (0.0, EYE_Y_MM, EYE_Z_MM), win_plane))
        if args.win_eye_max > 0.0:
            # L5: cổng đòi void ở mắt +-150 mm CẢ trong cửa sổ, không chỉ toàn
            # màn. Thêm hai lượt cửa sổ lệch mắt — ngoài "4 lượt gốc" của L0,
            # không vẽ vào sheet, chỉ để tính void_pct riêng.
            views.append(("win-side", (-args.win_eye_max, EYE_Y_MM, EYE_Z_MM),
                         win_plane))
            views.append(("win+side", (args.win_eye_max, EYE_Y_MM, EYE_Z_MM),
                         win_plane))

        def render_all():
            out = {}
            for tag, eye, plane in views:
                app._camera.set_screen(plane[0], plane[1], plane[2])
                app.draw(eye)                       # một khung cho ổn định
                out[tag] = app.draw(eye)
            return out

        plain = render_all()
        _mark_scene(app._renderer, content)
        marked = render_all()
    finally:
        app.close()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for tag, frame in plain.items():
        cv2.imwrite(str(out_dir / f"{args.name}_{args.profile}_{tag}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    from src.content.media_gate import import_photo, load_rgb

    original = load_rgb(import_photo(Path(args.image)))
    centre = plain["x+0"]
    row = {
        "image": args.name, "profile": args.profile, "K": k,
        "bake_s": round(bake_s, 2),
        "hole_pct": round(max(holes) if holes else 0.0, 3),
        "hole_per_layer": [round(v, 3) for v in holes],
        "deltaE_mean": round(delta_e_mean(centre, original), 2),
        "edge_step_pct": round(edge_step_px(centre), 2),
    }
    for tag in plain:
        row[f"void_pct_{tag}"] = round(void_pct(plain[tag], marked[tag]), 4)
        row[f"void_full_{tag}"] = round(
            void_pct(plain[tag], marked[tag], whole=True), 4)
    row["void_pct"] = round(max(row[f"void_pct_{t}"] for t in plain), 4)
    row["void_full_pct"] = round(max(row[f"void_full_{t}"] for t in plain), 4)
    (out_dir / f"{args.name}_{args.profile}.json").write_text(
        json.dumps(row, indent=1), encoding="utf-8")
    sheet(out_dir / f"sheet_{args.name}_{args.profile}.png", original, plain)
    print(json.dumps(row), flush=True)
    return 0


def sheet(path, original, plain):
    """Bảng tiếp xúc: ảnh gốc + 4 lượt dựng, mỗi ô rộng 420 px, có nhãn."""
    cell_w = 420
    tiles = [("original", original)] + [(t, plain[t]) for t in
                                        ("x-120", "x+0", "x+120", "win0")]
    imgs = []
    for tag, img in tiles:
        h, w = img.shape[:2]
        small = cv2.resize(img, (cell_w, max(1, int(round(h * cell_w / w)))),
                           interpolation=cv2.INTER_AREA)
        small = cv2.cvtColor(small, cv2.COLOR_RGB2BGR)
        label = np.zeros((22, cell_w, 3), np.uint8)
        cv2.putText(label, tag, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
        imgs.append(np.vstack([label, small]))
    height = max(im.shape[0] for im in imgs)
    canvas = np.zeros((height, cell_w * len(imgs), 3), np.uint8)
    for i, im in enumerate(imgs):
        canvas[:im.shape[0], i * cell_w:(i + 1) * cell_w] = im
    cv2.imwrite(str(path), canvas)


if __name__ == "__main__":
    raise SystemExit(main())
