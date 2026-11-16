#!/usr/bin/env python3
"""Đo thị sai (parallax) của từng lớp độ sâu, bằng số học qua ma trận Kooima.

CHỈ ĐỌC. Không cần OpenGL, không cần webcam, không sửa gì.

Cách đo: đặt mắt ảo ở X = −100, 0, +100 mm (Z = khoảng cách xem), dựng ma trận
chiếu lệch trục (off-axis projection) thật bằng
``src.math.projection.compute_off_axis_projection``, rồi chiếu một điểm world
cố định nằm trên từng lớp xuống pixel màn hình. Độ dịch pixel khi mắt đi 100 mm
chính là thị sai của lớp đó.

Đối chiếu với công thức giải tích: với vật ở độ sâu ``t`` SAU mặt phẳng màn
hình và mắt cách màn hình ``D``, hệ số dịch là ``t / (D + t)``.

LƯU Ý VỀ TOẠ ĐỘ — chỗ dễ đọc sai nhất:
    ``layer_z_near_m`` / ``layer_z_far_m`` trong content.json là toạ độ CỤC BỘ
    của content plane, KHÔNG phải độ sâu sau màn hình. Lớp được dựng ở z cục
    bộ rồi cả mặt phẳng bị dịch đi ``plane_position_m`` (hiện là −0.75 m)
    trong ``ContentPlane.draw``. Nên độ sâu thật sau màn hình là
    ``|plane_position_z| + |layer_z|``, không phải ``|layer_z|``.

Cách chạy:
    .\\.venv\\Scripts\\python.exe scripts\\measure_parallax.py
    .\\.venv\\Scripts\\python.exe scripts\\measure_parallax.py --eye-distance 440
    .\\.venv\\Scripts\\python.exe scripts\\measure_parallax.py --z-far -0.80
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.content.layer_clustering import (  # noqa: E402
    assign_layer_z,
    inter_layer_gap_m,
)
from src.math.projection import compute_off_axis_projection  # noqa: E402

CONTENT_JSON = PROJECT_ROOT / "assets" / "config" / "content.json"
SCREEN_JSON = PROJECT_ROOT / "assets" / "config" / "screen.json"
NEAR_M = 0.01
FAR_M = 8.0
VIEWPORT_W = 1280
VIEWPORT_H = 720


def _load(path: Path) -> dict:
    """Đọc JSON, trả dict rỗng nếu vắng mặt."""
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    """Đọc tham số dòng lệnh. Trả về Namespace."""
    content = _load(CONTENT_JSON)
    parser = argparse.ArgumentParser(description="Đo thị sai từng lớp")
    parser.add_argument("--eye-distance", type=float, default=600.0,
                        help="khoảng cách mắt → màn hình, mm")
    parser.add_argument("--eye-step", type=float, default=100.0,
                        help="độ dịch ngang của mắt, mm")
    parser.add_argument("--layers", type=int,
                        default=int(content.get("layer_count", 3)))
    parser.add_argument("--z-near", type=float,
                        default=float(content.get("layer_z_near_m", 0.0)))
    parser.add_argument("--z-far", type=float,
                        default=float(content.get("layer_z_far_m", -0.35)))
    parser.add_argument("--plane-z", type=float, default=None,
                        help="mặc định lấy từ content.json plane_position_m[2]")
    parser.add_argument("--aspect", type=float, default=736.0 / 552.0,
                        help="tỉ lệ ảnh; chiều cao nội dung suy từ đây khi "
                             "plane_height_m = 0 (mặc định t21.png 736x552)")
    return parser.parse_args(argv)


def _overscan_scale(t_mm: float, content_w_mm: float, content_h_mm: float,
                    screen_w_mm: float, screen_h_mm: float,
                    e_max_mm: float, design_d_mm: float) -> float:
    """Hệ số phóng quad cho một lớp — cùng công thức với ContentPlaneRenderer."""
    if e_max_mm <= 0.0 or t_mm <= 1e-6:
        return 1.0
    d = max(design_d_mm, 1.0)
    need_w = screen_w_mm * 0.5 * (d + t_mm) / d + e_max_mm * t_mm / d
    need_h = screen_h_mm * 0.5 * (d + t_mm) / d + e_max_mm * t_mm / d
    return max(1.0, need_w / (content_w_mm / 2), need_h / (content_h_mm / 2))


def _screen() -> tuple[float, float, np.ndarray]:
    """Kích thước màn hình vật lý (m) và tâm. screen.json thắng, giống app.py."""
    data = _load(SCREEN_JSON)
    width = float(data.get("screen_width", 0.344))
    height = float(data.get("screen_height", 0.194))
    center = np.asarray(data.get("screen_center", [0.0, 0.0, 0.0]), dtype=np.float64)
    return width, height, center


def _pixel_x(point_m: np.ndarray, eye_m: np.ndarray,
             width: float, height: float, center: np.ndarray) -> float:
    """Chiếu một điểm world xuống toạ độ pixel X, qua đúng ma trận của renderer."""
    half_w, half_h = width * 0.5, height * 0.5
    cx, cy, cz = center
    pa = np.array([cx - half_w, cy - half_h, cz])   # góc dưới-trái
    pb = np.array([cx + half_w, cy - half_h, cz])   # góc dưới-phải
    pc = np.array([cx - half_w, cy + half_h, cz])   # góc trên-trái
    projection, view = compute_off_axis_projection(pa, pb, pc, eye_m, NEAR_M, FAR_M)
    clip = projection @ view @ np.array([*point_m, 1.0], dtype=np.float64)
    if abs(clip[3]) < 1e-12:
        return float("nan")
    return float((clip[0] / clip[3] * 0.5 + 0.5) * VIEWPORT_W)


def main(argv: list[str] | None = None) -> int:
    """In bảng thị sai từng lớp. Trả về exit code."""
    args = parse_args(argv)
    content = _load(CONTENT_JSON)
    plane_z = (
        args.plane_z if args.plane_z is not None
        else float(content.get("plane_position_m", [0.0, 0.0, -0.75])[2])
    )
    width, height, center = _screen()
    screen_z = float(center[2])
    eye_d_m = args.eye_distance * 0.001
    step_m = args.eye_step * 0.001

    print(f"man hinh    : {width * 1000:.0f} x {height * 1000:.0f} mm, "
          f"viewport {VIEWPORT_W}x{VIEWPORT_H}")
    print(f"mat         : Z = {args.eye_distance:.0f} mm truoc man hinh, "
          f"dich +-{args.eye_step:.0f} mm")
    print(f"content plane: z = {plane_z:+.3f} m  "
          f"(= {abs(plane_z) * 1000:.0f} mm SAU man hinh)")
    print(f"lop         : K = {args.layers}, z cuc bo {args.z_near:+.3f} .. "
          f"{args.z_far:+.3f} m")
    gap = inter_layer_gap_m(args.z_near, args.z_far, args.layers)
    print(f"khoang cach giua hai lop ke nhau = {gap * 1000:.1f} mm\n")

    print(f"{'lop':>4} {'z cuc bo':>10} {'z world':>10} {'t sau man hinh':>16} "
          f"{'dich px /100mm':>16} {'t/(D+t) do duoc':>17} {'giai tich':>11}")
    print("-" * 92)

    rows = []
    for rank in range(args.layers):
        z_local = assign_layer_z(rank, args.layers, args.z_near, args.z_far)
        z_world = plane_z + z_local
        t_m = screen_z - z_world          # do sau SAU man hinh, duong
        point = np.array([0.0, 0.0, z_world])

        px = []
        for dx in (-step_m, 0.0, step_m):
            eye = np.array([dx, 0.0, screen_z + eye_d_m])
            px.append(_pixel_x(point, eye, width, height, center))
        shift_px = (px[2] - px[0]) * 0.5     # trung binh hai chieu, tren 100 mm

        # Quy doi pixel -> he so dich: 1 px = width/VIEWPORT_W met tren man hinh
        mm_per_px = width * 1000.0 / VIEWPORT_W
        factor_measured = abs(shift_px) * mm_per_px / args.eye_step
        factor_analytic = t_m / (eye_d_m + t_m)

        rows.append((rank, z_local, z_world, t_m * 1000, shift_px,
                     factor_measured, factor_analytic))
        print(f"{rank:4d} {z_local:+9.3f}m {z_world:+9.3f}m {t_m * 1000:13.0f} mm "
              f"{shift_px:+13.1f} px {factor_measured:17.4f} {factor_analytic:11.4f}")

    # --- Ngưỡng lộ vùng trống ------------------------------------------------
    #
    # Ở độ sâu t, vùng nhìn thấy trên mặt phẳng nội dung trải
    #     ±(w/2)·(D+t)/D  −  e·t/D
    # nên muốn không lộ trống tới ±e_max, nửa bề rộng nội dung phải ≥
    #     (w/2)·(D+t)/D + e_max·t/D
    #
    # CHÚ Ý: đây KHÔNG phải "tỉ lệ ảnh lọt khung". 100% lọt khung nghĩa là có
    # vùng trống bao quanh — tức HỎNG. Chỉ số để quyết định là cột e_void.
    content_w_mm = abs(float(content.get("plane_width_m", 1.28))) * 1000.0
    content_h_mm = abs(float(content.get("plane_height_m", 0.0))) * 1000.0
    if content_h_mm <= 1e-6:
        content_h_mm = content_w_mm / max(float(args.aspect), 1e-6)
    screen_w_mm, screen_h_mm = width * 1000.0, height * 1000.0
    d_mm = args.eye_distance
    # Overscan phóng quad từng lớp; nếu bỏ qua nó thì bảng này sẽ mâu thuẫn
    # với dòng log lúc chạy. Cùng công thức, cùng mặc định.
    e_max = float(content.get("overscan_e_max_mm", 150.0))
    design_d = float(content.get("overscan_design_distance_mm", 600.0))

    print(f"\n=== Nguong lo vung trong (overscan e_max={e_max:.0f} mm, "
          f"thiet ke @ D={design_d:.0f} mm) ===")
    print(f"noi dung : {content_w_mm:.0f} x {content_h_mm:.0f} mm "
          f"(nua be rong {content_w_mm / 2:.0f} mm, nua chieu cao {content_h_mm / 2:.0f} mm)")
    print(f"man hinh : {screen_w_mm:.0f} x {screen_h_mm:.0f} mm, ti le anh {args.aspect:.3f}")
    print()
    print(f"{'lop':>4} {'t':>8} {'nua rong CAN cho e_max':>32} "
          f"{'e_void X':>10} {'e_void Y':>10} {'CHAM TRUOC':>12}")
    print(f"{'':>4} {'':>8} {'100mm':>10}{'150mm':>11}{'200mm':>11} "
          f"{'':>10} {'':>10} {'':>12}")
    print("-" * 82)
    for rank, _zl, _zw, t_mm, _px, _fm, _fa in rows:
        need = [screen_w_mm / 2 * (d_mm + t_mm) / d_mm + e * t_mm / d_mm
                for e in (100.0, 150.0, 200.0)]
        k = _overscan_scale(t_mm, content_w_mm, content_h_mm,
                            screen_w_mm, screen_h_mm, e_max, design_d)
        e_x = (content_w_mm / 2 * k
               - screen_w_mm / 2 * (d_mm + t_mm) / d_mm) * d_mm / t_mm
        e_y = (content_h_mm / 2 * k
               - screen_h_mm / 2 * (d_mm + t_mm) / d_mm) * d_mm / t_mm
        binding = "X (ngang)" if e_x <= e_y else "Y (doc)"
        print(f"{rank:4d} {t_mm:7.0f}mm {need[0]:9.0f}mm{need[1]:10.0f}mm"
              f"{need[2]:10.0f}mm {e_x:9.0f}mm {e_y:9.0f}mm {binding:>12}")

    def _pair(t_mm: float) -> float:
        k = _overscan_scale(t_mm, content_w_mm, content_h_mm,
                            screen_w_mm, screen_h_mm, e_max, design_d)
        return min(
            (content_w_mm / 2 * k - screen_w_mm / 2 * (d_mm + t_mm) / d_mm) * d_mm / t_mm,
            (content_h_mm / 2 * k - screen_h_mm / 2 * (d_mm + t_mm) / d_mm) * d_mm / t_mm,
        )

    worst = min(_pair(r[3]) for r in rows)
    print(f"\n  >>> VOID XUAT HIEN TU +-{worst:.0f} mm (lop cham truoc quyet dinh)")
    print(f"  du dia con lai so voi e_max=150 mm: {worst - 150.0:+.0f} mm")

    near_f = rows[-1][6]
    far_f = rows[0][6]
    print(f"\n  ti so thi sai xa/gan = {far_f / near_f:.3f}x "
          f"(day la thu nao doc ra 'chieu sau')")
    print(f"  chenh lech dich px giua lop xa va lop gan = "
          f"{abs(rows[0][4] - rows[-1][4]):.1f} px tren {args.eye_step:.0f} mm")
    max_err = max(abs(r[5] - r[6]) for r in rows)
    print(f"  lech giua do duoc va giai tich: max {max_err:.2e} "
          f"(kiem tra bo do dung ma tran that)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
