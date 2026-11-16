r"""
Bang vat an C1/C1b: 7 anh, n = 3 vat, do 3 lan lay trung vi.

Chay DUONG VE THAT cua renderer (``_apply_layers``) nhung stub OpenGL bang
``__new__`` + ``mock.patch``, nen khong can cua so GL. Chi doc, khong ghi gi.

    .\.venv\Scripts\python.exe scripts\measure_props.py
"""
import json
import statistics as st
import sys
import time
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

ROOT = Path(r"D:\test_true3D")
sys.path.insert(0, str(ROOT))
from src.config.schema import ContentSettings  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.hidden_props import layer_planes  # noqa: E402
from src.content.hidden_props_search import place_props  # noqa: E402
from src.renderer import content_plane as cp  # noqa: E402

cs = ContentSettings.from_dict(
    json.load(open(ROOT / "assets/config/content.demo.json", encoding="utf-8")))
REST = tuple(float(v) for v in cs.overscan_design_eye_mm)
RADIUS = float(cs.game_prop_radius_mm)
DUMMY = type("M", (), {"STRIDE": 5, "__init__": lambda s, d: None})
est = DepthEstimator(str(cs.depth_model))
IMGS = ["t21", "t23", "t11", "t24", "t25", "t26", "t22"]
print(f"mat nghi {REST} mm   ban kinh {RADIUS:.0f} mm   n=3   "
      f"Dd={cs.overscan_design_distance_mm:.0f}  mode={cs.overscan_mode}  "
      f"relief={cs.relief_front_policy}")

for name in IMGS:
    rgb = cv2.cvtColor(cv2.imread(rf"D:\PycharmProjects\test_images\{name}.png"),
                       cv2.COLOR_BGR2RGB)
    depth = est.estimate(rgb)
    h, w = rgb.shape[:2]
    r = cp.ContentPlaneRenderer.__new__(cp.ContentPlaneRenderer)
    r._content, r._layers, r._mist = cs, [], None
    r._position = tuple(cs.plane_position_m)
    r._screen_w_mm, r._screen_h_mm = 344.0, 194.0
    r._plane_width = float(cs.plane_width_m)
    r._plane_height = 0.0
    r._tex_width, r._tex_height = w, h
    r._layer_shader, r._depth_worker, r._adaptive_k_report = object(), None, None
    r._disocclusion_runtime_enabled = bool(cs.disocclusion_fill_enabled)
    r._mist_runtime_enabled = bool(cs.mist_enabled)
    r._atmospheric_runtime_enabled = bool(cs.atmospheric_enabled)
    r._relief_runtime_enabled = bool(cs.relief_enabled)
    r._last_depth_rgb = rgb
    with mock.patch.object(cp, "TexturedMesh", DUMMY), \
         mock.patch.object(cp.ContentPlaneRenderer, "_create_rgba_texture",
                           lambda self, a: 0):
        r._apply_layers(depth, f"{name}.png")
    planes = layer_planes(r.layer_records())

    times, props = [], []
    for _ in range(3):
        t0 = time.perf_counter()
        props = place_props(planes, 3, RADIUS, REST)
        times.append((time.perf_counter() - t0) * 1000.0)
    zs = [f"{p.z_m * 1000:+.0f}" for p in planes]
    print(f"\n{name}.png  {w}x{h}  K={len(planes)}  z lop={zs}  "
          f"k={len(props)}  dat vat {st.median(times):.0f} ms (trung vi 3)")
    if not props:
        op = [f"L{i} alpha>=128: {100.0 * (p.alpha >= 128).mean():.0f}%"
              for i, p in enumerate(planes)]
        gaps = [f"{(planes[i + 1].z_min_m - planes[i].z_max_m) * 1000:.0f}mm"
                for i in range(len(planes) - 1)]
        print(f"   0 vat — khe giua lop: {gaps} | {op}")
        continue
    # Bien ngang tai moi Z, tu quad THAT cua lop sau nhat (rang buoc chat nhat).
    plate = planes[0]
    hw, cx, t = plate.quad_w_m * 500.0, plate.centre_x_m * 1000.0, -plate.z_m * 1000.0

    def margin(d, sign):
        u = d / (d + t)
        return abs((sign * 172.0 - u * (cx + sign * hw)) / (1.0 - u) - REST[0])

    m450, m330 = (margin(450.0, -1), margin(450.0, +1)), (margin(330.0, -1), margin(330.0, +1))
    print(f"   bien ngang: Z450 trai {m450[0]:.0f} / phai {m450[1]:.0f} mm,"
          f"  Z330 trai {m330[0]:.0f} / phai {m330[1]:.0f} mm")
    for p in props:
        x, y, z = p["pos_mm"]
        r450, r330 = p["reveal_450_mm"], p["reveal_330_mm"]
        ok450 = abs(r450) <= (m450[0] if r450 < 0 else m450[1])
        s330 = "khong" if r330 is None else f"{r330:+.1f}"
        ok330 = (r330 is not None
                 and abs(r330) <= (m330[0] if r330 < 0 else m330[1]))
        print(f"   vat ({x:+6.1f},{y:+6.1f}) z={-z:6.1f}mm  truc={p['reveal_axis']}  "
              f"lo@450={r450:+6.1f} {'sach' if ok450 else 'VOID'}  "
              f"lo@330={s330:>7} {'sach' if ok330 else 'VOID'}")
