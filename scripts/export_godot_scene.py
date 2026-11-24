r"""
TASK_godot_port_1111 Phase 1 — xuất bộ lớp đã nướng sang Godot.

File MỚI duy nhất được phép thêm vào cây nộp bài (``D:\hcpd_demo`` chỉ-đọc ở
mọi chỗ khác). Đọc cache nướng qua ĐÚNG ``src.content.bake.BakeWorker`` mà
app dùng (tiến trình con, không đẩy onnxruntime vào luồng gọi script này —
nướng nếu thiếu cache, trúng cache nếu đã có).

Ghi ra ``D:\hcpd_godot\scenes_data\<tên ảnh>\``:
    layerNN.png   — RGBA, alpha THẲNG (không nhân trước), đúng mảng
                    ``BakedLayer.rgba`` ghi thẳng ra đĩa.
    scene.json    — mỗi lớp {file, z_m, width_m, height_m, centre_m}, cộng
                    kích thước màn và các khoá hồ sơ đã dùng.

Kích thước lớp: MỘT hệ số ``s`` DÙNG CHUNG cho mọi lớp (không phải một hệ số
riêng từng lớp như chế độ "scale" cũ của Python — đó chính là nguồn làm lớp
mất đăng ký, xem ``results/layers_1110/ghost_ab/reg_offset.csv``). Chọn ``s``
nhỏ nhất để LỚP XA NHẤT phủ kín màn ở mắt x = ±120 mm, D = 600 mm; áp đúng
``s`` đó cho MỌI lớp nên khoảng cách tương đối giữa các lớp trên màn giữ
nguyên ở mọi vị trí mắt — đăng ký không đổi.

Dùng:
    .venv\Scripts\python.exe scripts\export_godot_scene.py <đường dẫn ảnh>
    .venv\Scripts\python.exe scripts\export_godot_scene.py <ảnh> --name u2_cat
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

GODOT_ROOT = Path(r"D:\hcpd_godot")
GODOT_SCENES = GODOT_ROOT / "scenes_data"
GAME_PROFILE_PATH = ROOT / "assets" / "config" / "content.game.json"
SCREEN_JSON_PATH = ROOT / "assets" / "config" / "screen.json"

#: TASK_godot_port_1111 Phase 1 — mắt thử khi chọn hệ số phủ kín.
COVER_EYE_X_MM = 120.0
COVER_EYE_D_MM = 600.0
#: Những khoá hồ sơ GHI LẠI trong scene.json để Godot/người đọc biết đã dùng
#: giá trị nào (không phải để Godot ĐỌC lại file Python).
RECORDED_PROFILE_KEYS = (
    "layer_z_near_m", "layer_z_far_m", "layer_count", "adaptive_k_enabled",
    "edge_style", "edge_guided_refine", "fill_method", "plate_fill",
    "relief_enabled", "far_fog_mode",
)


def uniform_cover_scale(base_w_m: float, base_h_m: float, far_z_m: float,
                        screen_w_m: float, screen_h_m: float) -> float:
    """
    Hệ số ``s`` nhỏ nhất để một quad rộng ``base_w_m × s`` ở độ sâu
    ``far_z_m`` (so với mặt kính z = 0) phủ kín màn ``screen_w_m ×
    screen_h_m`` khi mắt ở x = ±``COVER_EYE_X_MM``, z = ``COVER_EYE_D_MM``.

    Suy từ phép chiếu tâm (cùng công thức ``project_to_design``): tia từ mắt
    qua MỖI mép màn, kéo tới độ sâu lớp, phải vẫn nằm trong mép quad. Trường
    hợp xấu nhất luôn ở mắt và mép màn NGƯỢC dấu nhau — giải trực tiếp ra
    công thức đóng, không cần dò.
    """
    d_m = COVER_EYE_D_MM / 1000.0
    e_m = COVER_EYE_X_MM / 1000.0
    perspective_s = (d_m - float(far_z_m)) / d_m     # > 1, far_z_m <= 0
    k_x = (2.0 * e_m * (perspective_s - 1.0) + perspective_s * screen_w_m) / base_w_m
    k_y = (perspective_s * screen_h_m) / base_h_m
    return max(k_x, k_y, 1.0)


def _wait_for_bake(image_path: Path, content):
    """Nướng (hoặc trúng cache) qua ĐÚNG ``BakeWorker`` — không nạp onnx ở đây."""
    from src.content.bake import BakeWorker

    worker = BakeWorker(ROOT)
    key = str(image_path)
    from src.content.media_gate import import_photo

    worker.submit(key, import_photo(image_path), content)
    t0 = time.perf_counter()
    while worker.busy:
        worker.poll()
        time.sleep(0.1)
        if time.perf_counter() - t0 > 900.0:
            raise TimeoutError("het gio cho nuong")
    result = worker.pop_ready(key)
    worker.stop()
    if result is None:
        raise RuntimeError("nuong hong — xem log tien trinh con o tren")
    return result


def export_scene(image_path: Path, name: str | None = None) -> Path:
    from src.config.schema import ContentSettings
    from src.content.layer_clustering import assign_layer_z

    data = json.loads(GAME_PROFILE_PATH.read_text(encoding="utf-8"))
    data["content_path"] = str(image_path)
    content = ContentSettings.from_dict(data)

    screen = json.loads(SCREEN_JSON_PATH.read_text(encoding="utf-8"))
    screen_w_m = float(screen["screen_width"])
    screen_h_m = float(screen["screen_height"])

    result = _wait_for_bake(image_path, content)
    layers = result.layers
    count = len(layers)
    base_w_m = float(content.plane_width_m)
    tex_h, tex_w = layers[0].mask.shape
    base_h_m = base_w_m * (tex_h / max(tex_w, 1))

    for rank, layer in enumerate(layers):
        layer.z_m = assign_layer_z(rank, count, content.layer_z_near_m,
                                   content.layer_z_far_m)
    # Lớp 0 luôn là XA NHẤT (quy ước của cluster_into_layers — far-to-near).
    far_z_m = layers[0].z_m
    scale = uniform_cover_scale(base_w_m, base_h_m, far_z_m, screen_w_m,
                               screen_h_m)

    out_name = name or Path(image_path).stem
    out_dir = GODOT_SCENES / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("layer*.png"):
        old.unlink()

    layer_records = []
    for rank, layer in enumerate(layers):
        file_name = f"layer{rank:02d}.png"
        bgra = cv2.cvtColor(np.ascontiguousarray(layer.rgba, dtype=np.uint8),
                            cv2.COLOR_RGBA2BGRA)
        cv2.imwrite(str(out_dir / file_name), bgra)
        layer_records.append({
            "file": file_name,
            "z_m": float(layer.z_m),
            "width_m": base_w_m * scale,
            "height_m": base_h_m * scale,
            "centre_m": [0.0, 0.0],
        })

    bugs = _export_bugs(layers, content, base_w_m * scale, base_h_m * scale,
                        screen_w_m, screen_h_m, image_path)

    scene = {
        "source_image": str(image_path),
        "k_clusters": int(result.k),
        "screen_m": [screen_w_m, screen_h_m],
        "base_w_m": base_w_m,
        "base_h_m": base_h_m,
        "uniform_scale_s": scale,
        "cover_eye_x_mm": COVER_EYE_X_MM,
        "cover_eye_d_mm": COVER_EYE_D_MM,
        "profile": {key: getattr(content, key) for key in RECORDED_PROFILE_KEYS},
        "layers": layer_records,
        "bugs": bugs,
        "level": LEVEL_FOR_EXPORT,
        "level_seconds": LEVEL_TABLE[LEVEL_FOR_EXPORT][2],
        "level_requested": LEVEL_TABLE[LEVEL_FOR_EXPORT][1],
    }
    (out_dir / "scene.json").write_text(json.dumps(scene, indent=1),
                                        encoding="utf-8")
    print(f"K={result.k} s={scale:.4f} bugs={len(bugs)} -> {out_dir}")
    return out_dir


#: TASK_godot_port_1111 Phase 2 — ván tối giản dùng cấp 2 (8 con, 150 s).
LEVEL_FOR_EXPORT = 2
#: Mắt "nghỉ" giả định cho bản Godot: không có bước đo mắt nghỉ riêng (ngoài
#: phạm vi Phase 2) — dùng thẳng mắt thử của Phase 1 (0, 0, 600 mm).
REST_EYE_MM = (0.0, 0.0, COVER_EYE_D_MM)
LEVEL_TABLE = {1: ("De", 6, 180, 2, False), 2: ("Co ban", 8, 150, 3, False),
              3: ("Kho", 10, 150, 3, True), 4: ("Xam thuc", 12, 120, 4, True),
              5: ("Chau A", 12, 120, 4, True)}


def _export_bugs(layers, content, scaled_w_m: float, scaled_h_m: float,
                 screen_w_m: float, screen_h_m: float,
                 image_path: Path) -> list[dict]:
    """
    Đặt bọ bằng ĐÚNG mã Python (``place_props``/``start_game``) trên hình học
    lớp đã scale ``s`` — KHÔNG dựng lại thuật toán đặt chỗ nào trong Godot.
    Gán cỡ + nguỵ trang bằng ĐÚNG ``src.ui_album.bugs`` (TASK_bugs_look_0130).

    ``screen_w_m``/``screen_h_m`` là màn hình VẬT LÝ thật (``screen.json``),
    dùng để quy đổi vị trí chiếu sang pixel ảnh gốc khi lấy màu nguỵ trang —
    khác hẳn ``scaled_w_m``/``scaled_h_m`` (kích thước QUAD của lớp, chỉ dùng
    để đặt chỗ/ẩn-hiện).
    """
    from src.content.media_gate import import_photo, load_rgb
    from src.content.hidden_props import LayerPlane
    from src.content.prop_game import start_game
    from src.ui_album.bugs import assign_bug_camo, assign_bug_sizes

    planes = [
        LayerPlane(alpha=np.ascontiguousarray(layer.rgba[..., 3]),
                  z_m=float(layer.z_m), quad_w_m=scaled_w_m,
                  quad_h_m=scaled_h_m, centre_x_m=0.0, centre_y_m=0.0,
                  z_min_m=float(layer.z_m), z_max_m=float(layer.z_m))
        for layer in layers
    ]
    radius_mm = float(content.game_prop_radius_mm)
    count = LEVEL_TABLE[LEVEL_FOR_EXPORT][1]
    game = start_game(planes, radius_mm, REST_EYE_MM[:2], REST_EYE_MM[2],
                      time.perf_counter(), Path(image_path).name,
                      count=count)
    assign_bug_sizes(game.props, Path(image_path).name)
    rgb = load_rgb(import_photo(image_path))
    if rgb is not None:
        assign_bug_camo(game.props, rgb, REST_EYE_MM,
                        (screen_w_m * 1000.0, screen_h_m * 1000.0))
    return [{"pos_mm": list(prop["pos_mm"]), "radius_mm": float(radius_mm),
            "size_k": float(prop.get("size_k", 3.0)),
            "camo_rgb": list(prop.get("camo_rgb", (0.0, 0.0, 0.0)))}
           for prop in game.props]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Xuat bo lop sang Godot")
    ap.add_argument("image", help="duong dan anh nguon")
    ap.add_argument("--name", default=None,
                    help="ten thu muc con (mac dinh = ten file anh)")
    args = ap.parse_args(argv)
    export_scene(Path(args.image), args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
