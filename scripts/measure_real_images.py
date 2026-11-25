r"""
PB-059 (2): chạy đường pipeline của app trên ẢNH THẬT của người dùng, không mở
cửa sổ. Chỉ đọc; không sửa ``src/``, không ghi file config nào.

Đường đi sao y app: ``ImageSource`` (chịu được đường dẫn Unicode) →
``DepthEstimator`` → ``_apply_layers`` THẬT của ``ContentPlaneRenderer``, stub
OpenGL bằng ``__new__`` + ``mock.patch`` y như ``scripts/measure_props.py``.
Phần đo nằm ở ``_real_image_metrics.py``.

``load_to_ready_ms`` = ``nạp ảnh + depth + _apply_layers``, cộng **y hệt nhau
cho cả hai ``layer_method``**: depth không phụ thuộc layer_method nên chỉ chạy
một lần mỗi ảnh, nhưng chi phí của nó được tính vào **cả hai** hàng. Bản
30/10 sáng cộng depth vào mỗi hàng ``kmeans`` mà bỏ khỏi hàng ``kmeans_vote``,
nên hàng vote trông nhanh hơn 1,3–2,1 s dù làm nhiều việc hơn. Model được hâm
một lượt KHÔNG tính giờ trước vòng đo, nên ảnh đầu cũng có ``depth_ms`` nóng
(lượt nạp trọng số 12,5 s chỉ còn ở dòng in ra, không vào CSV).

``seam_hard_frac`` cần mask từng lớp mà ``_apply_layers`` không giữ, nên có
thêm MỘT lượt ``cluster_into_layers`` chỉ để lấy mask — lượt phụ đó KHÔNG tính
vào thời gian.

    .\.venv\Scripts\python.exe scripts\measure_real_images.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _real_image_metrics import (apply_layers, build_stub,  # noqa: E402
                                 preview_png, seam_hard_frac)
from src.config.schema import ContentSettings  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.hidden_props import layer_planes  # noqa: E402
from src.content.hidden_props_search import place_props  # noqa: E402
from src.content.image_source import ImageSource  # noqa: E402

PHOTOS = Path(r"D:\test_true3D space\real_data\photos")
OUT_DIR = Path(r"D:\test_true3D space\task_done\PB-059_real_data")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
N_PROPS = 3
FIELDS = ["file", "scene_type", "layer_method", "width", "height", "depth_ms",
          "K", "n_props_hidden", "seam_hard_frac", "load_to_ready_ms", "error"]
#: PB-070b: chấm cả hai cách gán nhãn lớp trong CÙNG một CSV để so trực tiếp.
LAYER_METHODS = ("kmeans", "kmeans_vote")


def prepare(path, estimator):
    """``(rgb, depth, img_ms, depth_ms)`` — phần KHÔNG phụ thuộc layer_method."""
    t0 = time.perf_counter()
    rgb = ImageSource(import_photo(path)).tick(0.0)
    if rgb is None:
        raise ValueError("ImageSource tra ve None")
    img_ms = (time.perf_counter() - t0) * 1000.0
    t0 = time.perf_counter()
    depth = estimator.estimate(rgb)
    return rgb, depth, img_ms, (time.perf_counter() - t0) * 1000.0


def warm_up(path, estimator):
    """Hâm model một lượt KHÔNG tính giờ; lần gọi đầu gồm cả nạp trọng số."""
    t0 = time.perf_counter()
    prepare(path, estimator)
    print(f"ham model tren {path.name}: "
          f"{(time.perf_counter() - t0) * 1000.0:.0f} ms (gom nap trong so)",
          flush=True)


def error_rows(path, scene_type, exc):
    """Một hàng lỗi cho mỗi ``layer_method`` khi ảnh không nạp/đo được."""
    return [{**{key: "" for key in FIELDS}, "file": path.name,
             "layer_method": method, "scene_type": scene_type,
             "error": f"{type(exc).__name__}: {exc}"}
            for method in LAYER_METHODS]


def run_one(path, cs, scene_type, out_dir, method, prepared):
    """
    Một hàng CSV cho (ảnh, layer_method); lỗi ghi vào cột ``error``.

    ``prepared`` là ``(rgb, depth, img_ms, depth_ms)`` của ``prepare()``: depth
    chạy một lần mỗi ảnh nhưng thời gian của nó vào **cả hai** hàng, nên cột
    ``load_to_ready_ms`` so sánh được giữa hai ``layer_method``.
    """
    from dataclasses import replace

    row = {key: "" for key in FIELDS}
    row["file"], row["scene_type"] = path.name, scene_type
    row["layer_method"] = method
    rgb, depth, img_ms, depth_ms = prepared
    row["height"], row["width"] = rgb.shape[0], rgb.shape[1]
    row["depth_ms"] = round(depth_ms, 1)
    try:
        cs = replace(cs, layer_method=method)
        t0 = time.perf_counter()
        stub = build_stub(cs, rgb)
        apply_layers(stub, depth, path.name)
        apply_ms = (time.perf_counter() - t0) * 1000.0
        row["load_to_ready_ms"] = round(img_ms + depth_ms + apply_ms, 1)
        planes = layer_planes(stub.layer_records())
        row["K"] = len(planes)
        row["n_props_hidden"] = len(place_props(
            planes, N_PROPS, float(cs.game_prop_radius_mm),
            tuple(float(v) for v in cs.overscan_design_eye_mm)))
        frac, layers = seam_hard_frac(depth, rgb, cs, method)
        row["seam_hard_frac"] = "" if frac is None else round(frac, 6)
        preview_png(layers, out_dir / f"{path.stem}_{method}.png")
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="PB-059 (2) anh that")
    ap.add_argument("--photos", default=str(PHOTOS))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--scene-types", default=None,
                    help="JSON {ten_file: nhan}; thieu thi de trong")
    args = ap.parse_args(argv)
    folder, out_dir = Path(args.photos), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = (json.loads(Path(args.scene_types).read_text(encoding="utf-8"))
              if args.scene_types else {})
    photos = [p for p in sorted(folder.iterdir())
              if p.is_file() and p.suffix.lower() in EXT]
    if not photos:
        print(f"Khong co anh nao trong {folder}")
        return 1
    cs = ContentSettings.from_dict(json.loads(
        (ROOT / "assets/config/content.json").read_text(encoding="utf-8")))
    estimator = DepthEstimator(str(cs.depth_model), str(cs.depth_backend))
    warm_up(photos[0], estimator)
    rows = []
    for n, path in enumerate(photos, 1):
        try:
            prepared = prepare(path, estimator)
        except Exception as exc:        # ảnh hỏng: một hàng lỗi cho mỗi method
            rows.extend(error_rows(path, labels.get(path.name, ""), exc))
            print(f"[{n}/{len(photos)}] {path.name} LOI: {exc}", flush=True)
            continue
        for method in LAYER_METHODS:
            row = run_one(path, cs, labels.get(path.name, ""), out_dir, method,
                          prepared)
            rows.append(row)
            print(f"[{n}/{len(photos)}] {path.name} {method:11s} "
                  f"{row['width']}x{row['height']} K={row['K']} "
                  f"props={row['n_props_hidden']} "
                  f"seam={row['seam_hard_frac']} depth={row['depth_ms']}ms "
                  f"ready={row['load_to_ready_ms']}ms {row['error']}",
                  flush=True)
    csv_path = out_dir / "real_images.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} hang -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
