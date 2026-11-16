r"""
PB-075 — nạp sẵn cache lấp khe cho các ảnh demo, để lúc trình diễn không phải
ngồi chờ LaMa.

    .\.venv\Scripts\python.exe scripts\precompute_fill.py anh1.png anh2.jpg
    .\.venv\Scripts\python.exe scripts\precompute_fill.py --demo-config

Chạy **đúng** đường tách lớp của app (cùng `content.demo.json`, cùng
`DepthEstimator`, cùng `cluster_into_layers`) nhưng **không cần GL**. Mọi lượt
lấp đi qua `src/content/lama_fill.py`, nên kết quả rơi vào đúng cache theo nội
dung mà app sẽ đọc lại (`artifacts/lama_cache/`, đọc lại mất 7 ms).

Cache băm theo **nội dung ảnh + mặt nạ lỗ**, không theo tên file hay theo
khoá. Nên nếu sau này đổi một tham số làm đổi ảnh hoặc đổi mặt nạ (ví dụ
`layer_z_far_m`, `edge_style`), cache cũ **không trúng nữa** — không sai gì
cả, chỉ là phải tính lại. Chạy lại script này sau mỗi lần đổi cấu hình demo.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config.schema import ContentSettings  # noqa: E402
from src.content.depth_estimator import DepthEstimator  # noqa: E402
from src.content.disocclusion import DisocclusionParams  # noqa: E402
from src.content.lama_fill import CACHE_DIR  # noqa: E402
from src.content.layer_clustering import (LayerEdgeParams,  # noqa: E402
                                          LayerStagingParams,
                                          cluster_into_layers)

#: PB-050: app thu ảnh về cạnh dài này lúc nạp; phải làm y hệt thì mới cùng cache.
MAX_LONG_SIDE = 2048


def load_demo_content() -> ContentSettings:
    """``ContentSettings`` đúng như `content.demo.json` đang commit."""
    data = json.loads((ROOT / "assets/config/content.demo.json")
                      .read_text(encoding="utf-8"))
    return ContentSettings.from_dict(data)


def load_rgb(path: Path):
    """Ảnh RGB đã thu về cạnh dài ``MAX_LONG_SIDE`` như app làm."""
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise RuntimeError(f"khong doc duoc anh: {path}")
    height, width = bgr.shape[:2]
    scale = min(1.0, MAX_LONG_SIDE / float(max(height, width)))
    if scale < 1.0:
        bgr = cv2.resize(bgr, (int(round(width * scale)),
                               int(round(height * scale))),
                         interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def precompute(path: Path, content: ContentSettings, depth_estimator) -> dict:
    """Tách lớp một ảnh; trả vài số để in ra."""
    rgb = load_rgb(path)
    t0 = time.perf_counter()
    depth = depth_estimator.estimate(rgb)
    depth_s = time.perf_counter() - t0
    params = DisocclusionParams(
        enabled=bool(content.disocclusion_fill_enabled),
        fill_px=int(content.disocclusion_fill_px),
        inpaint_radius=int(content.disocclusion_inpaint_radius),
        feather_px=int(content.disocclusion_feather_px),
        max_work_dim=int(content.disocclusion_max_work_dim),
        fill_method=str(content.fill_method),
        plate_fill=str(content.plate_fill))
    t0 = time.perf_counter()
    result = cluster_into_layers(
        rgb, depth, n_clusters=int(content.layer_count), seed=0,
        staging=LayerStagingParams(), edge=LayerEdgeParams(),
        disocclusion=params)
    return {"anh": path.name, "lop": len(result.layers),
            "depth_s": round(depth_s, 1),
            "tach_lop_s": round(time.perf_counter() - t0, 1)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="PB-075 nap san cache lap khe")
    ap.add_argument("images", nargs="*", type=Path)
    ap.add_argument("--demo-config", action="store_true",
                    help="chi in cau hinh dang dung roi thoat")
    args = ap.parse_args(argv)
    content = load_demo_content()
    print(f"cau hinh demo: fill_method={content.fill_method} · "
          f"plate_fill={content.plate_fill} · cache={CACHE_DIR}", flush=True)
    if args.demo_config:
        return 0
    if not args.images:
        ap.error("can it nhat mot duong dan anh")
    if str(content.plate_fill).lower() == "telea" \
            and str(content.fill_method).lower() == "telea":
        print("CA HAI khoa dang la 'telea' — Telea khong dung cache, "
              "chay script nay khong co tac dung gi.", flush=True)
        return 0
    before = len(list(CACHE_DIR.glob("*.png"))) if CACHE_DIR.exists() else 0
    estimator = DepthEstimator(model_name=str(content.depth_model))
    for path in args.images:
        print(precompute(path, content, estimator), flush=True)
    after = len(list(CACHE_DIR.glob("*.png"))) if CACHE_DIR.exists() else 0
    print(f"cache: {before} -> {after} muc", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
