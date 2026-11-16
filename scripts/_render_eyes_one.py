r"""
PB-074 Phase 0b — một ảnh, ba vị trí mắt, hai lượt dựng. Chạy một mình.

Gọi qua ``render_eyes.py`` (nó mở một tiến trình con cho mỗi ảnh). Ở đây mới
là chỗ mở ngữ cảnh GL, nên một tiến trình chỉ có đúng một ngữ cảnh.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _look3d_metrics import (MARK_A, fill_tex_ratio,  # noqa: E402
                             halo_dl, rim_dl, share_pct, source_masks)
from src.config.loader import merge_user_config  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402

OUT_ROOT = Path(r"D:\test_true3D space\task_done\PB-074_look3d")
PHOTOS = Path(r"D:\test_true3D space\real_data\photos")
TEST_IMAGES = Path(r"D:\PycharmProjects\test_images")
EYES = (-100.0, 0.0, 100.0)
#: y và z của mắt = mắt thiết kế của demo (``overscan_design_*``).
EYE_Y_MM, EYE_Z_MM = 100.0, 450.0
MARK_TOP = (255, 255, 255)
#: PB-075b: bộ đánh dấu TÁCH vùng lấp. Dùng lại đúng ba kênh của bộ cũ, chỉ
#: đổi nghĩa kênh lục: đệm (≤ 0,0014 % ở mọi ảnh, PB-074 Phase 0b) nhường chỗ
#: cho tấm nền, nên không cần thêm lượt dựng nào.
MARK_SPLIT = {"void": (255, 0, 0), "pad": (0, 0, 0), "fill": (0, 0, 255),
              "plate": (0, 255, 0)}

IMAGES = {
    "t25": TEST_IMAGES / "t25.png",
    "t22": TEST_IMAGES / "t22.png",
    "t24": TEST_IMAGES / "t24.png",
    "t4": TEST_IMAGES / "t4.png",
    "IMG_20251128_155315": PHOTOS / "IMG_20251128_155315.jpg",
    "Messenger_creation": PHOTOS /
    "Messenger_creation_9B215370-A1BA-4362-B10A-3C85A885764F.jpeg",
}
#: Bộ khoá nội dung của từng preset; ``demo`` = y hệt ``content.demo.json``.
PRESETS = {"demo": {},
           "diecut": {"edge_style": "diecut"},
           "fsr": {"fill_method": "fsr"},
           "lama": {"fill_method": "lama"},
           # PB-075: chi doi cach lap TAM NEN.
           "shiftmap": {"plate_fill": "shiftmap"},
           "plate_lama": {"plate_fill": "lama"},
           "shadow": {"layer_shadow": True},
           "diecut_shadow": {"edge_style": "diecut", "layer_shadow": True},
           "box": {"surround": "box", "overscan_e_max_mm": 0.0},
           # Phase 6: bốn preset người dùng chấm bằng mắt. P0 = demo hôm nay;
           # P1 không có "fill tốt nhất" vì FSR trượt cả hai tiêu chí nên cách
           # lấp vẫn là telea.
           "p1": {"edge_style": "diecut"},
           "p2": {"edge_style": "diecut", "layer_shadow": True},
           "p3": {"edge_style": "diecut", "layer_shadow": True,
                  "surround": "box", "overscan_e_max_mm": 0.0},
           # Phase 6: biên độ thị sai — chỉ đổi độ sâu lớp xa.
           "zfar060": {"layer_z_far_m": -0.60},
           "zfar085": {"layer_z_far_m": -0.85}}


def content_for(image_path, preset, marks):
    """``ContentSettings`` của demo, trỏ vào ảnh, cộng khoá của preset."""
    data = json.loads((ROOT / "assets/config/content.demo.json")
                      .read_text(encoding="utf-8"))
    data["content_path"] = str(image_path)
    data.update(PRESETS[preset])
    content = ContentSettings.from_dict(data)
    if marks is not None:
        mark = tuple(c / 255.0 for c in marks["void"])
        content = replace(content, void_clear_color=mark)
        # Ở chế độ hộp, vách hộp lấy màu ``backdrop_color`` và vách **là nội
        # dung cố ý**, không phải chỗ hở. Tô nó thì đếm cả vách thành "void"
        # (đo thử trên t25: 12–23 %). Chỉ tô tấm backdrop ở chế độ void, nơi
        # nhìn thấy nó đúng là đã nhìn lọt qua mép ảnh.
        if str(content.surround).lower().strip() != "box":
            content = replace(content, backdrop_color=mark)
    return content


def set_marks(marks):
    """Đặt các biến đánh dấu; ``None`` = xoá hết."""
    names = {"pad": "HCPD_MARK_PAD", "fill": "HCPD_MARK_FILL",
             "top": "HCPD_MARK_TOP", "plate": "HCPD_MARK_PLATE"}
    for key, env in names.items():
        if marks is None or key not in marks:
            os.environ.pop(env, None)
        else:
            os.environ[env] = ",".join(str(int(v)) for v in marks[key])


def render_pass(config, content, screen_cal, marks):
    """Mở một phiên GL, chờ lớp, vẽ ba vị trí mắt; trả (ảnh theo mắt, K, giây)."""
    from _render_eyes_gl import OffscreenApp

    set_marks(marks)
    app = OffscreenApp(config, content, screen_cal)
    try:
        t0 = time.perf_counter()
        waited = app.wait_ready((0.0, EYE_Y_MM, EYE_Z_MM))
        if waited is None:
            raise RuntimeError("het gio cho tach lop")
        build_s = time.perf_counter() - t0
        frames = {}
        for eye_x in EYES:
            app.draw((eye_x, EYE_Y_MM, EYE_Z_MM))      # một khung để ổn định
            frames[eye_x] = app.draw((eye_x, EYE_Y_MM, EYE_Z_MM))
        return frames, app.layer_count(), build_s
    finally:
        app.close()
        set_marks(None)


def rows_for(name, preset, plain, marked, dark, k, build_s, split=False):
    """Một hàng mỗi vị trí mắt."""
    out = []
    for eye_x in EYES:
        masks = source_masks(marked[eye_x], dark[eye_x])
        near = masks["void"] & masks["pad"] & masks["fill"]   # trắng = lớp gần
        void = masks["void"] & ~near
        pad = masks["pad"] & ~near
        fill = masks["fill"] & ~near
        # PB-075b: ở bộ MARK_SPLIT, kênh lục là TẤM NỀN chứ không phải đệm.
        plate = pad if split else None
        frame = plain[eye_x]
        # Ở chế độ tách, kênh lục KHÔNG còn là đệm, nên hai cột liên quan tới
        # đệm phải để trống chứ không được báo nhầm số của tấm nền.
        all_fill = (fill | plate) if split else fill
        out.append({
            "image": name, "eye_x_mm": eye_x, "preset": preset,
            "width": frame.shape[1], "height": frame.shape[0], "K": k,
            "build_s": round(build_s, 2),
            "edge_void_pct": share_pct(void) if split
            else share_pct(void | pad),
            "void_pct": share_pct(void),
            "pad_pct": "" if split else share_pct(pad),
            "fill_visible_pct": share_pct(all_fill),
            "band_fill_pct": share_pct(fill) if split else "",
            "plate_fill_pct": share_pct(plate) if split else "",
            "fill_tex_ratio": fill_tex_ratio(frame, all_fill),
            # PB-075b: vân RIÊNG của từng vùng. Diện tích và phương sai là hai
            # chuyện khác nhau — dải nằm dọc đường bao vật (chỗ tương phản
            # cao) nên có thể lấn át chỉ số dù diện tích nhỏ hơn tấm nền.
            "band_tex_ratio": fill_tex_ratio(frame, fill) if split else "",
            "plate_tex_ratio": fill_tex_ratio(frame, plate) if split else "",
            "halo_dL": halo_dl(frame, near),
            "rim_dL": rim_dl(frame, near), "error": ""})
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="PB-074 mot anh, ba vi tri mat")
    ap.add_argument("--image", required=True, choices=sorted(IMAGES))
    ap.add_argument("--preset", default="demo", choices=sorted(PRESETS))
    ap.add_argument("--tag", default="before")
    ap.add_argument("--split-fill", action="store_true",
                    help="PB-075b: tach vung lap thanh tam nen / dai lop giua")
    args = ap.parse_args(argv)
    from src.engine.app import HcpdApplication

    config = merge_user_config(ROOT / "assets/config/default.json")
    # Dùng CHÍNH hàm hiệu chuẩn màn của app (nó chỉ cần ``self._config``), để
    # không có bản sao công thức thứ hai.
    holder = type("ConfigHolder", (), {"_config": config})()
    screen_cal = HcpdApplication._load_screen_calibration(holder, ROOT)
    path = IMAGES[args.image]
    marks = ({**MARK_SPLIT, "top": MARK_TOP} if args.split_fill
             else {**MARK_A, "top": MARK_TOP})
    dark_marks = {key: (0, 0, 0) for key in marks}
    plain, k, build_s = render_pass(
        config, content_for(path, args.preset, None), screen_cal, None)
    marked, _, _ = render_pass(
        config, content_for(path, args.preset, marks), screen_cal, marks)
    dark, _, _ = render_pass(
        config, content_for(path, args.preset, dark_marks), screen_cal,
        dark_marks)
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    for eye_x, frame in plain.items():
        cv2.imwrite(str(out_dir / f"{args.image}_{args.preset}_x{int(eye_x):+d}.png"),
                    cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    rows = rows_for(args.image, args.preset, plain, marked, dark, k, build_s,
                    split=args.split_fill)
    (out_dir / f"{args.image}.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
