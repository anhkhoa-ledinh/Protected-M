r"""
PB-074 Phase 0b — dựng mỗi ảnh ở ba vị trí mắt qua đúng đường GL của app.

Mỗi ảnh chạy **hai lượt dựng** trong hai tiến trình con riêng (một ngữ cảnh GL
một lần cho chắc): lượt ``plain`` = cấu hình demo thật, ảnh lưu ra PNG; lượt
``mark`` = tô void/backdrop ĐỎ, đệm LỤC, lấp khe LAM, lớp gần nhất TRẮNG, và
một lượt ``dark`` tô cả bốn bằng ĐEN để lấy hiệu từng kênh (xem
``src/content/debug_marks.py``).

    .\.venv\Scripts\python.exe scripts\render_eyes.py
    .\.venv\Scripts\python.exe scripts\render_eyes.py --tag after --preset diecut
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _render_eyes_one import EYES, IMAGES, OUT_ROOT, PRESETS  # noqa: E402

FIELDS = ["image", "eye_x_mm", "preset", "width", "height", "K", "build_s",
          "edge_void_pct", "void_pct", "pad_pct", "fill_visible_pct",
          "band_fill_pct", "plate_fill_pct",
          "band_tex_ratio", "plate_tex_ratio",
          "fill_tex_ratio", "halo_dL", "rim_dL", "error"]


def run_one(name, preset, tag, timeout_s=900, split=False):
    """Chạy MỘT ảnh trong tiến trình con; trả danh sách hàng CSV."""
    out_json = OUT_ROOT / tag / f"{name}.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    cmd = [str(ROOT / ".venv/Scripts/python.exe"),
           str(ROOT / "scripts" / "_render_eyes_one.py"),
           "--image", name, "--preset", preset, "--tag", tag]
    if split:
        cmd.append("--split-fill")
    print(f"  -> {name} [{preset}]", flush=True)
    proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True,
                          timeout=timeout_s)
    if proc.returncode != 0 or not out_json.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
        print(f"     LOI rc={proc.returncode}: {' | '.join(tail)}", flush=True)
        return [{**{k: "" for k in FIELDS}, "image": name, "preset": preset,
                 "eye_x_mm": eye, "error": f"rc={proc.returncode}"}
                for eye in EYES]
    rows = json.loads(out_json.read_text(encoding="utf-8"))
    for row in rows:
        print(f"     eye {row['eye_x_mm']:+5.0f}: void {row['edge_void_pct']}% "
              f"fill {row['fill_visible_pct']}% tex {row['fill_tex_ratio']} "
              f"halo {row['halo_dL']} rim {row['rim_dL']}", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="PB-074 render o ba vi tri mat")
    ap.add_argument("--tag", default="before")
    ap.add_argument("--preset", default="demo", choices=sorted(PRESETS))
    ap.add_argument("--images", nargs="*", default=None)
    ap.add_argument("--split-fill", action="store_true")
    args = ap.parse_args(argv)
    names = args.images if args.images else list(IMAGES)
    rows = []
    for name in names:
        rows.extend(run_one(name, args.preset, args.tag,
                            split=args.split_fill))
    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir.parent / f"{args.tag}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\n{len(rows)} hang -> {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
