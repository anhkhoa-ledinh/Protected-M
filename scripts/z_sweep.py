r"""
TASK_game_1110 L4 — quét biên độ thị sai trên cả bộ ảnh review.

Mỗi ảnh một tiến trình con (``_z_sweep_one.py``): một phiên GL, bốn giá trị
``layer_z_far_m`` dùng lại đúng một lượt nướng. Ra ``results/layers_1110/z/``:
sheet 4×3 mỗi ảnh, JSON tỉ lệ đặt chỗ, và ``ratios.csv`` gộp.

    .\.venv\Scripts\python.exe scripts\z_sweep.py --set eight
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layer_review import review_set  # noqa: E402

OUT_DIR = ROOT / "results" / "layers_1110" / "z"
FIELDS = ["image", "layer_z_far_m", "placed", "requested", "ratio",
          "outside_viewport", "raw_candidates", "bake_s", "error"]


def run_one(name, path, timeout_s):
    cmd = [sys.executable, str(ROOT / "scripts" / "_z_sweep_one.py"),
          "--image", str(path), "--name", name, "--out", str(OUT_DIR)]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=timeout_s, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        print(f"     HET GIO sau {timeout_s}s", flush=True)
        return [{"image": name, "error": "timeout"}]
    row_file = OUT_DIR / f"{name}.json"
    if proc.returncode != 0 or not row_file.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
        print(f"     LOI rc={proc.returncode}: {' | '.join(tail)}", flush=True)
        return [{"image": name, "error": f"rc={proc.returncode}"}]
    rows = json.loads(row_file.read_text(encoding="utf-8"))
    for row in rows:
        row["error"] = ""
        print(f"     z={row['layer_z_far_m']:+.2f}: dat {row['placed']}/"
              f"{row['requested']} (ti le {row['ratio']}), ngoai o "
              f"{row['outside_viewport']}", flush=True)
    print(f"     ({time.perf_counter() - t0:.0f}s)", flush=True)
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description="L4 quet z")
    ap.add_argument("--set", default="eight", choices=("all", "user", "t", "eight"))
    ap.add_argument("--only", default="")
    ap.add_argument("--reuse", action="store_true")
    ap.add_argument("--timeout", type=float, default=1800.0)
    args = ap.parse_args(argv)

    images = review_set(args.set)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        images = {k: v for k, v in images.items() if k in keep}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"L4: {len(images)} anh -> {OUT_DIR}", flush=True)

    rows = []
    for name, path in sorted(images.items()):
        row_file = OUT_DIR / f"{name}.json"
        if args.reuse and row_file.exists():
            print(f"  -> {name} (da co san)", flush=True)
            cached = json.loads(row_file.read_text(encoding="utf-8"))
            for row in cached:
                row["error"] = ""
            rows.extend(cached)
            continue
        print(f"  -> {name}", flush=True)
        rows.extend(run_one(name, path, args.timeout))

    with (OUT_DIR / "ratios.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    (OUT_DIR / "ratios.json").write_text(json.dumps(rows, indent=1),
                                         encoding="utf-8")
    ok = [r for r in rows if not r.get("error")]
    print(f"\nXong: {len(ok)}/{len(rows)} hang tot -> {OUT_DIR / 'ratios.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
