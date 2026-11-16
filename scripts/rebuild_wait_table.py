r"""
SPIKE-12 — dựng lại ``gate_wait_table.csv`` với ``p`` ĐÃ GỒM depth khung khoá.

Bảng cũ (30/10 sáng) dùng ``p`` chỉ có decode + flow + tầng lớp = 15,3 ms →
65 fps, tức bỏ qua chi phí thật của depth. Bảng này chia đều depth khung khoá
cho ``k`` khung nội dung: ``p(k) = p_base + t_depth_khoá / k``. Chỉ đọc
``gate_summary.csv``, không đo lại.

    .\.venv\Scripts\python.exe scripts\rebuild_wait_table.py
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _video_gate_parts import amortised_p, wait_table  # noqa: E402

OUT_DIR = Path(r"D:\test_true3D space\task_done\SPIKE-12_video_gate")
KS = (1, 5, 10, 20)             # k = 10 là mức khuyến nghị của SPIKE-12


def rows_for(summary_rows, ks=KS):
    """Mọi hàng bảng chờ: mỗi (clip, threads, k) một bộ f × L."""
    out = []
    for stat in summary_rows:
        p_base = float(stat["p_s"])
        key_ms = float(stat["t_keyframe_depth_ms"])
        for k in ks:
            p_k = amortised_p(p_base, key_ms, k)
            for row in wait_table(p_k):
                row.update({"clip": stat["clip"], "threads": int(stat["threads"]),
                            "k_keyframe": k, "p_base_s": round(p_base, 4),
                            "keyframe_depth_ms": key_ms,
                            "content_fps_max": round(1.0 / p_k, 1)})
                out.append(row)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description="SPIKE-12 wait table (p co depth)")
    ap.add_argument("--dir", default=str(OUT_DIR))
    args = ap.parse_args(argv)
    folder = Path(args.dir)
    with open(folder / "gate_summary.csv", encoding="utf-8") as fh:
        summary = list(csv.DictReader(fh))
    rows = rows_for(summary)
    keys = sorted({k for r in rows for k in r})
    with open(folder / "gate_wait_table.csv", "w", newline="",
              encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {folder / 'gate_wait_table.csv'} ({len(rows)} hang)")
    for row in rows:
        if row["threads"] == 2 and row["k_keyframe"] == 10:
            print(f"  k=10 threads=2 f={row['f_fps']} L={row['clip_s']}: "
                  f"p={row['p_s']}s ({row['content_fps_max']} fps) "
                  f"p*f={row['p_times_f']} W={row['W_s']}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
