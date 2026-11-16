r"""
SPIKE-12 — cổng cho PB-065: ĐO THÔI, không đổi ``src/``.

Chạy đường "producer" của thiết kế video 2.5D trên clip thật, ở cỡ làm việc
640 px cạnh dài, rồi dựng bảng thời gian chờ W. Phần đo tranh chấp CPU (bước 3
của TASK_video25d Phase 0) cần người ngồi trước webcam nên KHÔNG nằm ở đây.

    .\.venv\Scripts\python.exe scripts\video_gate.py --threads 2

Bước 3 (tranh chấp CPU) dùng hai cờ thêm, vẫn chỉ đo:

    ... scripts\video_gate.py --load 60          # tai nen producer PB-065
    ... scripts\video_gate.py --contend-report <frame_log.csv> --label alone
"""
from __future__ import annotations

import argparse
import csv
import statistics as st
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from _video_gate_parts import (DEPTH_SHORTEST, dis, flip_rate,  # noqa: E402
                               labels_from_centroids, light_layering,
                               load_depth, wait_table, warp_by_flow,
                               work_resize)
from src.content.layer_clustering import cluster_depth_layers  # noqa: E402

VIDEOS = Path(r"D:\test_true3D space\real_data\videos")
OUT_DIR = Path(r"D:\test_true3D space\task_done\SPIKE-12_video_gate")
CONTENT_FPS = 5.0
MAX_SAMPLES = 100


def sample_indices(n_frames, fps, content_fps, cap):
    """Chỉ số khung nội dung ở ``content_fps``, cắt còn tối đa ``cap`` mẫu."""
    step = max(1, int(round(float(fps) / float(content_fps))))
    idx = list(range(0, int(n_frames), step))
    if len(idx) > cap:                      # thưa đều ra, vẫn trải cả clip
        keep = np.linspace(0, len(idx) - 1, cap).round().astype(int)
        idx = [idx[i] for i in keep]
    return idx


def measure(path, threads, content_fps=CONTENT_FPS):
    """Đo từng khung nội dung; trả (dict thống kê, danh sách hàng thô)."""
    depth_of = load_depth(threads)
    cap = cv2.VideoCapture(str(path))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    idx = sample_indices(n_frames, fps, content_fps, MAX_SAMPLES)
    flows = {name: dis(name) for name in ("ULTRAFAST", "FAST")}
    rows, key_gray, key_depth, centroids = [], None, None, None
    prev_a = prev_b = None
    t_key_ms = []
    for n, i in enumerate(idx):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            continue
        t0 = time.perf_counter()
        rgb = work_resize(bgr)
        t_dec = (time.perf_counter() - t0) * 1000.0
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        row = {"frame": i, "t_decode_ms": round(t_dec, 2)}
        if key_gray is None:                # khung khoá đầu: depth + tâm cụm
            t0 = time.perf_counter()
            key_depth = depth_of(rgb)
            t_key_ms.append((time.perf_counter() - t0) * 1000.0)
            _, centroids = cluster_depth_layers(key_depth, 3, seed=0)
            key_gray = gray
        for name, algo in flows.items():
            t0 = time.perf_counter()
            flow = algo.calc(key_gray, gray, None)
            warped = warp_by_flow(key_depth, flow)
            row[f"t_flow_{name.lower()}_ms"] = round(
                (time.perf_counter() - t0) * 1000.0, 2)
            if name == "ULTRAFAST":
                lab_b = labels_from_centroids(warped, centroids)
                _, t_lay = light_layering(rgb, lab_b, len(centroids))
                row["t_layer_ms"] = round(t_lay, 2)
                row["flip_b_pct"] = round(flip_rate(prev_b, lab_b), 3)
                prev_b = lab_b
        t0 = time.perf_counter()
        d_now = depth_of(rgb)               # (a) depth + k-means moi khung
        lab_a, _ = cluster_depth_layers(d_now, 3, seed=0)
        row["t_depth_per_frame_ms"] = round(
            (time.perf_counter() - t0) * 1000.0, 2)
        row["flip_a_pct"] = round(flip_rate(prev_a, lab_a), 3)
        prev_a = lab_a
        rows.append(row)
        print(f"  [{n + 1}/{len(idx)}] frame {i}: "
              f"flow_u={row['t_flow_ultrafast_ms']} "
              f"flow_f={row['t_flow_fast_ms']} lay={row['t_layer_ms']} "
              f"flipA={row['flip_a_pct']} flipB={row['flip_b_pct']}", flush=True)
    cap.release()

    def med(key):
        vals = [r[key] for r in rows if isinstance(r.get(key), float)
                and r[key] == r[key]]
        return (round(st.median(vals), 2) if vals else float("nan"),
                round(float(np.percentile(vals, 90)), 2) if vals else float("nan"))

    stats = {"clip": path.name, "threads": threads, "n_sampled": len(rows),
             "n_frames": n_frames, "fps": round(fps, 3),
             "depth_shortest": DEPTH_SHORTEST,
             "t_keyframe_depth_ms": round(st.median(t_key_ms), 1) if t_key_ms
             else float("nan")}
    for key in ("t_decode_ms", "t_flow_ultrafast_ms", "t_flow_fast_ms",
                "t_layer_ms", "t_depth_per_frame_ms", "flip_a_pct",
                "flip_b_pct"):
        stats[f"{key}_med"], stats[f"{key}_p90"] = med(key)
    # p = giây/khung nội dung của đường (b): decode + flow ULTRAFAST + layering.
    stats["p_s"] = round((stats["t_decode_ms_med"]
                          + stats["t_flow_ultrafast_ms_med"]
                          + stats["t_layer_ms_med"]) / 1000.0, 4)
    return stats, rows


def step3_mode(args, clips):
    """Hai chế độ của bước 3 (tranh chấp CPU): tải nền và đọc nhật ký khung."""
    if args.contend_report:                 # chỉ đọc CSV, không mở clip
        from _video_gate_contend import contention_report
        for key, value in contention_report(args.contend_report,
                                            args.label).items():
            print(f"{key}={value}")
        return 0
    from _video_gate_contend import run_load
    run_load(clips[0], args.load, threads=max(args.threads))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description="SPIKE-12 video gate (measure)")
    ap.add_argument("--videos", default=str(VIDEOS))
    ap.add_argument("--out", default=str(OUT_DIR))
    ap.add_argument("--threads", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--load", type=float, default=None, metavar="SECONDS",
                    help="chay tai nen producer PB-065 (buoc 3, tranh chap)")
    ap.add_argument("--contend-report", default=None, metavar="FRAME_CSV",
                    help="doc nhat ky khung cua app -> FPS dem theo cua so 5 s")
    ap.add_argument("--label", default="run")
    args = ap.parse_args(argv)
    folder, out_dir = Path(args.videos), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    clips = [p for p in sorted(folder.iterdir())
             if p.suffix.lower() in {".mp4", ".mov", ".avi", ".mkv"}]
    if not clips:
        print(f"Khong co clip nao trong {folder}")
        return 1
    if args.contend_report or args.load is not None:
        return step3_mode(args, clips)
    all_stats, all_rows, waits = [], [], []
    for clip in clips:
        for threads in args.threads:
            print(f"{clip.name}  threads={threads}")
            stats, rows = measure(clip, threads)
            all_stats.append(stats)
            for row in rows:
                row.update({"clip": clip.name, "threads": threads})
            all_rows.extend(rows)
            for w in wait_table(stats["p_s"]):
                w.update({"clip": clip.name, "threads": threads})
                waits.append(w)
    for name, data in (("gate_summary.csv", all_stats),
                       ("gate_frames.csv", all_rows),
                       ("gate_wait_table.csv", waits)):
        keys = sorted({k for d in data for k in d})
        with open(out_dir / name, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=keys)
            writer.writeheader()
            writer.writerows(data)
        print(f"-> {out_dir / name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
