r"""
SPIKE-12 bước 3 — tải nền của PB-065 và cách đọc nhật ký khung. ĐO THÔI.

``run_load``: đường "producer" của video 2.5D (decode → DIS ULTRAFAST → kéo
depth khung khoá → tầng lớp nhẹ, depth lại mỗi ``k`` khung) chạy **theo nhịp
phát f fps**, không hết sức. ``contention_report``: đọc CSV nhật ký khung của
app (PB-058) ra **FPS đếm theo cửa sổ 5 giây**, KHÔNG phải FPS EMA của F1.
"""
from __future__ import annotations

import csv
import statistics as st
import time

import cv2

from _video_gate_parts import (dis, labels_from_centroids, light_layering,
                               load_depth, warp_by_flow, work_resize)
from src.content.layer_clustering import cluster_depth_layers

LOAD_FPS = 5.0
LOAD_KEYFRAME_EVERY = 10        # k = 10, mức khuyến nghị của SPIKE-12
LOAD_K = 3

# ``track_ms`` là giá trị LẤY MẪU mỗi khung render; luồng tracking quay nhanh
# hơn webcam nên phần lớn vòng rỗng (~0,2 ms). Phân bố hai cục: nhật ký PB-058
# ``hybrid_t22_vsync1_realeye`` có 2403 mẫu < 1 ms, 95 mẫu 1–5 ms, **0 mẫu
# 5–10 ms**, 1341 mẫu ≥ 20 ms. Ngưỡng 5 ms nằm giữa khoảng trống; chốt TRƯỚC
# khi có bất kỳ số tranh chấp nào.
TRACK_WORK_MS = 5.0


def _produce(frame_rgb, state, depth_of, flow, is_key):
    """Một khung nội dung của producer; trả ms đã tốn."""
    t0 = time.perf_counter()
    gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
    if is_key or state.get("key_gray") is None:
        depth = depth_of(frame_rgb)
        _, state["centroids"] = cluster_depth_layers(depth, LOAD_K, seed=0)
        state["key_gray"], state["key_depth"] = gray, depth
        warped = depth
    else:
        f = flow.calc(state["key_gray"], gray, None)
        warped = warp_by_flow(state["key_depth"], f)
    labels = labels_from_centroids(warped, state["centroids"])
    light_layering(frame_rgb, labels, len(state["centroids"]))
    return (time.perf_counter() - t0) * 1000.0


def run_load(clip, seconds, content_fps=LOAD_FPS,
             keyframe_every=LOAD_KEYFRAME_EVERY, threads=2):
    """Tải nền PB-065 ``seconds`` giây; không theo kịp nhịp thì
    ``fps_achieved`` tụt xuống — số đó phải báo kèm kết quả tranh chấp."""
    depth_of = load_depth(threads)          # nạp model TRƯỚC khi tính giờ
    cap = cv2.VideoCapture(str(clip))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    step = max(1, int(round(fps / float(content_fps))))
    flow, state = dis("ULTRAFAST"), {}
    period, busy, produced, keys, idx = 1.0 / float(content_fps), 0.0, 0, 0, 0
    t_start = next_due = time.perf_counter()
    print(f"[load] bat dau: {seconds:.0f}s, f={content_fps}, k={keyframe_every}",
          flush=True)
    while time.perf_counter() - t_start < float(seconds):
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx % max(1, n_frames))
        ok, bgr = cap.read()
        if not ok or bgr is None:
            idx = 0
            continue
        is_key = (produced % int(keyframe_every)) == 0
        busy += _produce(work_resize(bgr), state, depth_of, flow, is_key) / 1000.0
        produced, keys = produced + 1, keys + int(is_key)
        idx += step
        next_due += period
        delay = next_due - time.perf_counter()
        # tụt nhịp (delay ≤ 0): bám lại mốc hiện tại, không dồn nợ
        next_due = next_due if delay > 0 else time.perf_counter()
        time.sleep(max(0.0, delay))
    cap.release()
    elapsed = time.perf_counter() - t_start
    out = {"load_s": round(elapsed, 2), "frames": produced, "keyframes": keys,
           "fps_target": content_fps, "keyframe_every": keyframe_every,
           "fps_achieved": round(produced / elapsed, 2),
           "duty_pct": round(100.0 * busy / elapsed, 1),
           "ms_per_frame": round(1000.0 * busy / max(1, produced), 1)}
    print(f"[load] xong: {out}", flush=True)
    return out


def read_frame_log(path, measure_s=60.0, tail_s=1.0):
    """Các dòng khung trong cửa sổ đo = ``measure_s`` giây cuối, trừ đuôi."""
    rows = []
    with open(path, encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            try:
                rows.append({k: float(v) for k, v in row.items()})
            except (TypeError, ValueError):
                continue
    if not rows:
        return []
    end = max(r["t_s"] for r in rows) - float(tail_s)
    start = end - float(measure_s)
    return [r for r in rows if start <= r["t_s"] <= end]


def window_fps(rows, window_s=5.0):
    """FPS **đếm được** từng cửa sổ ``window_s`` giây: số khung / bề rộng."""
    if not rows:
        return []
    t0 = min(r["t_s"] for r in rows)
    span = max(r["t_s"] for r in rows) - t0
    n_win = int(span // float(window_s))
    out = []
    for i in range(max(1, n_win)):
        lo = t0 + i * float(window_s)
        count = sum(1 for r in rows if lo <= r["t_s"] < lo + float(window_s))
        out.append(round(count / float(window_s), 2))
    return out


def _med_p90(values):
    """``(trung vị, p90)`` kiểu nearest-rank; ``(-1, -1)`` khi rỗng."""
    if not values:
        return -1.0, -1.0
    ordered = sorted(values)
    return (round(st.median(ordered), 2),
            round(ordered[int(0.9 * (len(ordered) - 1))], 2))


def contention_report(path, label, window_s=5.0, measure_s=60.0):
    """Một hàng kết quả tranh chấp từ nhật ký khung của app."""
    rows = read_frame_log(path, measure_s)
    fps_win = window_fps(rows, window_s)
    track = [r["track_ms"] for r in rows if "track_ms" in r]
    work = [v for v in track if v >= TRACK_WORK_MS]
    track_med, track_p90 = _med_p90(track)
    work_med, work_p90 = _med_p90(work)
    frame_med, _ = _med_p90([r["frame_ms"] for r in rows if "frame_ms" in r])
    return {"run": label, "frames": len(rows), "windows": len(fps_win),
            "window_s": window_s,
            "fps_windows": " ".join(f"{v:g}" for v in fps_win),
            "fps_window_med": round(st.median(fps_win), 2) if fps_win else -1.0,
            "fps_window_min": min(fps_win) if fps_win else -1.0,
            "track_ms_med": track_med, "track_ms_p90": track_p90,
            "track_work_n": len(work), "track_work_ms_med": work_med,
            "track_work_ms_p90": work_p90, "track_ms_max": max(track) if track
            else -1.0, "frame_ms_med": frame_med}
