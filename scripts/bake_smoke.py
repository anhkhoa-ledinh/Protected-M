r"""
TASK_game_1110 L1 cổng nghiệm thu — 10 lượt nướng liên tiếp, không chết, không treo.

Đi qua **đúng** ``BakeWorker`` mà app dùng (tiến trình ``spawn``, không GL),
nên cái được chấm là chính đường chạy thật. Mỗi ảnh nướng một lần rồi nạp lại
ngay lần thứ hai: lần hai phải là **cache trúng** và không mở tiến trình nào.

    .\.venv\Scripts\python.exe scripts\bake_smoke.py
    .\.venv\Scripts\python.exe scripts\bake_smoke.py --base assets/config/content.game.json
"""
from __future__ import annotations

import argparse
import json
import logging
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.layer_review import review_set  # noqa: E402
from src.config.schema import ContentSettings  # noqa: E402
from src.content.bake import BakeWorker  # noqa: E402
from src.content.media_gate import import_photo  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description="L1: 10 luot nuong lien tiep")
    ap.add_argument("--base", default="assets/config/content.demo.json")
    ap.add_argument("--count", type=int, default=10)
    ap.add_argument("--out", default="results/layers_1110/bake_smoke.json")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    data = json.loads((ROOT / args.base).read_text(encoding="utf-8"))
    images = sorted(review_set("all").items())[:args.count]
    if len(images) < args.count:
        raise SystemExit(f"chi tim duoc {len(images)} anh, can {args.count}")

    worker = BakeWorker(ROOT)
    rows = []
    for name, path in images:
        content = ContentSettings.from_dict({**data, "content_path": str(path)})
        row = {"image": name, "bake_s": None, "cache_s": None, "k": None,
               "cache_hit": False, "error": ""}
        t0 = time.perf_counter()
        worker.submit(name, import_photo(path), content)
        while worker.busy:
            worker.poll()
            time.sleep(0.2)
        result = worker.pop_ready(name)
        if result is None:
            row["error"] = "khong co ket qua (xem log)"
            rows.append(row)
            print(f"  {name}: HONG", flush=True)
            if worker.failed:
                raise SystemExit("BakeWorker bao hong — dung lai")
            continue
        row["bake_s"] = round(time.perf_counter() - t0, 2)
        row["k"] = int(result.k)

        # Lần hai: phải trúng cache và không mở tiến trình nào.
        t1 = time.perf_counter()
        worker.submit(name + "#2", import_photo(path), content)
        row["cache_hit"] = not worker.busy
        again = worker.pop_ready(name + "#2")
        row["cache_s"] = round(time.perf_counter() - t1, 3)
        row["error"] = "" if (again is not None and row["cache_hit"]) \
            else "lan hai khong trung cache"
        rows.append(row)
        print(f"  {name}: K={row['k']} nuong {row['bake_s']}s, "
              f"cache {row['cache_s']}s hit={row['cache_hit']}", flush=True)
    worker.stop()

    times = [r["bake_s"] for r in rows if r["bake_s"] is not None]
    bad = [r for r in rows if r["error"]]
    summary = {
        "n": len(rows), "ok": len(rows) - len(bad),
        "bake_mean_s": round(statistics.mean(times), 2) if times else None,
        "bake_max_s": round(max(times), 2) if times else None,
        "bake_min_s": round(min(times), 2) if times else None,
        "cache_max_s": round(max((r["cache_s"] or 0.0) for r in rows), 3),
        "rows": rows,
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    print(f"\n{summary['ok']}/{summary['n']} luot tot — "
          f"trung binh {summary['bake_mean_s']}s, lon nhat {summary['bake_max_s']}s, "
          f"cache lon nhat {summary['cache_max_s']}s")
    print(f"-> {out}")
    return 0 if not bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
