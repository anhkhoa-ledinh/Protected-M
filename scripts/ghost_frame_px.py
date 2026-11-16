r"""
TASK_ghost_ab_1111 Phase 1 — frame_px, qua ``_ghost_frame_gl.py`` (một tiến
trình GL mỗi ảnh × biến thể, cơ chế đánh dấu node y như ``void_pct``).
"""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from layer_review import PROFILES  # noqa: E402

OUT_DIR = ROOT / "results" / "layers_1110" / "ghost_ab"
IMAGES = {
    "t16": r"D:\PycharmProjects\test_images\t16.png",
}


def main() -> int:
    from layer_review import review_set

    images = review_set("eight")
    names = ("t16", "u1_anime_room", "u2_cat")
    variants = ("ghost_v0", "ghost_v1", "ghost_v2", "ghost_v3")
    rows = []
    for name in names:
        path = images[name]
        for variant in variants:
            _base, keys = PROFILES[variant]
            cmd = [sys.executable, str(ROOT / "scripts" / "_ghost_frame_gl.py"),
                  "--image", str(path), "--name", name, "--variant", variant,
                  "--keys", json.dumps(keys)]
            proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                                  text=True, timeout=300, encoding="utf-8",
                                  errors="replace")
            if proc.returncode != 0:
                tail = (proc.stdout + proc.stderr).strip().splitlines()[-6:]
                print(f"LOI {name} {variant}: {' | '.join(tail)}")
                rows.append({"image": name, "variant": variant, "error": "rc"})
                continue
            row = json.loads(proc.stdout.strip().splitlines()[-1])
            rows.append(row)
            print(f"{name:16} {variant:10} frame_px max={row['frame_px_max']}",
                  flush=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fields = ["image", "variant", "frame_px_x-120", "frame_px_x+0",
             "frame_px_x+120", "frame_px_win0", "frame_px_max", "error"]
    with (OUT_DIR / "frame_px.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})
    print(f"-> {OUT_DIR / 'frame_px.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
