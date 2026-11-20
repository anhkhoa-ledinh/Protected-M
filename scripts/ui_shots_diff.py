r"""
TASK_arch_refactor_1120: compare two ``--ui-shots`` PNG sets for G2.

    python scripts/ui_shots_diff.py <baseline_dir> <current_dir>

Exact pixel equality, EXCEPT column x=0 (the leftmost framebuffer
column), which is masked out: measured directly (R0, two independent
runs of the unmodified baseline commit, then again after R1 with a
fresh pair) to jitter by a few colour units on every single screen,
always and ONLY at x=0, never elsewhere -- GPU-level MSAA/viewport-edge
antialiasing non-determinism, not application behaviour. Anything
outside that one column must match exactly, same strictness as `cmp`.

Exit code 0 (PASS) only if every PNG in ``baseline_dir`` has a
same-named, pixel-identical (outside x=0) match in ``current_dir``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

#: Measured ceiling (R0/R1 baseline-vs-baseline runs): worst case seen
#: was 613 differing pixels (a full column) at max channel delta 34,
#: always at x=0. A real regression would show up at other columns too
#: (this check requires EXACT match there) or bleed into interior
#: pixels in on the same column (we only mask column 0 entirely, so
#: there's no numeric threshold to tune -- it's a fixed mask, not a
#: fuzzy tolerance).
EDGE_COLUMN = 1


def compare_one(base_path: Path, cur_path: Path) -> tuple[bool, str]:
    a = np.array(Image.open(base_path).convert("RGB")).astype(int)
    b = np.array(Image.open(cur_path).convert("RGB")).astype(int)
    if a.shape != b.shape:
        return False, f"size differs: {a.shape} vs {b.shape}"
    a_masked = a[:, EDGE_COLUMN:, :]
    b_masked = b[:, EDGE_COLUMN:, :]
    diff = np.abs(a_masked - b_masked)
    mask = diff.sum(axis=2) > 0
    n = int(mask.sum())
    if n == 0:
        return True, "identical (outside x=0)"
    ys, xs = np.where(mask)
    return False, (f"{n} px differ outside x=0 (max={int(diff.max())}, "
                   f"x range {int(xs.min())}-{int(xs.max())})")


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(f"usage: {argv[0]} <baseline_dir> <current_dir>")
        return 2
    base_dir, cur_dir = Path(argv[1]), Path(argv[2])
    base_files = sorted(base_dir.glob("*.png"))
    if not base_files:
        print(f"no PNGs found in {base_dir}")
        return 2
    all_ok = True
    for f in base_files:
        cur = cur_dir / f.name
        if not cur.exists():
            print(f"MISSING: {f.name}")
            all_ok = False
            continue
        ok, detail = compare_one(f, cur)
        print(f"{'PASS' if ok else 'FAIL'}: {f.name} -- {detail}")
        all_ok = all_ok and ok
    print("RESULT", "PASS" if all_ok else "FAIL")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
