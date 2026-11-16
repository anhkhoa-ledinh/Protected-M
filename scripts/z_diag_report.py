#!/usr/bin/env python3
"""
PHASE A reporting helper: summarize artifacts/z_diag.csv.

Read-only. Prints exactly what the Phase A acceptance report asks for:
min / max / std per numeric column, the reference_depth_raw anchor, the count
of frames with presence < 0.25, and the first 20 rows.

Usage:
    .\\.venv\\Scripts\\python.exe scripts\\z_diag_report.py
    .\\.venv\\Scripts\\python.exe scripts\\z_diag_report.py --csv path\\to\\z_diag.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]

STAT_COLUMNS = (
    "tvec_z_raw",
    "depth_raw",
    "z_after_scale",
    "z_after_iris",
    "z_after_clamp",
    "z_published",
)


def _to_float_array(rows: list[dict], column: str) -> np.ndarray:
    values: list[float] = []
    for row in rows:
        raw = (row.get(column) or "").strip()
        if not raw:
            continue
        try:
            values.append(float(raw))
        except ValueError:
            continue
    return np.asarray(values, dtype=np.float64)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--csv",
        type=Path,
        default=PROJECT_ROOT / "artifacts" / "z_diag.csv",
        help="Path to z_diag.csv (default: artifacts/z_diag.csv)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=20,
        help="How many leading rows to print (default: 20)",
    )
    args = parser.parse_args(argv)

    if not args.csv.exists():
        print(f"ERROR: {args.csv} not found.")
        print('Enable "debug": {"z_diagnostics": true} in assets/config/default.json,')
        print("run main.py, lean in and out for ~20 s, then quit and re-run this.")
        return 1

    with args.csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if not rows:
        print(f"ERROR: {args.csv} has a header but no data rows.")
        return 1

    print(f"CSV       : {args.csv}")
    print(f"Total rows: {len(rows)}")
    print()

    print(f"{'column':<22}{'count':>7}{'min':>12}{'max':>12}{'std':>12}{'range':>12}")
    print("-" * 77)
    for column in STAT_COLUMNS:
        values = _to_float_array(rows, column)
        if values.size == 0:
            print(f"{column:<22}{0:>7}{'-':>12}{'-':>12}{'-':>12}{'-':>12}")
            continue
        print(
            f"{column:<22}{values.size:>7}"
            f"{values.min():>12.2f}{values.max():>12.2f}"
            f"{values.std():>12.4f}{values.max() - values.min():>12.2f}"
        )
    print()

    anchors = sorted(set(_to_float_array(rows, "reference_depth_raw").tolist()))
    if not anchors:
        print("reference_depth_raw : (never populated)")
    elif len(anchors) == 1:
        print(f"reference_depth_raw : {anchors[0]:.4f}  (constant, as expected)")
    else:
        print(f"reference_depth_raw : NOT CONSTANT — {len(anchors)} distinct values")
        print(f"                      {anchors[:10]}")

    presence = _to_float_array(rows, "presence")
    low = int((presence < 0.25).sum()) if presence.size else 0
    print(f"presence            : {presence.size} samples, {low} below 0.25")

    rmse = _to_float_array(rows, "reproj_rmse_px")
    if rmse.size:
        print(
            f"reproj_rmse_px      : min={rmse.min():.2f} "
            f"median={np.median(rmse):.2f} max={rmse.max():.2f}"
        )
    print()

    head = max(0, int(args.head))
    print(f"--- first {head} rows ---")
    fieldnames = list(rows[0].keys())
    print(",".join(fieldnames))
    for row in rows[:head]:
        print(",".join(str(row.get(name, "")) for name in fieldnames))

    return 0


if __name__ == "__main__":
    sys.exit(main())
