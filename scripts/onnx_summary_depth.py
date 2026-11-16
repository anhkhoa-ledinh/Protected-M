r"""
PB-060: gop ``rows_<backend>.csv`` + ``meta_<backend>.csv`` thanh bang tong.

    .\.venv\Scripts\python.exe scripts\onnx_summary_depth.py

Ra ``artifacts/pb060_onnx/onnx_compare.csv`` (backend x anh) va ``summary.csv``
(mot dong moi backend), roi in bang kem hai phep kiem dat truoc.
"""
from __future__ import annotations

import csv
import statistics as st
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import onnx_depth_common as C  # noqa: E402

OUT = ROOT / "artifacts" / "pb060_onnx"
SUM_FIELDS = ["backend", "model_mb", "peak_rss_mb", "t_pre_ms_median",
              "t_infer_ms_median", "max_abs_diff", "mean_abs_diff",
              "max_abs_diff_export_shape", "n_export_shape",
              "absrel_mean", "absrel_ci_lo", "absrel_ci_hi", "delta1_mean",
              "n_app", "n_nyu", "threads"]
# Head cua HF nuong cung buoc upsample cuoi thanh 518x686 (xem onnx_export_depth),
# nen voi anh ra dung shape nay ONNX moi trung torch; cot rieng de thay ro.
EXPORT_SHAPE = "1x3x518x686"


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def summarize(backend, rows, meta):
    """Mot dong tong: trung vi do tre 8 anh app, AbsRel + CI bootstrap 100 anh NYU."""
    app = [r for r in rows if r["kind"] == "app"]
    nyu = [r for r in rows if r["kind"] == "nyu"]
    absrels = [float(r["absrel"]) for r in nyu]
    delta1s = [float(r["delta1"]) for r in nyu]
    lo, hi = C.bootstrap_ci(absrels, n_boot=1000, seed=0)
    same = [r for r in app if r["tensor_shape"] == EXPORT_SHAPE]
    return {
        "backend": backend,
        "model_mb": meta["model_mb"],
        "peak_rss_mb": meta["peak_rss_mb"],
        "t_pre_ms_median": round(st.median(float(r["t_pre_ms"]) for r in app), 1),
        "t_infer_ms_median": round(st.median(float(r["t_infer_ms"]) for r in app), 1),
        "max_abs_diff": f"{max(float(r['max_abs_diff']) for r in app):.8f}",
        "mean_abs_diff": f"{st.mean(float(r['mean_abs_diff']) for r in app):.8f}",
        "max_abs_diff_export_shape":
            f"{max(float(r['max_abs_diff']) for r in same):.8f}" if same else "",
        "n_export_shape": len(same),
        "absrel_mean": f"{st.mean(absrels):.5f}",
        "absrel_ci_lo": f"{lo:.5f}", "absrel_ci_hi": f"{hi:.5f}",
        "delta1_mean": f"{st.mean(delta1s):.5f}",
        "n_app": len(app), "n_nyu": len(nyu), "threads": C.THREADS,
    }


def preset_checks(by_backend):
    """Hai phep kiem dat truoc Phase 1 cho ``ort_fp32`` so voi ``torch_fp32``."""
    ort = by_backend["ort_fp32"]
    ref = by_backend["torch_fp32"]
    max_diff = float(ort["max_abs_diff"])
    d_absrel = abs(float(ort["absrel_mean"]) - float(ref["absrel_mean"]))
    print("\n-- Phep kiem dat truoc (ort_fp32 vs torch_fp32)")
    print(f"  max_abs_diff = {max_diff:.8f}  (nguong <= 1e-3)  "
          f"{'PASS' if max_diff <= 1e-3 else 'FAIL'}")
    print(f"  |delta AbsRel| = {d_absrel:.5f}  (nguong <= 0.001)  "
          f"{'PASS' if d_absrel <= 0.001 else 'FAIL'}")
    only = float(ort["max_abs_diff_export_shape"])
    print(f"  chan doan: rieng {ort['n_export_shape']}/{ort['n_app']} anh ra dung "
          f"{EXPORT_SHAPE} thi max_abs_diff = {only:.8f}")
    print("  ort_int8: chi bao so, khong dat nguong.")


def main():
    all_rows, summary, by_backend = [], [], {}
    for backend in C.BACKENDS:
        rows_path, meta_path = OUT / f"rows_{backend}.csv", OUT / f"meta_{backend}.csv"
        if not rows_path.exists():
            print(f"Thieu {rows_path.name} — chay onnx_compare_depth.py {backend} truoc")
            return 2
        rows = read_csv(rows_path)
        all_rows.extend(rows)
        line = summarize(backend, rows, read_csv(meta_path)[0])
        summary.append(line)
        by_backend[backend] = line
    with open(OUT / "onnx_compare.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)
    with open(OUT / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUM_FIELDS)
        writer.writeheader()
        writer.writerows(summary)
    print(",".join(SUM_FIELDS))
    for line in summary:
        print(",".join(str(line[k]) for k in SUM_FIELDS))
    preset_checks(by_backend)
    return 0


if __name__ == "__main__":
    sys.exit(main())
