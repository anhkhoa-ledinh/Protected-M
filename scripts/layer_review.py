r"""
TASK_game_1110 L0 — bộ chấm tách lớp: dựng bằng **đúng renderer thật**, ngoài
màn, rồi đếm.

Mỗi (ảnh × profile) chạy trong MỘT tiến trình con (``_layer_review_one.py``):
một ngữ cảnh GL một lần, và một lượt tách lớp chết không kéo theo cả lượt
chấm. Ra:

- ``sheet_<ảnh>_<profile>.png`` — ảnh gốc + 4 lượt dựng cạnh nhau;
- ``rows.csv`` — một hàng mỗi (ảnh × profile);
- ``rows.json`` — y hệt, để script khác đọc.

    .\.venv\Scripts\python.exe scripts\layer_review.py --tag baseline
    .\.venv\Scripts\python.exe scripts\layer_review.py --tag game --profile game
    .\.venv\Scripts\python.exe scripts\layer_review.py --tag zsweep ^
        --profile z060,z035,z022,z014 --set user
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

OUT_ROOT = ROOT / "results" / "layers_1110"
TEST_IMAGES = Path(r"D:\PycharmProjects\test_images")


def _resolve_by_nfc(parent: Path, nfc_name: str) -> Path:
    """
    Tìm con của ``parent`` có tên NFC-tương đương ``nfc_name``.

    OneDrive ghi tên thư mục/file tiếng Việt xuống đĩa ở dạng **NFD** (dấu là
    ký tự tổ hợp riêng), còn chuỗi gõ trong mã nguồn này là **NFC** (dấu dính
    liền chữ cái) — hai chuỗi trông giống hệt nhau nhưng khác byte, nên so
    sánh thẳng hay ``Path.exists()`` trên chuỗi NFC luôn trả ``False``. Dò
    bằng cách liệt kê thư mục rồi so khớp sau khi đã chuẩn hoá cả hai về NFC
    thì không phụ thuộc dạng nào đang nằm trên đĩa.

    Không thấy thì trả lại ``parent / nfc_name`` y nguyên — gọi sau đó sẽ báo
    thiếu file, rõ ràng hơn một ngoại lệ ở đây.
    """
    import unicodedata

    if not parent.is_dir():
        return parent / nfc_name
    target = unicodedata.normalize("NFC", nfc_name)
    for entry in parent.iterdir():
        if unicodedata.normalize("NFC", entry.name) == target:
            return entry
    return parent / nfc_name


#: Ảnh của người dùng, tìm được ngày 10/11 trong thư mục ảnh OneDrive. Chúng là
#: đúng ba tấm mà báo lỗi 07/11 nói tới: u1 = phòng anime (ám xanh lục),
#: u2 = con mèo (khối mèo xám nhoè), u3 = "vạn trận phá" (nhân vật nhân đôi).
#: Dò qua ``_resolve_by_nfc``: tên có dấu tiếng Việt, OneDrive ghi NFD.
USER_DIR = _resolve_by_nfc(Path.home() / "OneDrive", "Hình ảnh")
USER_IMAGES = {
    "u1_anime_room": USER_DIR / "1673415206261.jpg",
    "u2_cat": USER_DIR / "3a8b328b4d2dd1720d44ad1b20bd43c2.jpg",
    "u3_van_tran_pha": _resolve_by_nfc(USER_DIR, "vạn trận phá.PNG"),
}
T_IMAGES = {f"t{i}": TEST_IMAGES / f"t{i}.png" for i in range(1, 32)}

#: L0 (10/11): t11, t12, t17 hoá ra là BẢN SAO của chính u2, u1, u3 (cùng kích
#: thước pixel, cùng mọi số đo tới 3 chữ số thập phân). Bộ "eight" chính thức
#: của báo cáo L vì vậy KHÔNG dùng ba tên đó — tránh đếm trùng một ảnh hai lần.
EIGHT_SET = ("u1_anime_room", "u2_cat", "u3_van_tran_pha",
            "t1", "t15", "t16", "t18", "t20")

#: Bộ khoá của từng profile, chồng lên file nền (``--base``).
PROFILES = {
    "baseline": ("assets/config/content.demo.json", {}),
    "game": ("assets/config/content.game.json", {}),
    # L4: quét biên độ thị sai trên profile game.
    "z060": ("assets/config/content.game.json", {"layer_z_far_m": -0.60}),
    "z035": ("assets/config/content.game.json", {"layer_z_far_m": -0.35}),
    "z022": ("assets/config/content.game.json", {"layer_z_far_m": -0.22}),
    "z014": ("assets/config/content.game.json", {"layer_z_far_m": -0.14}),
    # L5: kiểm void ở ±150 mm thay vì ±120 mm.
    "game_e150": ("assets/config/content.game.json", {}),
    # TASK_ghost_ab_1111 Phase 1: bốn biến thể đăng ký lớp (chỉ khoá cấu hình,
    # không đụng src/). V0 = y hệt đã commit.
    "ghost_v0": ("assets/config/content.game.json",
                {"layer_z_far_m": -0.22}),
    "ghost_v1": ("assets/config/content.game.json",
                {"layer_z_far_m": -0.22, "overscan_mode": "pad"}),
    "ghost_v2": ("assets/config/content.game.json",
                {"layer_z_far_m": -0.22, "overscan_design_eye_mm": [0.0, 0.0]}),
    "ghost_v3": ("assets/config/content.game.json",
                {"layer_z_far_m": -0.22, "overscan_mode": "pad",
                 "overscan_design_eye_mm": [0.0, 0.0]}),
}

FIELDS = ["image", "profile", "K", "bake_s", "hole_pct", "deltaE_mean",
          "edge_step_pct", "void_pct", "void_full_pct", "void_pct_x-120",
          "void_pct_x+0", "void_pct_x+120", "void_pct_win0",
          "void_pct_win-side", "void_pct_win+side",
          "void_full_x-120", "void_full_x+0", "void_full_x+120",
          "void_full_win0", "void_full_win-side", "void_full_win+side",
          "hole_per_layer", "error"]


def review_set(name: str) -> dict:
    """Bộ ảnh cần chấm; chỉ giữ file thật sự có trên đĩa."""
    if name == "user":
        chosen = dict(USER_IMAGES)
    elif name == "t":
        chosen = dict(T_IMAGES)
    elif name == "eight":
        # Bộ 8 tấm CHÍNH THỨC của báo cáo L: 3 ảnh người dùng + 5 tấm t khác.
        all_images = {**USER_IMAGES, **T_IMAGES}
        chosen = {k: all_images[k] for k in EIGHT_SET}
    else:
        chosen = {**USER_IMAGES, **T_IMAGES}
    return {k: v for k, v in chosen.items() if v.exists()}


def run_one(name, path, profile, out_dir, timeout_s):
    """Một tiến trình con; trả hàng CSV (hoặc hàng lỗi)."""
    base, keys = PROFILES[profile]
    cmd = [sys.executable, str(ROOT / "scripts" / "_layer_review_one.py"),
           "--image", str(path), "--name", name, "--profile", profile,
           "--base", base, "--keys", json.dumps(keys), "--out", str(out_dir)]
    if profile == "game_e150":
        cmd += ["--eye-max", "150", "--win-eye-max", "150"]
    t0 = time.perf_counter()
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=timeout_s, encoding="utf-8",
                              errors="replace")
    except subprocess.TimeoutExpired:
        print(f"     HET GIO sau {timeout_s}s", flush=True)
        return {"image": name, "profile": profile, "error": "timeout"}
    row_file = out_dir / f"{name}_{profile}.json"
    if proc.returncode != 0 or not row_file.exists():
        tail = (proc.stdout + proc.stderr).strip().splitlines()[-4:]
        print(f"     LOI rc={proc.returncode}: {' | '.join(tail)}", flush=True)
        return {"image": name, "profile": profile,
                "error": f"rc={proc.returncode}"}
    row = json.loads(row_file.read_text(encoding="utf-8"))
    row["error"] = ""
    print(f"     K={row['K']} bake={row['bake_s']}s hole={row['hole_pct']}% "
          f"void={row['void_pct']}% dE={row['deltaE_mean']} "
          f"edge={row['edge_step_pct']}% ({time.perf_counter() - t0:.0f}s)",
          flush=True)
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description="L0 bo cham tach lop")
    ap.add_argument("--tag", default="baseline")
    ap.add_argument("--profile", default="baseline",
                    help="mot hoac nhieu profile, cach nhau bang dau phay")
    ap.add_argument("--set", default="all",
                    choices=("all", "user", "t", "eight"))
    ap.add_argument("--only", default="", help="loc ten anh, cach nhau bang phay")
    ap.add_argument("--timeout", type=float, default=1800.0)
    ap.add_argument("--reuse", action="store_true",
                    help="doc {ten}_{profile}.json da co thay vi dung lai; "
                         "chi chay anh con thieu, roi gop rows.csv tu DIA")
    args = ap.parse_args(argv)

    images = review_set(args.set)
    if args.only:
        keep = {s.strip() for s in args.only.split(",") if s.strip()}
        images = {k: v for k, v in images.items() if k in keep}
    profiles = [p.strip() for p in args.profile.split(",") if p.strip()]
    for p in profiles:
        if p not in PROFILES:
            raise SystemExit(f"khong biet profile: {p}")

    out_dir = OUT_ROOT / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"L0: {len(images)} anh x {len(profiles)} profile -> {out_dir}",
          flush=True)

    rows = []
    for name, path in sorted(images.items()):
        for profile in profiles:
            row_file = out_dir / f"{name}_{profile}.json"
            if args.reuse and row_file.exists():
                row = json.loads(row_file.read_text(encoding="utf-8"))
                row["error"] = ""
                print(f"  -> {name} [{profile}] (da co san)", flush=True)
                rows.append(row)
                continue
            print(f"  -> {name} [{profile}]", flush=True)
            rows.append(run_one(name, path, profile, out_dir, args.timeout))

    with (out_dir / "rows.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in FIELDS})
    (out_dir / "rows.json").write_text(json.dumps(rows, indent=1),
                                       encoding="utf-8")
    ok = [r for r in rows if not r.get("error")]
    print(f"\nXong: {len(ok)}/{len(rows)} hang tot -> {out_dir / 'rows.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
