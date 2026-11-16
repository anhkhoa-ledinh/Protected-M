"""
Chấm ảnh chụp giao diện v3 với MỘT bộ ảnh tham chiếu bất kỳ.

Dùng đúng phép chấm của ``assets/ui_v3/tools/compare.py`` (thu về 640×360 xám,
lấy ô 10×10 tệ nhất), chỉ khác ở chỗ bộ tham chiếu là tham số. Cần nó vì trên
máy này ``tools/preview.py`` — bản tham chiếu mà ``expected/`` sinh ra từ đó —
cũng KHÔNG khớp ``expected/`` (ô tệ nhất 47,9): Pillow/FreeType ở đây rasterise
chữ khác máy đã xuất bộ asset. Muốn biết phần port có trung thành hay không thì
phải chấm với preview.py chạy **tại chỗ**, chứ chấm với ``expected/`` là đo lẫn
cả sai lệch của môi trường.

    python scripts/ui_v3_score.py <thu_muc_anh_chup> <thu_muc_tham_chieu>
"""

from __future__ import annotations

import glob
import os
import sys

from PIL import Image, ImageChops, ImageStat

PASS_AT = 24.0


def score(reference: str, shot: str) -> float:
    """Ô 10×10 tệ nhất giữa hai ảnh, sau khi thu về 640×360 xám."""
    first = Image.open(reference).convert("L").resize((640, 360), Image.LANCZOS)
    second = Image.open(shot).convert("L").resize((640, 360), Image.LANCZOS)
    diff = ImageChops.difference(first, second)
    return max(ImageStat.Stat(diff.crop((x, y, x + 10, y + 10))).mean[0]
               for y in range(0, 360, 10) for x in range(0, 640, 10))


def main(argv) -> int:
    shots, pack = argv[1], argv[2]
    pattern = os.path.join(pack, "v3_*.*")
    rows = []
    for reference in sorted(glob.glob(pattern)):
        name = os.path.splitext(os.path.basename(reference))[0]
        shot = os.path.join(shots, name + ".png")
        rows.append((name, score(reference, shot) if os.path.exists(shot) else None))
    for name, worst in rows:
        state = "MISSING" if worst is None else (
            f"{worst:6.1f}  " + ("PASS" if worst <= PASS_AT else "FAIL"))
        print(f"{name:28s} {state}")
    values = [w for _n, w in rows if w is not None]
    ok = len(values) == len(rows) and all(w <= PASS_AT for w in values)
    print("RESULT", "PASS" if ok else "FAIL", "(worst %.1f)" % max(values or [0]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
