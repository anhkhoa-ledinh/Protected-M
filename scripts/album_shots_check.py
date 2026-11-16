r"""
Tự kiểm bộ ảnh màn của giao diện album (TASK_album_fix_1106 F1.4).

    .\.venv\Scripts\python.exe scripts\album_shots_check.py

Hỏng thì **báo to và trả mã khác 0**: lượt chụp đầu có 18/50 ảnh đen mà báo
cáo vẫn ghi "50 PNG" vì không ai mở ra xem.
"""
from __future__ import annotations

import hashlib
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SHOTS = ROOT / "results" / "album_ui_1106"
#: Ảnh "đồng nhất" = thu về 64×36 mà còn ít hơn ngần này màu khác nhau.
MIN_COLOURS = 50


def check(folder: Path = SHOTS) -> int:
    """In kết quả tự kiểm; trả số lỗi tìm được."""
    files = sorted(folder.glob("*.png"))
    if not files:
        print(f"KHONG CO ANH NAO trong {folder}")
        return 1
    uniform, digests, sizes = [], defaultdict(list), defaultdict(int)
    for path in files:
        with Image.open(path) as image:
            sizes[image.size] += 1
            thumb = image.convert("RGB").resize((64, 36))
            colours = len(set(thumb.getdata()))
        if colours < MIN_COLOURS:
            uniform.append((path.name, colours))
        digests[hashlib.md5(path.read_bytes()).hexdigest()].append(path.name)
    duplicates = [names for names in digests.values() if len(names) > 1]
    print(f"anh: {len(files)} · kich thuoc: {dict(sizes)}")
    print(f"khong dong nhat: {len(files) - len(uniform)}/{len(files)}")
    print(f"md5 khac nhau: {len(digests)}/{len(files)}")
    for name, colours in uniform:
        print(f"  DONG NHAT: {name} ({colours} mau)")
    for names in duplicates:
        print(f"  TRUNG MD5: {names}")
    return len(uniform) + len(duplicates)


if __name__ == "__main__":
    raise SystemExit(1 if check() else 0)
