"""Compare app shot-mode screenshots with the pack's expected/ images.
python compare.py <shots_dir> <pack_dir>        (shots named <screen>.png, any 16:9 size)
Both are resized to 640x360 greyscale. A screen's score is the WORST mean absolute difference over all 10x10 cells (0-255).
PASS when every screen scores <= 24."""
import sys, os, glob
from PIL import Image, ImageChops, ImageStat
shots, pack = sys.argv[1], sys.argv[2]
rows = []
for exp in sorted(glob.glob(os.path.join(pack, 'expected', 'v3_*.jpg'))):
    name = os.path.basename(exp)[:-4]
    shot = os.path.join(shots, name + '.png')
    if not os.path.exists(shot):
        rows.append((name, None))
        continue
    a = Image.open(exp).convert('L').resize((640, 360), Image.LANCZOS)
    b = Image.open(shot).convert('L').resize((640, 360), Image.LANCZOS)
    diff = ImageChops.difference(a, b)
    worst = max(ImageStat.Stat(diff.crop((x, y, x + 10, y + 10))).mean[0] for y in range(0, 360, 10) for x in range(0, 640, 10))
    rows.append((name, worst))
for n, w in rows:
    print(f'{n:28s} ' + ('MISSING' if w is None else f'{w:6.1f}  ' + ('PASS' if w <= 24 else 'FAIL')))
ok = all(w is not None and w <= 24 for _, w in rows)
print('RESULT', 'PASS' if ok else 'FAIL', '(worst %.1f)' % max([w for _, w in rows if w is not None] or [0]))
sys.exit(0 if ok else 1)
