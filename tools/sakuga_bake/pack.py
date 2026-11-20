"""Pack raw baked frames into one trimmed atlas per variant + fx.json.

python3 pack.py out/raw DEST     -> DEST/sheets/*.png, DEST/words/*.png, DEST/fx.json
Frame rects are in image px (1920 basis). Offsets are from the effect's anchor (the point the game passes in).
"""
import json, os, shutil, sys
import numpy as np
from PIL import Image


def bleed(im):
    """Fill RGB under transparent pixels within 4 px of opaque ones with the nearby colour.

    The rest of the transparent area gets one flat colour, which keeps the PNG small.

    Alpha is untouched, so the image looks identical when blended. It only matters for
    mipmaps: GL averages straight-alpha RGB, and black under alpha 0 would darken edges.
    """
    import cv2
    a = np.asarray(im, dtype=np.float32)
    rgb, al = a[..., :3], a[..., 3:4] / 255.0
    empty = al[..., 0] == 0
    if not empty.any() or empty.all():
        return im
    out = rgb.copy(); todo = empty.copy()
    num, den = rgb * al, al[..., 0]
    for r in (1, 2, 4):          # mip levels 0-1 only ever read 1-2 px past an edge
        k = (2 * r + 1, 2 * r + 1)
        nb, db = cv2.blur(num, k), cv2.blur(den, k)
        hit = todo & (db > 1e-4)
        out[hit] = nb[hit] / db[hit][:, None]
        todo &= ~hit
        if not todo.any():
            break
    if todo.any():
        out[todo] = (num.sum((0, 1)) / max(den.sum(), 1e-6))
    res = np.concatenate([np.clip(out, 0, 255), a[..., 3:4]], axis=2).round().astype(np.uint8)
    return Image.fromarray(res, 'RGBA')


RAW, DEST = sys.argv[1], sys.argv[2]
meta = json.load(open(os.path.join(RAW, 'meta.json')))
S = meta['scale']; AXp, AYp = meta['anchor'][0] * S, meta['anchor'][1] * S
os.makedirs(os.path.join(DEST, 'sheets'), exist_ok=True)
os.makedirs(os.path.join(DEST, 'words'), exist_ok=True)
MAXW = int(os.environ.get('FX_MAXW', 2048))
out = {'version': 1, 'source': 'sakuga-vfx core.js (MIT, Justin Finkelman), recoloured to the ui_v3 palette',
       'image_scale': S, 'fps': meta['fps'], 'rgb_bled': True, 'effects': {}, 'words': {}}
total_px = 0
for name, r in meta['recipes'].items():
    eff = {'duration': r['duration'], 'words': r.get('words', []), 'variants': []}
    for v in r['variants']:
        d = os.path.join(RAW, v['dir'])
        frames = []
        for f in range(v['frames']):
            im = Image.open(os.path.join(d, f'{f:03d}.png')).convert('RGBA')
            bb = im.getchannel('A').getbbox()
            frames.append((im.crop(bb), bb) if bb else (None, None))
        while frames and frames[-1][0] is None:      # drop the empty tail
            frames.pop()
        # shelf packing, tallest first
        order = sorted([i for i, (im, _) in enumerate(frames) if im], key=lambda i: -frames[i][0].height)
        x = y = shelf_h = 0; W = 0; pos = {}
        for i in order:
            im = frames[i][0]
            if x + im.width > MAXW:
                x, y, shelf_h = 0, y + shelf_h + 2, 0
            pos[i] = (x, y); x += im.width + 2; shelf_h = max(shelf_h, im.height); W = max(W, x)
        H = y + shelf_h
        atlas = Image.new('RGBA', (max(1, W), max(1, H)), (0, 0, 0, 0))
        recs = []
        for i, (im, bb) in enumerate(frames):
            if im is None:
                recs.append(None); continue
            atlas.paste(im, pos[i])
            recs.append({'rect': [pos[i][0], pos[i][1], im.width, im.height],
                         'offset': [round(bb[0] - AXp), round(bb[1] - AYp)]})
        fn = f"{v['dir']}.png"
        atlas = bleed(atlas)
        atlas.save(os.path.join(DEST, 'sheets', fn), optimize=True)
        total_px += atlas.width * atlas.height
        eff['variants'].append({'sheet': f'sheets/{fn}', 'size': [atlas.width, atlas.height], 'frames': recs})
    out['effects'][name] = eff
for fn in sorted(os.listdir(os.path.join(RAW, 'words'))):
    bleed(Image.open(os.path.join(RAW, 'words', fn)).convert('RGBA')).save(os.path.join(DEST, 'words', fn), optimize=True)
for w, m in meta['words'].items():
    out['words'][w] = {'traced_w': m['traced_w'], 'traced_h': m['traced_h'], 'pad': m['pad'],
                       'styles': ['ink', 'hollow', 'dotted', 'crit']}
json.dump(out, open(os.path.join(DEST, 'fx.json'), 'w'), indent=1)
print('atlas px', total_px, '-> RGBA MB', round(total_px * 4 / 2**20, 1))
