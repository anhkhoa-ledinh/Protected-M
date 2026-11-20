"""Contact sheet of a few baked effects on a sky background, drawn with fxref.py from fx.json.
   python3 tools/sakuga_bake/preview.py assets/fx out.jpg
"""
import sys, json
sys.path.insert(0, '.')
from fxref import WordAnim
from PIL import Image
D = sys.argv[1] if len(sys.argv) > 1 else 'assets/fx'; m = json.load(open(f'{D}/fx.json')); S = m['image_scale']; fps = m['fps']
def render(name, var, f, bg, ax, ay, mirror=False):
    e = m['effects'][name]; v = e['variants'][var]
    atlas = Image.open(f"{D}/{v['sheet']}")
    out = bg.copy()
    fr = v['frames'][f] if f < len(v['frames']) else None
    if fr:
        x, y, w, h = fr['rect']; im = atlas.crop((x, y, x + w, y + h)); ox, oy = fr['offset']
        if mirror: im = im.transpose(Image.FLIP_LEFT_RIGHT); ox = -ox - w
        out.alpha_composite(im, (int(ax * S + ox), int(ay * S + oy)))
    for wc in e['words']:
        a = WordAnim(m['words'][wc['word']]['traced_w'], ax + wc['dx'], ay + wc['dy'], wc['size'], wc['rot'], wc['life'])
        st = None
        for _ in range(f + 1): st = a.step(1 / fps)
        if not st: continue
        cx, cy, sc, ang = st
        im = Image.open(f"{D}/words/{wc['word']}_{wc['style']}.png")
        im = im.resize((max(1, int(im.width * sc * S)), max(1, int(im.height * sc * S))), Image.LANCZOS).rotate(-ang * 57.2958, expand=True, resample=Image.BICUBIC)
        out.alpha_composite(im, (int(cx * S - im.width / 2), int(cy * S - im.height / 2)))
    return out
sky = Image.new('RGBA', (int(600 * S), int(480 * S)), (215, 232, 248, 255))  # 900x720 at image_scale 1.5
cells = [('ui_confirm', 0, 4), ('ui_confirm', 0, 14), ('star_five', 0, 6), ('star_pop', 0, 8), ('toast_in', 0, 5), ('load_done', 0, 5), ('hit_block', 0, 3), ('hit_kill', 1, 5)]
tiles = [render(n, v, f, sky, 300, 260, mirror=(n == 'hit_kill')).resize((450, 360)) for n, v, f in cells]
sheet = Image.new('RGBA', (450 * 4, 360 * 2))
for i, t in enumerate(tiles): sheet.paste(t, ((i % 4) * 450, (i // 4) * 360))
sheet.convert('RGB').save(sys.argv[2] if len(sys.argv) > 2 else 'fx_preview.jpg', quality=85)
