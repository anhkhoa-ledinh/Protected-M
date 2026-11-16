"""Reference compositor for the ui_v3 pack = the SHOT-MODE rule. Every text shows its `sample`, every sprite with
`default: true` is drawn at its `rect`, pips use `sample_on`, bars use `sample_frac`, pills use their first style,
photo slots get sample/sample_room.png, the viewport is RGB(60,70,80), the camera RGB(90,90,100).
Order: base -> photos -> top -> bars -> pips -> pills -> sprites -> texts.
python3 preview.py <pack_dir> <out_dir> [photo.jpg]"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

PACK, OUTD = sys.argv[1], sys.argv[2]
PHOTO = sys.argv[3] if len(sys.argv) > 3 else None
M = json.load(open(os.path.join(PACK, 'manifest.json')))
K = M['image_scale']
_fonts = {}


def font(ff, fw, fs):
    files = M['fonts']['files'][ff]
    w = min(files, key=lambda k: abs(int(k) - int(fw)))
    key = (ff, w, fs)
    if key not in _fonts:
        _fonts[key] = ImageFont.truetype(os.path.join(PACK, files[w]), round(fs * K))
    return _fonts[key]


def rgba(h, a=1.0):
    h = h.lstrip('#')
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4)) + (round(255 * a),)


def run_width(r, txt):
    f = font(r['ff'], r['fw'], r['fs'])
    return f.getlength(txt) + r.get('ls', 0) * K * max(0, len(txt) - 1) if r.get('ls') else f.getlength(txt)


def draw_runs(d, rect, runs, al, values=None):
    x, y, w, h = [v * K for v in rect]
    texts = [(r, (r['c'].upper() if r.get('upper') else r['c'])) for r in runs]
    total = sum(run_width(r, t) for r, t in texts)
    cx = x if al == 'left' else (x + w - total if al == 'right' else x + (w - total) / 2)
    big = max(texts, key=lambda rt: rt[0]['fs'])[0]
    fb = font(big['ff'], big['fw'], big['fs'])
    asc, desc = fb.getmetrics()
    base_y = y + (h - (asc + desc)) / 2 + asc
    for r, t in texts:
        f = font(r['ff'], r['fw'], r['fs'])
        col = rgba(r['col'] or '#24415F')
        if r.get('ls'):
            for ch in t:
                d.text((cx, base_y), ch, font=f, fill=col, anchor='ls')
                cx += f.getlength(ch) + r['ls'] * K
        else:
            d.text((cx, base_y), t, font=f, fill=col, anchor='ls')
            cx += f.getlength(t)


def paste(canvas, path, rect, scale=None):
    im = Image.open(os.path.join(PACK, path)).convert('RGBA')
    x, y, w, h = rect
    tw, th = round(w * K), round(h * K)
    if im.size != (tw, th):
        im = im.resize((tw, th), Image.LANCZOS)
    layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    layer.paste(im, (round(x * K), round(y * K)))
    return Image.alpha_composite(canvas, layer)


def cover(canvas, src, rect, clip=None):
    x, y, w, h = [round(v * K) for v in rect]
    im = Image.open(src).convert('RGBA')
    s = max(w / im.width, h / im.height)
    im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
    ox, oy = (im.width - w) // 2, (im.height - h) // 2
    im = im.crop((ox, oy, ox + w, oy + h))
    layer = Image.new('RGBA', canvas.size, (0, 0, 0, 0))
    if clip:
        mask = Image.new('L', (w, h), 0)
        md = ImageDraw.Draw(mask)
        for c in clip:
            cx, cy, cw, ch = [round(v * K) for v in c]
            md.rectangle([cx - x, cy - y, cx - x + cw, cy - y + ch], fill=255)
        layer.paste(im, (x, y), mask)
    else:
        layer.paste(im, (x, y))
    return Image.alpha_composite(canvas, layer)


os.makedirs(OUTD, exist_ok=True)
for name, s in M['screens'].items():
    base = Image.open(os.path.join(PACK, s['base'])).convert('RGBA')
    c = base
    for ph in s['photos']:
        if ph['kind'] in ('photo',) and PHOTO:
            c = cover(c, PHOTO, ph['rect'], ph.get('clip'))
        elif ph['kind'] in ('viewport', 'camera'):
            d = ImageDraw.Draw(c)
            x, y, w, h = [v * K for v in ph['rect']]
            d.rectangle([x, y, x + w, y + h], fill=(60, 70, 80, 255) if ph['kind'] == 'viewport' else (90, 90, 100, 255))
    if s['top']:
        c = Image.alpha_composite(c, Image.open(os.path.join(PACK, s['top'])).convert('RGBA'))
    d = ImageDraw.Draw(c)
    for b in s['bars']:
        x, y, w, h = [v * K for v in b['rect']]
        r = b['radius'] * K
        fw = w * b['sample_frac']
        if b.get('rest_col'):
            d.rounded_rectangle([x, y, x + w, y + h], r, fill=rgba(b['rest_col']))
        d.rounded_rectangle([x, y, x + max(fw, 2 * r), y + h], r, fill=rgba(b['col']))
    for p in s['pips']:
        n = len(p['rects'])
        for i, rr in enumerate(p['rects']):
            st = p['on'] if p['sample_on'][i] else p['off']
            x, y, w, h = [v * K for v in rr]
            fill = rgba(st['fill'], st.get('fill_op', 1)) if st.get('fill') else None
            outline = rgba(st['stroke']) if st.get('stroke') else None
            sw = round(st.get('sw', 1) * K)
            if p['shape'] == 'ellipse':
                d.ellipse([x, y, x + w, y + h], fill=fill, outline=outline, width=sw)
            else:
                d.rounded_rectangle([x, y, x + w, y + h], p['radius'] * K, fill=fill, outline=outline, width=sw)
    for p in s['pills']:
        st = list(p['styles'].values())[0]
        f = p['font']
        tw = font(f['ff'], f['fw'], f['fs']).getlength(p['sample'] or '') / K + (f.get('ls') or 0) * len(p['sample'] or '')
        x, y, w, h = p['rect']
        w = tw + 2 * p['pad']
        d.rounded_rectangle([x * K, y * K, (x + w) * K, (y + h) * K], p['radius'] * K, fill=rgba(st['bg']))
        draw_runs(d, [x, y, w, h], [{'c': p['sample'], 'ff': f['ff'], 'fw': f['fw'], 'fs': f['fs'], 'col': st['fg'], 'ls': f.get('ls', 0)}], 'center')
    for sp in s['sprites']:
        if sp.get('default'):
            c = paste(c, sp['file'], sp['rect'])
    d = ImageDraw.Draw(c)
    for t in s['texts']:
        draw_runs(d, t['rect'], t['runs'] if len(t['runs']) > 1 or not t.get('wrap') else t['runs'], t['al'])
    c.convert('RGB').save(os.path.join(OUTD, name + '.png'))
print('ok', len(M['screens']))
