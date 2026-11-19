// Bake Sakuga VFX effects into PNG frames by running the original core.js in headless Chromium.
//   node bake.mjs OUT_DIR
// Writes OUT_DIR/raw/<recipe>_<seed>/NNN.png (transparent, 1920-basis px), OUT_DIR/raw/meta.json and OUT_DIR/raw/words/*.png.
import { chromium } from 'playwright';
import { mkdirSync, writeFileSync, readFileSync } from 'node:fs';
import { join, resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const OUT = resolve(process.argv[2] || 'out');
const RAW = join(OUT, 'raw');
const SCALE = Number(process.env.FX_SCALE || 1.5);  // design px (1280 basis) -> image px. 1.5 = 1920 basis (assets/fx), 2 = 2560 basis (assets/fx@2x)
const FPS = 30;
const VW = 640, VH = 560;        // design px around the anchor
const AX = 320, AY = 320;        // anchor in design px
// recipe: [duration s, seeds]
const PLAN = {
  hit_kill: [1.0, [1, 2]], hit_crit: [1.05, [1, 2]], hit_block: [0.5, [1, 2, 3]],
  ui_confirm: [1.45, [1]], ui_tap: [0.45, [1]],
  star_pop: [0.95, [1, 2]], star_five: [0.95, [1]],
  toast_in: [1.05, [1]], load_done: [1.05, [1]], load_edge: [0.75, [1, 2, 3]],
};

mkdirSync(RAW, { recursive: true });
const browser = await chromium.launch();
const ctx = await browser.newContext({ viewport: { width: VW, height: VH }, deviceScaleFactor: SCALE });
const page = await ctx.newPage();
const errors = [];
page.on('pageerror', (e) => errors.push(String(e)));
const meta = { scale: SCALE, fps: FPS, view: [VW, VH], anchor: [AX, AY], recipes: {} };

for (const [name, [dur, seeds]] of Object.entries(PLAN)) {
  meta.recipes[name] = { duration: dur, variants: [] };
  for (const seed of seeds) {
    await page.goto('file://' + join(HERE, 'bake.html') + '?seed=' + seed);
    await page.waitForFunction(() => !!window.SakugaFX && !!window.runRecipe);
    const words = await page.evaluate(([n, x, y]) => window.runRecipe(n, x, y), [name, AX, AY]);
    const dir = join(RAW, `${name}_${seed}`); mkdirSync(dir, { recursive: true });
    const n = Math.round(dur * FPS);
    for (let f = 0; f < n; f++) {
      await page.evaluate((steps) => { for (let i = 0; i < steps; i++) window.SakugaFX.advance(1 / 60); }, 60 / FPS);
      await page.evaluate(() => new Promise((r) => requestAnimationFrame(() => r())));
      await page.screenshot({ path: join(dir, String(f).padStart(3, '0') + '.png'), omitBackground: true });
    }
    meta.recipes[name].variants.push({ seed, dir: `${name}_${seed}`, frames: n });
    meta.recipes[name].words = words;
    process.stdout.write(`${name}_${seed}: ${n} frames\n`);
  }
}

// ---- words: paint the traced lettering like core.js bake(), at its native traced size
const core = readFileSync(join(HERE, 'vendor/core.js'), 'utf8');
const m = core.match(/^const WORDS = (\{.*\});$/m);
if (!m) throw new Error('WORDS not found in core.js');
const STYLES = { ink: ['#24415F', '#ffffff', false], hollow: ['#ffffff', '#24415F', false], dotted: [null, '#24415F', true], crit: ['#E8676B', '#ffffff', false] };
mkdirSync(join(RAW, 'words'), { recursive: true });
await page.goto('file://' + join(HERE, 'bake.html'));
const wordMeta = {};
for (const w of Object.keys(JSON.parse(m[1]))) {
  for (const [st, [fc, oc, dots]] of Object.entries(STYLES)) {
    const res = await page.evaluate(([WORDS, w, fc, oc, dots]) => {
      const [W, H, d] = WORDS[w], p = new Path2D(d), pad = 24, e = W * 0.035;
      const c = document.createElement('canvas'); c.width = W + pad * 2; c.height = H + pad * 2;
      const x = c.getContext('2d'); x.translate(pad, pad); x.lineJoin = 'round';
      x.strokeStyle = x.fillStyle = oc; x.lineWidth = e * 2; x.stroke(p); x.fill(p);
      if (dots) {
        const t = document.createElement('canvas'); t.width = 22; t.height = 44; const y = t.getContext('2d');
        y.fillStyle = '#fff'; y.fillRect(0, 0, 22, 44); y.fillStyle = oc; y.beginPath();
        for (const [px, py] of [[0, 0], [22, 0], [0, 44], [22, 44], [11, 22]]) { y.moveTo(px + 5.2, py); y.arc(px, py, 5.2, 0, Math.PI * 2); } y.fill();
        x.fillStyle = x.createPattern(t, 'repeat');
      } else x.fillStyle = fc;
      x.fill(p);
      return { url: c.toDataURL('image/png'), traced_w: W, traced_h: H, pad };
    }, [JSON.parse(m[1]), w, fc, oc, dots]);
    writeFileSync(join(RAW, 'words', `${w}_${st}.png`), Buffer.from(res.url.split(',')[1], 'base64'));
    wordMeta[w] = { traced_w: res.traced_w, traced_h: res.traced_h, pad: res.pad };
  }
}
meta.words = wordMeta;
writeFileSync(join(RAW, 'meta.json'), JSON.stringify(meta, null, 1));
await browser.close();
if (errors.length) { console.error('page errors:', errors); process.exit(1); }
