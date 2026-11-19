# sakuga_bake

Bakes the drawn effects of [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx) (MIT, Justin Finkelman) into the sprite sheets in `assets/fx/`.
The game cannot run the original JavaScript (it is a Python + OpenGL app), so the original `core.js` runs in headless Chromium,
the clock is stepped at 30 fps and every frame is saved; `pack.py` trims the frames and packs one atlas per variant.

The baked files are committed. You only need this folder to change or add an effect.

- `vendor/core.js`, `vendor/LICENSE`: the upstream core, unmodified (commit in `vendor/SOURCE.txt`).
- `bake.html`: the recipes (which effects fire where, in design px on the 1280x720 basis) and the ui_v3 palette.
  Words are not drawn into the sheets; each recipe returns word cues and the game draws `words/*.png` itself.
- `bake.mjs`: drives Chromium. Needs Node 22+ and the `playwright` package with Chromium.
- `pack.py`: trims and packs, writes `fx.json`.
- `fxref.py`: reference Python port of the word motion (the `Spring` and `sfx()` from core.js). The game's port must match it.
- `preview.py`: contact sheet, to look at the result.

```
node tools/sakuga_bake/bake.mjs build/fxbake
python tools/sakuga_bake/pack.py build/fxbake/raw assets/fx
python tools/sakuga_bake/preview.py assets/fx build/fx_preview.jpg
```

Randomness is seeded (`?seed=N`), so a re-bake gives the same frames.

`assets/fx@2x` (for screens wider than 1920 px) is the same bake at scale 2; atlases may be up to 4096 px wide:

```
FX_SCALE=2 node tools/sakuga_bake/bake.mjs build/fxbake2x
FX_MAXW=4096 python tools/sakuga_bake/pack.py build/fxbake2x/raw assets/fx@2x
```

`pack.py` fills RGB under fully transparent pixels with the nearby colour (`bleed()`, alpha untouched) and writes `rgb_bled: true`, so the game needs no colour bleed at load time.
