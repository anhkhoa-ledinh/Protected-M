# assets/fx@2x

The Sakuga effects of `assets/fx` baked at `image_scale` 2 (2560x1440 basis): same recipes, seeds, frames, offsets (scaled) and words, sharper sheets. Same MIT notice as `assets/fx` (see `THIRD_PARTY_NOTICES.md`). Read `assets/fx/README.md` for the format.

Used only when the framebuffer is wider than 1920 px. Made with:

```
FX_SCALE=2 node tools/sakuga_bake/bake.mjs build/fxbake2x
FX_MAXW=4096 python tools/sakuga_bake/pack.py build/fxbake2x/raw assets/fx@2x
```
