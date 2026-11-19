# assets/fx: baked Sakuga effects

Made by `tools/sakuga_bake/` from [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx) (MIT, Justin Finkelman), recoloured to the ui_v3 palette.
These files are third-party art under MIT (see `THIRD_PARTY_NOTICES.md`), not under the CC BY-NC asset license.

## fx.json

- `image_scale` 1.5: sheet px = design px x 1.5 (1920x1080 basis, same as `assets/ui_v3`).
- `rgb_bled` true: RGB under fully transparent pixels is already filled with the nearby colour (4 px band, flat colour beyond), so mipmaps get no dark fringe. Alpha and every visible pixel are unchanged. A loader must **skip** its own colour bleed for this pack.
- `fps` 30. Frame `f` is shown for `t` in `[f/30, (f+1)/30)` after the effect is spawned.
- `effects.<name>.variants[i]`:
  - `sheet`: the atlas PNG.
  - `frames[f]`: `{rect: [x, y, w, h]}` in sheet px and `offset: [ox, oy]`, the frame's top-left relative to the anchor in sheet px.
    `null` is an empty frame. The effect's sprite part ends after `len(frames)` frames.
  - Mirrored playback (for variety): draw the frame flipped horizontally at `ox' = -ox - w`. Words are not mirrored.
- `effects.<name>.words`: word cues `{word, style, at, dx, dy, size, rot, life}`.
  - `dx`, `dy`: design px from the anchor.
  - `size`: the word's traced width in design px.
  - `at`: start delay in seconds.
  - Draw `words/<word>_<style>.png` centred, with the motion in `tools/sakuga_bake/fxref.py` (`WordAnim`).
    The PNG is the traced word plus `pad` px on each side, so its drawn width is `png_width * scale * image_scale` sheet px.
- An effect is finished when its frames and all its words are finished.

## Effects

| name | where | anchor |
|---|---|---|
| `hit_kill` | a bug is killed | the hit point |
| `hit_crit` | a bug is killed by a head hit | the hit point |
| `hit_block` | any miss: the shot lands on the photo or on a nearer layer | the crosshair |
| `ui_confirm` | a primary button is activated | the button centre (made for ~220x56) |
| `ui_tap` | any other button | the button centre |
| `toast_in` | a toast has just landed | the toast centre (made for ~320x56) |
| `load_edge` | the layer-split bar is filling, every ~0.25 s | the bar's leading edge |
| `load_done` | the layer-split bar reached 100 % | the bar's right end |
| `star_pop`, `star_five` | step/tilt/slot pops; WIN | see `src/fx/` hooks |

Words: `don` (impact), `pon` (pop), `zuza` (skid), `gogo` (rumble), `kachi` (click), `shu` (swish).
Styles: `ink` (navy fill, white edge), `hollow` (white fill, navy edge), `dotted` (screentone), `crit` (red fill, white edge).

## assets/fx@2x

The same effects baked at `image_scale` 2 (2560x1440 basis), same seeds, same frames and words; only the sheets are bigger. Use it only when the framebuffer is wider than 1920 px: at 1920 or below this 1.5x pack is already at or above screen resolution, and the 2x pack costs about 1.8x the memory.
