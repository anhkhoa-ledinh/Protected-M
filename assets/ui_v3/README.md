# ui_v3 — Protected M, "v2 · Redesign" asset pack

Exported on 07/11/2026 from Penpot, page **"v2 · Redesign"** (boards `v3_*`), plus the helper board `_v3_states`, which holds the state variants that are not drawn on the screens.
- The design space is 1280×720.
- Every PNG is rendered at `image_scale` = **1.5** (1920×1080).
- The loadout's `item_tile:*` and `weapon_tile:*` sprites are rendered at 4×, so they stay sharp when scaled up.
- The look has one theme only (sky / day). There is no night palette.

| Folder | Contents |
|---|---|
| `manifest.json` | The contract (below). |
| `<screen>/base.png` | The full screen with every app-drawn element removed. |
| `<screen>/top.png` | A transparent layer drawn **after** photos and the 3D viewport. Present only when the screen has a photo slot or a viewport. It carries the HUD cards and the decor on photos. It also carries a "corner ring" cut from `base`, so a square photo or viewport gets the design's rounded corners. |
| `sprites/*.png` | Pieces the app shows, hides, moves or swaps. |
| `fonts/` | Static TTFs: Nunito 600/700/800/900, Baloo 2 700/800, Patrick Hand 400, and Inter (arrows only). All SIL OFL. |
| `design/<board>.jpg` | The Penpot boards as designed, 1280×720. This is the visual truth. |
| `expected/<screen>.jpg` | `tools/preview.py` composited from this pack with the design's sample values. Your app screenshots should match these. |
| `tools/preview.py` | A **reference implementation** of the draw contract in PIL. Read it, and copy its logic where it helps. |
| `sample/sample_room.png` | A sample photo. |

## Draw order (every frame)
1. `base`.
2. `photos[]`, in order:
   - `kind` `photo`: the user's photo, cover-fit (scale to fill, centre crop) into `rect`. If `clip` is present, draw it only inside those rects. With no photo, draw nothing, because `base` already shows the sample art.
   - `kind` `viewport`: the HCPD 3D scene. Use a glScissor crop of the full-window Kooima projection, as today.
   - `kind` `camera`: the latest webcam frame, mirrored, cover-fit.
3. `top`, if any.
4. `bars[]`: a rounded rectangle in `col`, `radius`, width = `rect.w × fraction`, never narrower than 2×radius. If `rest_col` is set, first fill the whole track in `rest_col`. For `intact`, that pink part is "eaten".
5. `pips[]`: a row of small shapes. Each one is `on` or `off`. Styles: `fill`, `fill_op`, `stroke`, `sw`, `radius`, `shape`.
6. `pills[]`: a rounded rectangle at `rect.x, rect.y` with height `rect.h`. Width = text width + 2×`pad`. The text is centred and uses `font`. The colours come from one entry of `styles`.
7. `sprites[]`: draw only the ones whose state is active (key rules below).
   - `rect` is where the sprite sits by default.
   - `anchor` is the design element it belongs to.
   - To put a sprite on another element T, draw it at `rect + (T.x − anchor.x, T.y − anchor.y)`.
   - `sel:*` sprites list their possible `targets`.
8. `texts[]`: `runs` share one baseline.
   - Alignment `al` is `left`, `center` or `right`, within `rect`.
   - Centre the line box of the largest run vertically in `rect`.
   - `ls` is letter spacing in px. `upper` means uppercase.
   - Replace the run text with the live value. For a two-run text such as `7` + `/12`, the first run is the live number and the second is `/<total>`.
   - `wrap: true` means wrap at `rect.w` with `line_h`.
9. Hover, focus and messages. Hover: a 2 px `#3B72B4` ring at 45 % opacity around the hovered hotspot. The keyboard focus ring is 2.5 px, offset 3 px.

`hotspots[]` are click rectangles (`n` = name).

## Key rules
`<k>:<state>` sprites are alternatives: draw exactly one of them.

| Screen | Key | Shown when |
|---|---|---|
| sign up | `valid_user`, `valid_pass2`, `hint_ok` | The field is valid: username non-empty; passwords ≥ 8 chars and matching. |
| settings | `seg_theme:day\|night` | Current theme. Night has no palette in v2 (see TASK). |
| settings | `seg_display:full\|window` | Current display mode. |
| settings | `face:ok\|none` | Tracker sees a face. |
| settings | `volume_knob` | Always. Its centre sits at the end of the `volume` bar. |
| level select | `sel:level` | On the selected `level_row_n`. |
| level select | `new_bug` (+ `new_bug_name`) | Level ≥ 2 (the bug introduced at that level). |
| choose photo | `damage_badge` (+ `damage_text`) | The chosen photo was played before and lost part of itself. |
| loadout | `sel:weapon` | On the chosen weapon. |
| loadout | `sticker:<item>` / `sticker_placed` | One of the two for each sticker. `sticker_placed` + `sticker_placed_label` "In slot n" when the item is in a slot. |
| loadout | `slot_empty:n` / `slot_filled` | One of the two for each slot. `slot_filled` is drawn with `item_tile:<item>` + `slot_key/slot_name/slot_sub` (see `slot_layout`). |
| loadout | `tip:need_weapon` / `tip:level` (+ `tip_text`) | No weapon chosen yet / a weapon is chosen. |
| loadout | `btn_start:off\|on` | No weapon / a weapon is chosen. |
| tilt check | `check:done\|current\|todo` | One per checklist row: face, tilt left, tilt right. |
| tilt check | `btn_enter:off\|on` | Rows not done / all three done. |
| tilt check | `glare` | Always. Shift it horizontally with head x. |
| tilt check | `tilt_indicator` (+ `tilt_value`) | Always. |
| HUD | `head:good\|lost` | Tracker has a face. |
| HUD | `crosshair` | Always. It follows the aim point; its centre is the centre of `anchor`. |
| HUD | `st_block` | Stamina active (level 5 / Asian Mode). Hide `st_value` and the `st` bar with it. |
| HUD | `bite` | `eaten` ≥ 10 %. |
| HUD | `effect:<name>` (+ `effect_<name>_left`) | The effect is active. Lay the active ones out left to right from x = 52 with a gap of 8. |
| HUD | `islot:*`, `ikey:*`, `iused`, `item_tile:*` | Item slots (see `item_slot_layout`). |
| HUD | `weapon_icon:Precision gun` | Weapon card icon for the precision gun. For the other two weapons, draw the loadout sprite `weapon_tile:<name>` into the same rect. |
| PvP found | `countdown_arc` | Always. |
| PvP result | `crown` | You won. |
| PvP result | `sel:winner` | On the winner's card (`targets.you` / `targets.opponent`). |

Every `rect` is `[x, y, w, h]` in design space. Font: `runs[].ff` + `fw` → `fonts.files[ff][fw]`.

## Extra fields
- `sprites[].group` + `default`: sprites that share a `group` are alternatives; draw exactly one of them. `default: true` marks the one used in shot mode. A sprite without a group but with `default: false` is a component the app places itself: `item_tile:*`, `weapon_tile:*`, `sticker_placed`, `slot_filled`, `iused`.
- `pips[].sample_on`: which pips are "on" in the design.
- `bars[].sample_frac`: the design's fill fraction.
- `item_slot_layout`, `slot_layout`, `slot_rects`, `sticker_rects`: offsets inside a HUD item slot or a loadout slot, relative to the slot's top-left corner, in design px.
- `screens[].notes`: free-text hints for that screen.

## Shot mode = `expected/`
`expected/<screen>.jpg` is exactly what the app must draw in shot mode:
- every text shows its `sample`;
- every sprite with `default: true` is drawn at its `rect`;
- pips use `sample_on`, bars use `sample_frac`, pills use their first style;
- every `photo` slot shows `sample/sample_room.png`;
- the viewport is a flat RGB(60, 70, 80) and the camera a flat RGB(90, 90, 100);
- there is no hover ring, no focus ring and no caret.

`tools/compare.py <shots_dir> <pack_dir>` scores the screenshots against `expected/`:
- Both images are reduced to 640×360 greyscale.
- The score is the worst mean absolute difference over all 10×10 cells.
- **PASS ≤ 24** for every screen.
- For reference, a 3 px shift or one missing HUD number fails. A different window size (1366×768) scores about 3.5.
