# Third-party notices

Protected M includes or uses the third-party material below. Each item keeps
its own license. Neither the MIT License ([LICENSE](LICENSE)) nor CC BY-NC 4.0
([LICENSE-ASSETS.md](LICENSE-ASSETS.md)) applies to it.

## Included in this repository

| File(s) | What it is | License |
|---|---|---|
| `assets/models/face_landmarker.task` | MediaPipe Face Landmarker model, Google LLC | Apache License 2.0. Full text: [licenses/Apache-2.0.txt](licenses/Apache-2.0.txt) |
| `assets/ui_v3/fonts/*.ttf`, `assets/ui_album/fonts/*.ttf` | Fonts: Baloo 2, Inter, Nunito, Patrick Hand, Barlow Condensed | SIL Open Font License 1.1. Copyright holders and full text are in the `OFL_*.txt` file next to each font |
| `assets/ui_album/sprites/b1_wk_topdown.png`, `b2_wk_topdown.png`, `b3_wk_topdown.png`, `b4_wk_topdown.png`, `wk_bugs_topdown_source_preview.png` | Bug sprites from [Top-down Alien Bug Sprites](https://whiteknightstudios.itch.io/top-down-bug-sprites) by W_K_Studio. The four sprites are cut, unchanged, from the pack's public preview image, which is kept as `wk_bugs_topdown_source_preview.png` | CC0 1.0 Universal (public domain dedication), as stated on the author's page |
| `assets/fx/**`, `tools/sakuga_bake/vendor/core.js` | Visual-effect atlases and sound-effect-word PNGs, rendered from [Sakuga VFX](https://github.com/finktheartist/sakuga-vfx) by Justin Finkelman at commit `c3dabb173928a3ad2372183103caf2e6f50953d0`, then recoloured to the ui_v3 palette. The lettering in `assets/fx/words/` comes from the library as-is. | MIT License. Full text: [licenses/MIT-sakuga-vfx.txt](licenses/MIT-sakuga-vfx.txt) |

## Downloaded or installed separately (not in this repository)

| Item | How you get it | License |
|---|---|---|
| Depth Anything V1 Small (`LiheYoung/depth-anything-small-hf`) | Downloaded by `transformers` on first run | Apache License 2.0 |
| LaMa ONNX (`lama_fp32.onnx`, [Carve/LaMa-ONNX](https://huggingface.co/Carve/LaMa-ONNX)) | Optional; you place it in `assets/models/` | Apache License 2.0 |
| Python packages in `requirements.txt` | Installed with `pip` | Each package's own license |
