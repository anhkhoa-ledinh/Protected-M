"""TASK_perf_textures_1119 Fix 1: pre-bleed UI PNGs offline (invisible change).

``src/ui_album/textures.py``'s ``bleed_transparent`` fills RGB under alpha=0
pixels with the colour of nearby opaque pixels, at full image resolution,
on the render thread, the first time each screen draws — ~300 ms for a
mostly-transparent 1920x1080 top layer (TASK_perf_textures_1119). Doing the
same fill OFFLINE, once, removes that cost from every future run.

Same algorithm as ``tools/sakuga_bake/pack.py``'s ``bleed()`` (not imported
from there — that file is a script, not a module): a 4 px band of the
nearby colour (progressive radius 1/2/4 blur, closest wins), then one flat
colour (the overall weighted average) for anything still untouched beyond
that band. Alpha is never touched.

Each rewritten file gets a PNG text chunk ``rgb_bled=1`` so the runtime
loader (Fix 2) can skip its own bleed pass for it.

Usage:
    python tools/prebleed.py             # bleed + report size change
    python tools/prebleed.py --check     # list files that WOULD change, no writes
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

ROOT = Path(__file__).resolve().parents[1]

RGB_BLED_KEY = "rgb_bled"


def bleed(im: Image.Image) -> Image.Image:
    """Same algorithm as ``tools/sakuga_bake/pack.py``'s ``bleed()``."""
    a = np.asarray(im, dtype=np.float32)
    rgb, al = a[..., :3], a[..., 3:4] / 255.0
    empty = al[..., 0] == 0
    if not empty.any() or empty.all():
        return im
    out = rgb.copy()
    todo = empty.copy()
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
    return Image.fromarray(res, "RGBA")


def is_marked(path: Path) -> bool:
    with Image.open(path) as im:
        return im.text.get(RGB_BLED_KEY) == "1" if hasattr(im, "text") else False


def has_mixed_alpha(path: Path) -> bool:
    with Image.open(path) as im:
        if im.mode != "RGBA":
            return False
        alpha = np.asarray(im.convert("RGBA"))[..., 3]
    return bool(np.any(alpha == 0)) and bool(np.any(alpha > 0))


def runtime_png_refs(manifest_path: Path, asset_root: Path) -> set[Path]:
    """Every PNG the app can load for this manifest: ``base``/``top`` of
    every screen, plus every screen's ``sprites[*].file``."""
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    refs: set[str] = set()
    for screen in data["screens"].values():
        for layer in ("base", "top"):
            v = screen.get(layer)
            if isinstance(v, str):
                refs.add(v)
            elif isinstance(v, dict):          # legacy album manifest: {theme: file}
                refs.update(x for x in v.values() if isinstance(x, str))
        for sprite in screen.get("sprites", ()):
            f = sprite.get("file")
            if f:
                refs.add(f)
    for rec in data.get("sprites", {}).values():       # legacy album: top-level sprite table
        if isinstance(rec, dict):
            refs.update(v for v in rec.values() if isinstance(v, str) and v.lower().endswith(".png"))
    return {asset_root / r for r in refs if r.lower().endswith(".png")}


def discover_targets() -> list[Path]:
    targets: set[Path] = set()
    targets |= runtime_png_refs(ROOT / "assets" / "ui_v3" / "manifest.json",
                                ROOT / "assets" / "ui_v3")
    album_manifest = ROOT / "assets" / "ui_album" / "manifest.json"
    if album_manifest.exists():
        targets |= runtime_png_refs(album_manifest, ROOT / "assets" / "ui_album")
    return sorted(p for p in targets if p.exists())


def main(argv: list[str]) -> int:
    check_only = "--check" in argv
    targets = discover_targets()
    changed: list[tuple[Path, int, int]] = []
    skipped_marked = 0
    skipped_opaque = 0
    for path in targets:
        if is_marked(path):
            skipped_marked += 1
            continue
        if not has_mixed_alpha(path):
            skipped_opaque += 1
            continue
        before = path.stat().st_size
        if check_only:
            changed.append((path, before, -1))
            continue
        with Image.open(path) as im:
            bled = bleed(im.convert("RGBA"))
        info = PngInfo()
        info.add_text(RGB_BLED_KEY, "1")
        bled.save(path, optimize=True, pnginfo=info)
        after = path.stat().st_size
        changed.append((path, before, after))

    total_before = sum(b for _p, b, _a in changed)
    total_after = sum((a if a >= 0 else b) for _p, b, a in changed)
    verb = "WOULD bleed" if check_only else "Bled"
    print(f"{verb} {len(changed)} file(s) "
         f"({skipped_marked} already marked, {skipped_opaque} fully opaque/transparent, skipped)")
    for path, before, after in changed:
        rel = path.relative_to(ROOT)
        if after >= 0:
            print(f"  {rel}: {before} -> {after} bytes ({after - before:+d})")
        else:
            print(f"  {rel}: {before} bytes")
    if not check_only:
        print(f"Total: {total_before} -> {total_after} bytes ({total_after - total_before:+d})")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
