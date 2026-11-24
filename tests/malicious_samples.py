"""Crafted files for the photo gate tests (and ``scripts/photo_gate_demo.py``).

Everything is built here from Pillow and the standard library: no download,
no binary fixture in git. ``make_samples(folder)`` writes one file per attack
and returns ``{name: path}``.
"""

from __future__ import annotations

import io
import os
import struct
import zipfile
import zlib
from pathlib import Path

from PIL import Image


def png_bytes(width: int = 64, height: int = 48, colour=(200, 40, 40)) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, "PNG")
    return buffer.getvalue()


def jpeg_bytes(width: int = 64, height: int = 48, colour=(40, 200, 40), **kw) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), colour).save(buffer, "JPEG", **kw)
    return buffer.getvalue()


def zip_bytes() -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as archive:
        archive.writestr("payload.txt", "not a photo")
    return buffer.getvalue()


def png_with_header_size(width: int, height: int) -> bytes:
    """A small valid-looking PNG whose IHDR claims ``width`` x ``height``."""
    data = bytearray(png_bytes())
    ihdr = struct.pack(">II", width, height) + bytes(data[24:29])
    data[16:29] = ihdr
    data[29:33] = struct.pack(">I", zlib.crc32(b"IHDR" + ihdr) & 0xFFFFFFFF)
    return bytes(data)


def animated_png() -> bytes:
    frames = [Image.new("RGB", (32, 32), c) for c in ((255, 0, 0), (0, 0, 255))]
    buffer = io.BytesIO()
    frames[0].save(buffer, "PNG", save_all=True, append_images=frames[1:])
    return buffer.getvalue()


def animated_webp() -> bytes:
    frames = [Image.new("RGB", (32, 32), c) for c in ((255, 0, 0), (0, 0, 255))]
    buffer = io.BytesIO()
    frames[0].save(buffer, "WEBP", save_all=True, append_images=frames[1:],
                   duration=100)
    return buffer.getvalue()


def make_samples(folder: Path) -> dict[str, Path]:
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    outside = folder.parent / (folder.name + "_outside")
    outside.mkdir(exist_ok=True)
    samples: dict[str, Path] = {}

    def put(name: str, data: bytes) -> None:
        samples[name] = folder / name
        samples[name].write_bytes(data)

    good = png_bytes()
    put("truncated.png", good[: len(good) // 2])
    put("jpeg_renamed.png", jpeg_bytes())
    put("huge_header.png", png_with_header_size(100_000, 100_000))
    put("bomb_just_over.png", png_with_header_size(9_000, 9_000))
    put("text_file.jpg", b"#!/bin/sh\necho this is not a photo\n" * 50)
    put("polyglot_png_zip.png", good + zip_bytes())
    put("polyglot_jpeg_zip.jpg", jpeg_bytes() + zip_bytes())
    put("empty.png", b"")
    put("animated.png", animated_png())
    put("animated.webp", animated_webp())
    put("wrong_extension.txt", good)

    real = outside / "secret.png"
    real.write_bytes(good)
    link = folder / "escape.png"
    try:
        link.symlink_to(real)
        samples["escape.png"] = link
    except (OSError, NotImplementedError):
        pass
    samples["folder.png"] = folder / "folder.png"
    samples["folder.png"].mkdir(exist_ok=True)
    if hasattr(os, "mkfifo"):
        samples["fifo.png"] = folder / "fifo.png"
        os.mkfifo(samples["fifo.png"])
    return samples
