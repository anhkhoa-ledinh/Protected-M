"""Header-only checks for a photo the server relays between two players.

The server never decodes pixels (no Pillow, no OpenCV): it checks size,
magic bytes and the dimensions written in the header, then forwards the bytes.
The receiving game still runs its own media gate (``src.content.media_gate``).
The limits here are stricter than the gate's on bytes (a relayed photo is
already re-encoded by the sender) and equal on pixels."""

from __future__ import annotations

import struct

MAX_PHOTO_BYTES = 4 * 1024 * 1024
#: Same number as ``media_gate.MAX_PIXELS`` (a test keeps them equal).
MAX_PIXELS = 64_000_000
MAX_SIDE = 16_384


class PhotoRefused(ValueError):
    pass


def _jpeg_size(data: bytes) -> tuple[int, int] | None:
    i, n = 2, len(data)
    while i + 9 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xFF:
            i += 1
            continue
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            h, w = struct.unpack(">HH", data[i + 5:i + 9])
            return w, h
        if marker in (0xD8, 0x01) or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        (length,) = struct.unpack(">H", data[i + 2:i + 4])
        if length < 2:
            return None
        i += 2 + length
    return None


def _png_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 24 or data[12:16] != b"IHDR":
        return None
    return struct.unpack(">II", data[16:24])


def _webp_size(data: bytes) -> tuple[int, int] | None:
    if len(data) < 30:
        return None
    kind = data[12:16]
    if kind == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
        w, h = struct.unpack("<HH", data[26:30])
        return w & 0x3FFF, h & 0x3FFF
    if kind == b"VP8L" and data[20] == 0x2F:
        bits = struct.unpack("<I", data[21:25])[0]
        return (bits & 0x3FFF) + 1, ((bits >> 14) & 0x3FFF) + 1
    if kind == b"VP8X":
        w = int.from_bytes(data[24:27], "little") + 1
        h = int.from_bytes(data[27:30], "little") + 1
        return w, h
    return None


def check(data: bytes) -> tuple[str, int, int]:
    """``(format, width, height)`` or :class:`PhotoRefused`."""
    if not data or len(data) > MAX_PHOTO_BYTES:
        raise PhotoRefused("size")
    if data[:3] == b"\xff\xd8\xff":
        fmt, size = "jpeg", _jpeg_size(data)
    elif data[:8] == b"\x89PNG\r\n\x1a\n":
        fmt, size = "png", _png_size(data)
    elif data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        fmt, size = "webp", _webp_size(data)
    else:
        raise PhotoRefused("format")
    if size is None:
        raise PhotoRefused("header")
    w, h = size
    if w < 1 or h < 1 or max(w, h) > MAX_SIDE or w * h > MAX_PIXELS:
        raise PhotoRefused("pixels")
    return fmt, w, h
