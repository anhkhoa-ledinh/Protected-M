"""The one door every photo goes through before any decoder sees it.

``import_photo`` takes a path from any entry point (file dialog, drag and drop,
``content*.json``, shared album download, PvP photo) and either returns a
:class:`CleanPhoto` or raises :class:`PhotoRejected`. Only a ``CleanPhoto``
is accepted by the rest of the pipeline (``ImageSource``, bake, the shared
album upload), and ``tests/test_architecture.py`` checks that no other module
calls ``cv2.imdecode`` / ``Image.open`` on a user photo.

What the gate does, in order:

1. resolves the real path; it must be a regular file (no device, folder or
   FIFO), and inside ``allowed_root`` when the caller gives one (a symlink
   that leaves the folder is refused);
2. size: not empty, at most ``MAX_FILE_BYTES``;
3. sniffs the magic bytes: PNG, JPEG or WebP only, and the file must end where
   the format says it ends (no ZIP or script glued behind a PNG/JPEG);
4. reads the header size **before** decoding and refuses over ``MAX_PIXELS``
   (Pillow's own bomb limit is set explicitly), and refuses multi-frame files;
5. decodes once from the bytes already in memory, applies the EXIF
   orientation, converts to 8-bit RGB, shrinks the long side to
   ``PIPELINE_MAX_SIDE_PX`` and re-encodes a clean PNG (no EXIF, no ICC, no
   text) into ``cache_dir`` under the sha256 of the pixels.

Nothing but ``cache_dir`` is ever written.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import stat
import struct
import warnings
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import numpy.typing as npt
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

#: What the UI shows for every rejection (the precise reason goes to the log).
USER_MESSAGE = "This file is not a photo we can open"

#: Local files may be bigger than a download (``shared_album.MAX_DOWNLOAD_BYTES``,
#: 20 MiB, stays the stricter cap on the network path): phone photos and the
#: 5823x2700 PNG the pipeline was tuned on are well past 20 MiB.
MAX_FILE_BYTES = 64 * 1024 * 1024
#: Header pixels allowed before decoding (8000 x 8000).
MAX_PIXELS = 64_000_000
#: Long side after the gate. Same value the pipeline always capped to
#: (``image_source.MAX_IMAGE_SIDE_PX``): the depth model sees no difference.
PIPELINE_MAX_SIDE_PX = 2048

#: Pillow raises ``DecompressionBombError`` past twice this and warns past it.
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

_PNG_SIG = b"\x89PNG\r\n\x1a\n"
_EXTENSIONS = {".png": "png", ".jpg": "jpeg", ".jpeg": "jpeg", ".webp": "webp"}
_PIL_FORMATS = {"PNG": "png", "JPEG": "jpeg", "MPO": "jpeg", "WEBP": "webp"}


class PhotoRejected(Exception):
    """The file is not a photo the app will open. ``str(exc)`` is the text the
    UI shows; ``reason`` is the specific cause, for the log."""

    def __init__(self, reason: str) -> None:
        super().__init__(USER_MESSAGE)
        self.reason = reason


@dataclass(frozen=True)
class CleanPhoto:
    """A photo that passed the gate: ``path`` is the clean PNG copy in the
    cache, ``sha256`` the hash of its pixels (also its file name)."""

    path: Path
    sha256: str
    width: int
    height: int


def _reject(reason: str) -> PhotoRejected:
    return PhotoRejected(reason)


# ---- 1 + 2: the file itself -------------------------------------------------

def _read_file(path: Path, allowed_root: Path | None) -> bytes:
    try:
        real = Path(os.path.realpath(path))
        info = os.stat(real)
    except (OSError, ValueError) as exc:
        raise _reject(f"cannot stat: {exc}") from exc
    if not stat.S_ISREG(info.st_mode):
        raise _reject("not a regular file")
    if allowed_root is not None:
        root = Path(os.path.realpath(allowed_root))
        if not real.is_relative_to(root):
            raise _reject("path leaves the allowed folder")
    if info.st_size == 0:
        raise _reject("empty file")
    if info.st_size > MAX_FILE_BYTES:
        raise _reject(f"file too large: {info.st_size} bytes")
    try:
        with open(real, "rb") as handle:
            now = os.fstat(handle.fileno())
            if (not stat.S_ISREG(now.st_mode)
                    or (now.st_ino, now.st_dev) != (info.st_ino, info.st_dev)):
                raise _reject("file changed while opening")
            data = handle.read(MAX_FILE_BYTES + 1)
    except OSError as exc:
        raise _reject(f"cannot read: {exc}") from exc
    if len(data) > MAX_FILE_BYTES:
        raise _reject("file too large")
    return data


# ---- 3: magic bytes and "the file ends where the format ends" ---------------

def _sniff(data: bytes) -> str:
    if data.startswith(_PNG_SIG):
        return "png"
    if data.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    raise _reject("unknown file signature")


def _png_end(data: bytes) -> int:
    """Offset just past IEND; refuses animated (``acTL``) files."""
    pos, size = len(_PNG_SIG), len(data)
    while pos + 12 <= size:
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"acTL":
            raise _reject("animated PNG")
        pos += 12 + length
        if kind == b"IEND":
            if pos > size:
                raise _reject("truncated PNG")
            return pos
    raise _reject("PNG has no IEND")


def _jpeg_end(data: bytes) -> int:
    """Offset just past the final EOI, walking the segments and the
    entropy-coded scans (so an EOI inside an EXIF thumbnail is skipped)."""
    pos, size = 2, len(data)
    while pos + 2 <= size:
        if data[pos] != 0xFF:
            raise _reject("bad JPEG marker")
        marker = data[pos + 1]
        if marker == 0xFF:                       # fill byte
            pos += 1
            continue
        if marker == 0xD9:                       # EOI
            return pos + 2
        if marker in (0x01,) or 0xD0 <= marker <= 0xD8:
            pos += 2
            continue
        if pos + 4 > size:
            break
        length = struct.unpack(">H", data[pos + 2:pos + 4])[0]
        if length < 2:
            raise _reject("bad JPEG segment")
        pos += 2 + length
        if marker == 0xDA:                       # SOS: skip the scan data
            while pos + 1 < size:
                if data[pos] == 0xFF and data[pos + 1] not in (0x00, 0xFF) \
                        and not 0xD0 <= data[pos + 1] <= 0xD7:
                    break
                pos += 1
    raise _reject("JPEG has no EOI")


def _webp_end(data: bytes) -> int:
    declared = struct.unpack("<I", data[4:8])[0] + 8
    if declared > len(data) or declared < 12:
        raise _reject("bad WebP size")
    return declared


def _check_container(data: bytes, kind: str) -> None:
    end = {"png": _png_end, "jpeg": _jpeg_end, "webp": _webp_end}[kind](data)
    if data[end:].strip(b"\x00\r\n \t"):
        raise _reject("data after the end of the image")


# ---- 4 + 5: header, one decode, clean PNG -----------------------------------

def _decode(data: bytes, kind: str,
            max_side_px: int | None = None) -> npt.NDArray[np.uint8]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data), formats=["PNG", "JPEG", "WEBP"]) \
                    as image:
                if _PIL_FORMATS.get(image.format or "") != kind:
                    raise _reject("content does not match its signature")
                width, height = image.size
                if width < 1 or height < 1 or width * height > MAX_PIXELS:
                    raise _reject(f"too many pixels: {width}x{height}")
                if max_side_px is not None and max(width, height) > max_side_px:
                    raise _reject(f"side over {max_side_px}px: {width}x{height}")
                if getattr(image, "n_frames", 1) > 1 or getattr(
                        image, "is_animated", False):
                    raise _reject("more than one frame")
                image.load()
                image = ImageOps.exif_transpose(image)
                if image.mode.startswith("I;16") or image.mode == "I":
                    gray = np.asarray(image, dtype=np.uint32)
                    shift = 8 if image.mode.startswith("I;16") else 0
                    rgb = np.repeat(np.clip(gray >> shift, 0, 255).astype(
                        np.uint8)[..., None], 3, axis=2)
                else:
                    rgb = np.asarray(image.convert("RGB"))
    except PhotoRejected:
        raise
    except (OSError, ValueError, SyntaxError, MemoryError, EOFError,
            Image.DecompressionBombError, Image.DecompressionBombWarning,
            struct.error, KeyError, IndexError, TypeError) as exc:
        raise _reject(f"cannot decode: {type(exc).__name__}: {exc}") from exc
    return np.ascontiguousarray(rgb)


def _cap_side(rgb: npt.NDArray[np.uint8]) -> npt.NDArray[np.uint8]:
    height, width = rgb.shape[:2]
    scale = PIPELINE_MAX_SIDE_PX / float(max(height, width))
    if scale >= 1.0:
        return rgb
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)


def _pixel_hash(rgb: npt.NDArray[np.uint8]) -> str:
    height, width = rgb.shape[:2]
    digest = hashlib.sha256(f"{width}x{height}:".encode())
    digest.update(np.ascontiguousarray(rgb).tobytes())
    return digest.hexdigest()


def default_cache_dir() -> Path:
    """Where clean copies live: ``assets/cache/photos`` of this checkout
    (``HCPD_PHOTO_CACHE`` overrides it, tests point it at a temp folder)."""
    override = os.environ.get("HCPD_PHOTO_CACHE")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "assets" / "cache" / "photos"


def import_photo(path: Path | str, cache_dir: Path | str | None = None, *,
                 allowed_root: Path | str | None = None,
                 strict_extension: bool = False,
                 max_side_px: int | None = None) -> CleanPhoto:
    """Run ``path`` through the gate; see the module docstring.

    ``allowed_root``: the folder the file must really live in (symlinks are
    resolved first). Callers that read a cache folder pass it; a file the user
    picked themselves has no root.

    ``strict_extension``: also require that the extension names the same
    format as the content. Off by default because real photos are often a
    JPEG saved as ``.png`` (``t21.png`` is one); the gate sniffs the content
    and writes a PNG either way, so the label cannot change what is decoded.
    The extension must still be one of png/jpg/jpeg/webp.

    ``max_side_px``: refuse (before decoding) a long side over this many pixels;
    the shared album uses it for its own 4096 px rule.
    """
    source = Path(path)
    try:
        kind_by_name = _EXTENSIONS.get(source.suffix.lower())
        if kind_by_name is None:
            raise _reject(f"extension not allowed: {source.suffix!r}")
        data = _read_file(source, Path(allowed_root) if allowed_root else None)
        kind = _sniff(data)
        if strict_extension and kind != kind_by_name:
            raise _reject(f"extension says {kind_by_name}, content is {kind}")
        _check_container(data, kind)
        rgb = _cap_side(_decode(data, kind, max_side_px))
        digest = _pixel_hash(rgb)
        height, width = rgb.shape[:2]
        out_dir = Path(cache_dir) if cache_dir is not None else default_cache_dir()
        out_dir.mkdir(parents=True, exist_ok=True)
        dest = out_dir / f"{digest}.png"
        if not dest.is_file():
            buffer = io.BytesIO()
            Image.fromarray(rgb, "RGB").save(buffer, "PNG", compress_level=3)
            tmp = out_dir / f".{digest}.{os.getpid()}.tmp"
            try:
                tmp.write_bytes(buffer.getvalue())
                os.replace(tmp, dest)
            finally:
                tmp.unlink(missing_ok=True)
    except PhotoRejected as exc:
        logger.warning("Photo rejected (%s): %s", exc.reason, source.name)
        raise
    except OSError as exc:
        logger.warning("Photo rejected (cannot write clean copy: %s): %s",
                       exc, source.name)
        raise PhotoRejected(f"cannot write clean copy: {exc}") from exc
    return CleanPhoto(dest, digest, int(width), int(height))


def load_rgb(photo: CleanPhoto) -> npt.NDArray[np.uint8]:
    """Pixels (H, W, 3, uint8) of a clean copy. The copy is checked against
    the hash it is named after, so a file swapped in the cache is refused."""
    if not isinstance(photo, CleanPhoto):
        raise TypeError("load_rgb needs a CleanPhoto (use media_gate.import_photo)")
    try:
        with Image.open(photo.path, formats=["PNG"]) as image:
            rgb = np.ascontiguousarray(np.asarray(image.convert("RGB")))
    except (OSError, ValueError) as exc:
        raise ValueError(f"clean copy unreadable: {photo.path.name}") from exc
    if _pixel_hash(rgb) != photo.sha256:
        raise ValueError(f"clean copy does not match its hash: {photo.path.name}")
    return rgb
