"""
Shared photo album on Cloudinary (TASK_shared_album_1111 Phase 2/3).

Unsigned upload only — no key, no signature, no Admin API/SDK. Reads
cloud_name/upload_preset/tag from ``assets/config/shared_album.json``;
missing file or an empty field disables the feature. Network I/O is plain
``urllib.request``; every call times out at 15 s and returns an error
*value*, never an exception — callers (off the GL thread) toast it.

Hardening (11/11): ``list_remote()``/``download()`` read data that ANYONE
with the unsigned upload preset can have written — a ``public_id`` is not a
filename and a response is not guaranteed well-formed. ``download()`` never
builds a path from ``public_id`` (hashed instead), URLs are built with
``urllib.parse.quote``, and the body is capped while streaming.
"""

from __future__ import annotations

import hashlib
import http.client
import io
import json
import logging
import mimetypes
import random
import ssl
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import quote

from PIL import Image

from src.content import media_gate

logger = logging.getLogger(__name__)

CONFIG_PATH = Path("assets/config/shared_album.json")
MAX_SIDE_PX = 2048
JPEG_QUALITY = 90
MAX_BYTES = 10 * 1024 * 1024
MAX_DOWNLOAD_BYTES = 20 * 1024 * 1024
#: P3.1: "verifies it decodes as an image and is <= 4096 px on the long side".
MAX_PHOTO_SIDE_PX = 4096
TIMEOUT_S = 15.0
NOT_CONFIGURED = "Shared album not configured"
EMPTY_ALBUM = "The shared album is empty"
NO_CONNECTION = "No connection"
LIST_BLOCKED = "Album list is blocked in the Cloudinary account"
#: Format whitelist for ``download()`` — the only values the filename may
#: end in. Anything else (incl. empty/weird strings from a bad response)
#: is rejected before touching the filesystem.
ALLOWED_FORMATS = frozenset({"jpg", "jpeg", "png", "webp"})
#: Network errors that must become an error *value*, never propagate.
_NETWORK_ERRORS = (urllib.error.URLError, TimeoutError, OSError,
                  http.client.HTTPException, ValueError)
_SSL_CTX = ssl.create_default_context()
_CHUNK = 65536


def load_config(path: Path | None = None) -> dict | None:
    """
    ``{cloud_name, upload_preset, tag}``, or ``None`` if disabled.

    ``upload_preset`` may come back as ``""``: it is only needed by
    ``upload()`` (checked there), while ``list_remote()``/``download()``
    need just ``cloud_name`` + ``tag`` — an account that only ever shares
    a tag publicly, with no unsigned preset, still gets Phase 3 working.
    """
    try:
        raw = json.loads((path or CONFIG_PATH).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    fields = {k: str(raw.get(k) or "").strip()
              for k in ("cloud_name", "upload_preset", "tag")}
    return None if not fields["cloud_name"] or not fields["tag"] else fields


def can_upload(config: dict | None = None) -> bool:
    """Whether ``upload()`` (as opposed to list/download) is usable."""
    config = load_config() if config is None else config
    return bool(config and config["upload_preset"])


def _reencode(path) -> bytes | None:
    """JPEG, long side <= 2048 px, quality 90, no EXIF. ``None`` = not a photo.

    The file is read only through ``media_gate`` (checked, EXIF turned into
    pixels, long side capped); here it is just encoded as JPEG."""
    try:
        rgb = media_gate.load_rgb(media_gate.import_photo(path))
    except (media_gate.PhotoRejected, ValueError):
        return None
    image = Image.fromarray(rgb, "RGB")
    image.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=JPEG_QUALITY)  # no exif= kwarg: EXIF dropped
    return buffer.getvalue()


def _multipart(fields: dict, filename: str, content: bytes) -> tuple[bytes, str]:
    """Hand-built ``multipart/form-data`` body — no field looks like a key."""
    boundary = uuid.uuid4().hex
    chunks = []
    for key, value in fields.items():
        chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                       f'name="{key}"\r\n\r\n{value}\r\n').encode("utf-8"))
    ctype = mimetypes.guess_type(filename)[0] or "application/octet-stream"
    chunks.append((f"--{boundary}\r\nContent-Disposition: form-data; "
                   f'name="file"; filename="{filename}"\r\n'
                   f"Content-Type: {ctype}\r\n\r\n").encode("utf-8"))
    chunks.append(content)
    chunks.append(f"\r\n--{boundary}--\r\n".encode("utf-8"))
    return b"".join(chunks), f"multipart/form-data; boundary={boundary}"


def _image_url(cloud_name: str, item: dict) -> str | None:
    """Public image URL from a (possibly remote, untrusted) ``item``.

    ``public_id`` is quoted with ``safe="/"``: Cloudinary folders are real
    "/"-separated path segments in the delivery URL, and keeping them is
    safe here — the file *on disk* (``download()``) is already named from a
    hash of ``public_id``, never from the value itself, so a folder-style
    public_id cannot turn into a path.

    ``None`` — never a raised ``KeyError`` — if a required field is missing,
    e.g. a response with no "version" (hardening 11/11)."""
    try:
        public_id = quote(str(item["public_id"]), safe="/")
        version = quote(str(item["version"]), safe="")
        fmt = quote(str(item["format"]), safe="")
    except (KeyError, TypeError):
        return None
    return (f"https://res.cloudinary.com/{quote(cloud_name, safe='')}/image/"
           f"upload/v{version}/{public_id}.{fmt}")


def _read_capped(response, limit: int) -> bytes | None:
    """``response`` read in chunks; ``None`` if it would exceed ``limit``."""
    parts, total = [], 0
    while True:
        chunk = response.read(_CHUNK)
        if not chunk:
            return b"".join(parts)
        total += len(chunk)
        if total > limit:
            return None
        parts.append(chunk)


def upload(path) -> str:
    """Re-encode and upload ``path``; the public URL, or a short error text."""
    config = load_config()
    if not can_upload(config):
        return NOT_CONFIGURED
    content = _reencode(path)
    if content is None:
        return "Not a photo"
    if len(content) > MAX_BYTES:
        return "Photo too large after re-encoding"
    body, content_type = _multipart({"upload_preset": config["upload_preset"]},
                                    "photo.jpg", content)
    url = f"https://api.cloudinary.com/v1_1/{quote(config['cloud_name'], safe='')}/image/upload"
    request = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": content_type})
    try:
        with urllib.request.urlopen(request, timeout=TIMEOUT_S,
                                    context=_SSL_CTX) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        message = err.headers.get("x-cld-error", "") if err.headers else ""
        try:
            payload = json.loads(err.read(65536).decode("utf-8"))
            detail = payload.get("error") if isinstance(payload, dict) else None
            if isinstance(detail, dict) and isinstance(detail.get("message"), str):
                message = detail["message"] or message
        except _NETWORK_ERRORS:
            pass
        finally:
            err.close()
        if message:
            return ("Upload failed: " + " ".join(message.split()))[:60]
        return f"Upload failed (server said {err.code})"
    except _NETWORK_ERRORS:
        return NO_CONNECTION
    if not isinstance(data, dict):
        return "Upload response was malformed"
    logger.info("Shared album: uploaded, public_id=%s", str(data.get("public_id", "")))
    return _image_url(config["cloud_name"], data) or "Upload response was malformed"


def list_remote() -> list[dict] | str | None:
    """Tagged images, ``LIST_BLOCKED`` for restricted lists, or ``None``."""
    config = load_config()
    if config is None:
        return None
    tag = quote(config["tag"], safe="")
    url = (f"https://res.cloudinary.com/{quote(config['cloud_name'], safe='')}"
          f"/image/list/{tag}.json")
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S,
                                    context=_SSL_CTX) as response:
            data = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        if err.code in (401, 404):
            logger.warning("Shared album: list bi chan (%d) — kiem tra "
                           "'Resource list' trong security settings cua "
                           "tai khoan Cloudinary", err.code)
            return LIST_BLOCKED
        return None
    except _NETWORK_ERRORS:
        return None
    if not isinstance(data, dict):
        return None
    resources = data.get("resources", ())
    return list(resources) if isinstance(resources, list) else None


def download(item: dict, dest_dir) -> Path | None:
    """
    Fetch ``item`` (an entry of ``list_remote()``) into ``dest_dir``.

    ``item`` comes from Cloudinary — NOT trusted: the filename is
    ``sha1(public_id)`` + a format checked against ``ALLOWED_FORMATS``,
    never ``public_id`` itself (no "../../x", no "a/b" becoming a
    subdirectory), and the resolved path is confirmed to stay inside
    ``dest_dir`` before anything is written.
    """
    config = load_config()
    if config is None:
        return None
    try:
        public_id = str(item["public_id"])
        fmt = str(item["format"]).strip().lower()
    except (KeyError, TypeError):
        return None
    if fmt not in ALLOWED_FORMATS:
        return None
    dest_dir = Path(dest_dir).resolve()
    digest = hashlib.sha1(public_id.encode("utf-8")).hexdigest()
    dest = (dest_dir / f"{digest}.{fmt}").resolve()
    if dest.parent != dest_dir or not dest.is_relative_to(dest_dir):
        return None
    if dest.exists():
        return dest
    url = _image_url(config["cloud_name"], item)
    if url is None:
        return None
    try:
        with urllib.request.urlopen(url, timeout=TIMEOUT_S,
                                    context=_SSL_CTX) as response:
            content = _read_capped(response, MAX_DOWNLOAD_BYTES)
    except _NETWORK_ERRORS:
        return None
    if content is None:
        return None
    try:
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(content)
    except OSError:
        return None
    return dest


def pick_random(items: list[dict], rng: random.Random | None = None) -> dict:
    """Uniformly random item — ``rng`` makes the pick reproducible in tests."""
    return (rng or random).choice(items)


def verify_photo(path, max_side: int = MAX_PHOTO_SIDE_PX) -> bool:
    """Passes ``media_gate`` and is no larger than ``max_side`` on the long side."""
    try:
        media_gate.import_photo(path, allowed_root=Path(path).parent,
                                max_side_px=max_side)
    except media_gate.PhotoRejected:
        return False
    return True


def pick_random_and_download(dest_dir, rng: random.Random | None = None) -> Path | str:
    """
    P3: list → pick one at random → download → verify, in one off-GL-thread call.

    Returns the cached :class:`Path` on success, or a toast-ready error text
    (``EMPTY_ALBUM``/``NO_CONNECTION``) — never raises.
    """
    items = list_remote()
    if isinstance(items, str):
        return items
    if items is None:
        return NO_CONNECTION
    if not items:
        return EMPTY_ALBUM
    path = download(pick_random(items, rng), dest_dir)
    if path is None:
        return NO_CONNECTION
    if not verify_photo(path):
        try:
            path.unlink()
        except OSError:
            pass
        return NO_CONNECTION
    return path
