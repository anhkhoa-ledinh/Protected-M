"""The photo gate (``src/content/media_gate.py``): crafted files are refused
without a crash, a normal photo passes and its clean copy carries no metadata,
and nothing outside the cache folder is written."""

from __future__ import annotations

import io
import random
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import malicious_samples as M  # noqa: E402

from src.content import media_gate as G  # noqa: E402

#: Every attack the gate must refuse in its default mode.
REJECTED = ["truncated.png", "huge_header.png", "bomb_just_over.png",
            "text_file.jpg", "empty.png", "animated.png", "animated.webp", "wrong_extension.txt",
            "folder.png", "fifo.png"]


def _files(root: Path) -> set[Path]:
    return {p for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


@pytest.fixture
def world(tmp_path):
    samples = M.make_samples(tmp_path / "malicious")
    return samples, tmp_path / "cache", tmp_path


@pytest.mark.parametrize("name", REJECTED)
def test_crafted_file_is_rejected_without_a_crash(world, name):
    samples, cache, _ = world
    if name not in samples:
        pytest.skip("not supported on this platform")
    with pytest.raises(G.PhotoRejected) as info:
        G.import_photo(samples[name], cache)
    assert str(info.value) == "This file is not a photo we can open"
    assert info.value.reason


def test_rejected_files_write_nothing(world):
    samples, cache, tmp = world
    before = _files(tmp)
    for name in REJECTED:
        if name in samples:
            with pytest.raises(G.PhotoRejected):
                G.import_photo(samples[name], cache)
    assert _files(tmp) == before          # no clean copy, no temp file left


@pytest.mark.parametrize("name", ["polyglot_png_zip.png", "polyglot_jpeg_zip.jpg"])
def test_extra_bytes_after_the_image_are_dropped_with_a_warning(world, name, caplog):
    samples, cache, _ = world
    with caplog.at_level("WARNING", logger="src.content.media_gate"):
        clean = G.import_photo(samples[name], cache)
    assert any("extra bytes" in r.getMessage() for r in caplog.records)
    raw = clean.path.read_bytes()
    assert b"PK\x03\x04" not in raw and b"payload.txt" not in raw
    with Image.open(clean.path) as image:
        assert image.format == "PNG" and image.size == (64, 48)
    assert G.load_rgb(clean).shape == (48, 64, 3)


def test_samsung_style_jpeg_with_trailing_data_loads_clean(tmp_path):
    path = tmp_path / "motion.jpg"
    path.write_bytes(M.jpeg_bytes() + b"MotionPhoto_Data" + bytes(range(256)) * 20)
    clean = G.import_photo(path, tmp_path / "cache")
    raw = clean.path.read_bytes()
    assert b"MotionPhoto_Data" not in raw
    with Image.open(clean.path) as image:
        assert image.format == "PNG"
        # the PNG ends exactly at its IEND chunk: nothing trails
        assert raw.endswith(b"IEND\xaeB`\x82")


def test_symlink_that_leaves_the_photo_folder_is_rejected(world):
    samples, cache, _ = world
    if "escape.png" not in samples:
        pytest.skip("no symlinks here")
    folder = samples["escape.png"].parent
    with pytest.raises(G.PhotoRejected, match="not a photo"):
        G.import_photo(samples["escape.png"], cache, allowed_root=folder)
    assert not cache.exists()


def test_jpeg_renamed_png_is_accepted_but_refused_in_strict_mode(world):
    # t21.png (the project's own test photo) is a JPEG with a .png name, so
    # the default only needs the content to be a photo; strict mode is the
    # "extension must agree" rule.
    samples, cache, _ = world
    with pytest.raises(G.PhotoRejected):
        G.import_photo(samples["jpeg_renamed.png"], cache, strict_extension=True)
    clean = G.import_photo(samples["jpeg_renamed.png"], cache)
    assert Image.open(clean.path).format == "PNG"


def test_a_file_that_is_a_photo_but_has_a_non_photo_extension_is_refused(world):
    samples, cache, _ = world
    with pytest.raises(G.PhotoRejected, match="not a photo"):
        G.import_photo(samples["wrong_extension.txt"], cache)


def test_file_over_the_size_limit_is_rejected(tmp_path, monkeypatch):
    path = tmp_path / "big.png"
    path.write_bytes(M.png_bytes())
    monkeypatch.setattr(G, "MAX_FILE_BYTES", 100)
    with pytest.raises(G.PhotoRejected) as info:
        G.import_photo(path, tmp_path / "cache")
    assert "too large" in info.value.reason


def test_normal_photo_passes_and_the_clean_copy_has_no_metadata(tmp_path):
    exif = Image.Exif()
    exif[0x0112] = 6                       # rotate 90 degrees
    exif[0x010E] = "secret description"
    source = tmp_path / "Hình ảnh" / "photo.JPG"
    source.parent.mkdir()
    Image.new("RGB", (80, 40), (10, 120, 200)).save(
        source, "JPEG", exif=exif, icc_profile=b"\x00" * 128)
    before = _files(tmp_path)

    clean = G.import_photo(source, tmp_path / "cache")

    assert isinstance(clean, G.CleanPhoto)
    assert (clean.width, clean.height) == (40, 80)            # EXIF turned into pixels
    assert clean.path.parent == tmp_path / "cache"
    assert clean.path.name == f"{clean.sha256}.png"
    assert _files(tmp_path) - before == {clean.path}           # only the clean copy
    with Image.open(clean.path) as image:
        assert image.format == "PNG" and image.mode == "RGB"
        assert "icc_profile" not in image.info and "exif" not in image.info
        assert not image.getexif()
        assert image.text == {} if hasattr(image, "text") else True
    raw = clean.path.read_bytes()
    assert b"secret description" not in raw and b"iCCP" not in raw
    again = G.import_photo(source, tmp_path / "cache")        # same hash, same file
    assert again == clean


def test_big_photo_is_capped_to_the_pipeline_size(tmp_path):
    path = tmp_path / "big.png"
    Image.new("RGB", (5823, 2700), (1, 2, 3)).save(path)
    clean = G.import_photo(path, tmp_path / "cache")
    assert (clean.width, clean.height) == (2048, 950)


def test_load_rgb_refuses_a_clean_copy_that_was_swapped(tmp_path):
    path = tmp_path / "p.png"
    path.write_bytes(M.png_bytes())
    clean = G.import_photo(path, tmp_path / "cache")
    assert G.load_rgb(clean).shape == (48, 64, 3)
    clean.path.write_bytes(M.png_bytes(colour=(1, 1, 1)))
    with pytest.raises(ValueError, match="hash"):
        G.load_rgb(clean)


def test_the_pipeline_accepts_only_a_clean_photo(tmp_path):
    from src.content.image_source import ImageSource

    path = tmp_path / "p.png"
    path.write_bytes(M.png_bytes())
    with pytest.raises(TypeError):
        ImageSource(path)
    with pytest.raises(TypeError):
        G.load_rgb(path)
    assert ImageSource(G.import_photo(path, tmp_path / "cache")).width == 64


def test_random_damage_never_crashes_the_gate(tmp_path):
    """Flip, cut and extend valid files at random: the answer is always a
    CleanPhoto or a PhotoRejected, never another exception."""
    rng = random.Random(1124)
    seeds = [("png", M.png_bytes(48, 32)), ("jpg", M.jpeg_bytes(48, 32)),
             ("png", M.animated_png()), ("webp", M.animated_webp())]
    buffer = io.BytesIO()
    Image.new("RGB", (48, 32), (9, 9, 9)).save(buffer, "WEBP")
    seeds.append(("webp", buffer.getvalue()))
    path_base = tmp_path / "fuzz"
    for index in range(300):
        ext, blob = seeds[index % len(seeds)]
        data = bytearray(blob)
        for _ in range(rng.randint(1, 6)):
            data[rng.randrange(len(data))] = rng.randrange(256)
        if rng.random() < 0.3:
            del data[rng.randrange(len(data)):]
        if rng.random() < 0.2:
            data += bytes(rng.randrange(256) for _ in range(rng.randint(1, 40)))
        path = path_base.with_suffix(f".{ext}")
        path.write_bytes(bytes(data))
        try:
            clean = G.import_photo(path, tmp_path / "cache")
        except G.PhotoRejected:
            continue
        assert G.load_rgb(clean).shape[2] == 3


def test_all_three_formats_pass(tmp_path):
    out = tmp_path / "cache"
    for name, data in (("a.png", M.png_bytes()), ("b.jpg", M.jpeg_bytes()),
                       ("c.jpeg", M.jpeg_bytes(progressive=True)),
                       ("d.webp", _webp())):
        path = tmp_path / name
        path.write_bytes(data)
        assert G.import_photo(path, out).width == 64


def _webp() -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (64, 48), (50, 60, 70)).save(buffer, "WEBP")
    return buffer.getvalue()


def test_gate_matches_the_old_decoder_pixel_for_pixel(tmp_path):
    """The gate must not change a single pixel of a normal photo (the
    automatch run depends on it): compare with ``cv2.imdecode``."""
    import cv2

    rng = np.random.default_rng(3)
    pixels = rng.integers(0, 256, (60, 90, 3), dtype=np.uint8)
    for name, kw in (("n.png", {}), ("n.jpg", {"quality": 90})):
        path = tmp_path / name
        Image.fromarray(pixels).save(path, **kw)
        old = cv2.cvtColor(cv2.imdecode(np.fromfile(str(path), np.uint8),
                                        cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
        new = G.load_rgb(G.import_photo(path, tmp_path / "cache"))
        assert np.array_equal(old, new), name


def test_a_rejected_photo_shows_its_error_on_the_album_screen(tmp_path):
    """The renderer toast is not visible in v3, so ``_try_load_media`` also
    puts the message on the album's own line (logic-only: no window here)."""
    from types import SimpleNamespace

    from src.engine.app import HcpdApplication
    from src.ui_album import v3_flow as F

    bad = tmp_path / "zip.png"
    bad.write_bytes(M.png_bytes() + M.zip_bytes())
    toasts = []
    app = HcpdApplication.__new__(HcpdApplication)
    flow = F.V3Flow()
    app._album = SimpleNamespace(flow=F, state=flow)
    app._album_options = SimpleNamespace(is_v3=True, enabled=True)
    app._renderer = SimpleNamespace(
        show_toast=lambda text, is_error=False: toasts.append(text),
        load_content_file=lambda src, label=None: (False, G.USER_MESSAGE))
    app._try_load_media(bad, 0.0)
    assert toasts == [G.USER_MESSAGE]
    assert flow.message == G.USER_MESSAGE
