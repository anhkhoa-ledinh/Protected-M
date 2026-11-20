"""
TASK_game_feel_1118 Phase 2: ``bleed_transparent`` (pure numpy/cv2, no GL)
and ``upload_rgba``'s filter parameters (mocked GL, no real context needed
— mirrors the mocked-GL pattern already used for other texture tests).

TASK_perf_textures_1119 Fix 2: same semantics, adapted for the fast
band-limited ``bleed_transparent_band`` (runtime fallback for PNGs
``tools/prebleed.py`` hasn't pre-bled), plus its ≤ 20 ms/megapixel budget
and ``load_rgba``'s skip-when-marked-or-opaque behaviour.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

import src.ui_album.textures as textures_module
from src.ui_album.textures import bleed_transparent, bleed_transparent_band


def test_bleed_alpha_is_unchanged():
    rng = np.random.default_rng(0)
    alpha = rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
    array = np.zeros((12, 12, 4), dtype=np.uint8)
    array[..., 3] = alpha
    array[..., :3] = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
    out = bleed_transparent(array)
    assert np.array_equal(out[..., 3], alpha)


def test_bleed_no_op_on_fully_opaque_image():
    rng = np.random.default_rng(1)
    array = np.empty((10, 10, 4), dtype=np.uint8)
    array[..., :3] = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    array[..., 3] = 255
    out = bleed_transparent(array)
    assert np.array_equal(out, array)


def test_bleed_transparent_pixel_next_to_red_becomes_reddish_not_black():
    array = np.zeros((9, 9, 4), dtype=np.uint8)   # all transparent, RGB=0
    array[4, 4] = [255, 0, 0, 255]                # one opaque red pixel
    out = bleed_transparent(array)
    neighbour = out[4, 5]                         # was [0, 0, 0, 0]
    assert neighbour[3] == 0                       # alpha still transparent
    assert int(neighbour[0]) > int(neighbour[1])
    assert int(neighbour[0]) > int(neighbour[2])
    assert int(neighbour[0]) > 0                   # not black anymore


def test_bleed_non_rgba_array_is_a_no_op():
    rgb = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = bleed_transparent(rgb)
    assert out is rgb


# ---- bleed_transparent_band (Fix 2: fast runtime fallback) ----------------

def test_band_alpha_is_unchanged():
    rng = np.random.default_rng(0)
    alpha = rng.integers(0, 256, size=(12, 12), dtype=np.uint8)
    array = np.zeros((12, 12, 4), dtype=np.uint8)
    array[..., 3] = alpha
    array[..., :3] = rng.integers(0, 256, size=(12, 12, 3), dtype=np.uint8)
    out = bleed_transparent_band(array)
    assert np.array_equal(out[..., 3], alpha)


def test_band_no_op_on_fully_opaque_image():
    rng = np.random.default_rng(1)
    array = np.empty((10, 10, 4), dtype=np.uint8)
    array[..., :3] = rng.integers(0, 256, size=(10, 10, 3), dtype=np.uint8)
    array[..., 3] = 255
    out = bleed_transparent_band(array)
    assert np.array_equal(out, array)


def test_band_no_op_on_fully_transparent_image():
    array = np.zeros((10, 10, 4), dtype=np.uint8)
    out = bleed_transparent_band(array)
    assert np.array_equal(out, array)          # nothing opaque to bleed FROM


def test_band_transparent_pixel_next_to_red_becomes_reddish_not_black():
    array = np.zeros((9, 9, 4), dtype=np.uint8)   # all transparent, RGB=0
    array[4, 4] = [255, 0, 0, 255]                # one opaque red pixel
    out = bleed_transparent_band(array)
    neighbour = out[4, 5]                         # was [0, 0, 0, 0], inside the band
    assert neighbour[3] == 0                       # alpha still transparent
    assert int(neighbour[0]) > int(neighbour[1])
    assert int(neighbour[0]) > int(neighbour[2])
    assert int(neighbour[0]) > 0                   # not black anymore


def test_band_pixel_far_from_any_opaque_pixel_gets_the_flat_average():
    array = np.zeros((40, 40, 4), dtype=np.uint8)
    array[0, 0] = [200, 0, 0, 255]
    array[0, 1] = [0, 0, 200, 255]                 # two opaque pixels, avg (100, 0, 100)
    out = bleed_transparent_band(array)
    far = out[30, 30]                              # far outside the 4px band
    assert far[3] == 0
    assert tuple(int(c) for c in far[:3]) == (100, 0, 100)


def test_band_non_rgba_array_is_a_no_op():
    rgb = np.full((4, 4, 3), 10, dtype=np.uint8)
    out = bleed_transparent_band(rgb)
    assert out is rgb


def _save_png(path, array, *, rgb_bled=False):
    from PIL import Image
    from PIL.PngImagePlugin import PngInfo

    info = None
    if rgb_bled:
        info = PngInfo()
        info.add_text("rgb_bled", "1")
    Image.fromarray(array, "RGBA").save(path, pnginfo=info)


def test_load_rgba_skips_bleed_when_marked_rgb_bled(tmp_path, monkeypatch):
    array = np.zeros((9, 9, 4), dtype=np.uint8)
    array[4, 4] = [255, 0, 0, 255]
    path = tmp_path / "marked.png"
    _save_png(path, array, rgb_bled=True)

    calls = []
    monkeypatch.setattr(textures_module, "bleed_transparent_band",
                        lambda a, *x, **k: (calls.append(1), a)[1])
    textures_module.load_rgba(path, max_width=0)
    assert calls == []


def test_load_rgba_skips_bleed_when_fully_opaque(tmp_path, monkeypatch):
    array = np.full((9, 9, 4), 255, dtype=np.uint8)
    path = tmp_path / "opaque.png"
    _save_png(path, array, rgb_bled=False)

    calls = []
    monkeypatch.setattr(textures_module, "bleed_transparent_band",
                        lambda a, *x, **k: (calls.append(1), a)[1])
    textures_module.load_rgba(path, max_width=0)
    assert calls == []


def test_load_rgba_bleeds_unmarked_file_with_transparent_pixels(tmp_path, monkeypatch):
    array = np.zeros((9, 9, 4), dtype=np.uint8)
    array[4, 4] = [255, 0, 0, 255]
    path = tmp_path / "unmarked.png"
    _save_png(path, array, rgb_bled=False)

    calls = []
    monkeypatch.setattr(textures_module, "bleed_transparent_band",
                        lambda a, *x, **k: (calls.append(1), a)[1])
    textures_module.load_rgba(path, max_width=0)
    assert calls == [1]


def test_band_meets_the_20ms_per_megapixel_budget():
    """TASK_perf_textures_1119 Fix 2: measured on THIS machine, not tuned.

    Shape matches the real worst case from results/perf_1119/profile_before.txt
    (a 1920x1080 ``top.png`` ~99% transparent, with a few opaque UI shapes —
    buttons/icons/text — as scattered BLOBS, not per-pixel noise. Per-pixel
    random opaque noise would put almost every pixel within the band of
    another opaque pixel, which no real UI art does."""
    rng = np.random.default_rng(2)
    h, w = 1080, 1920                              # ~2 MP, the worst-case UI top layer
    array = np.zeros((h, w, 4), dtype=np.uint8)
    array[..., :3] = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    array[..., 3] = 0
    for _ in range(40):                            # ~40 button/icon/text blobs
        y0 = rng.integers(0, h - 60)
        x0 = rng.integers(0, w - 200)
        array[y0:y0 + 40, x0:x0 + 160, 3] = 255

    bleed_transparent_band(array)                  # warm up (first-call cv2 cost)
    t0 = time.perf_counter()
    bleed_transparent_band(array)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    megapixels = (h * w) / 1_000_000.0
    per_mp = elapsed_ms / megapixels
    assert per_mp <= 20.0, f"{per_mp:.1f} ms/MP > 20 ms/MP budget ({elapsed_ms:.1f} ms total)"


def _mock_gl_module():
    gl = MagicMock()
    # Distinct sentinel values so assertions can tell which constant a
    # given glTexParameteri call used.
    for name in ("GL_TEXTURE_2D", "GL_TEXTURE_MIN_FILTER", "GL_TEXTURE_MAG_FILTER",
                "GL_TEXTURE_WRAP_S", "GL_TEXTURE_WRAP_T", "GL_TEXTURE_MAX_LEVEL",
                "GL_LINEAR", "GL_LINEAR_MIPMAP_LINEAR", "GL_CLAMP_TO_EDGE",
                "GL_RGBA", "GL_UNSIGNED_BYTE", "GL_UNPACK_ALIGNMENT"):
        setattr(gl, name, name)
    gl.glGenTextures.return_value = 7
    return gl


@pytest.fixture
def mocked_gl(monkeypatch):
    # ``monkeypatch.setattr`` only touches ``textures_module``'s OWN
    # module-level ``GL`` name and auto-reverts after the test — unlike
    # swapping ``sys.modules["OpenGL.GL"]`` and reload()-ing, which can
    # leave OTHER already-imported modules' ``GL`` references (and this
    # module's own, if the revert runs in the wrong order) pointing at a
    # disposed mock for the rest of the process.
    gl = _mock_gl_module()
    monkeypatch.setattr(textures_module, "GL", gl)
    yield gl, textures_module


def test_upload_rgba_default_uses_mipmap_filter_and_generates_mipmap(mocked_gl):
    gl, textures_module = mocked_gl
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    textures_module.upload_rgba(array)
    min_filter_calls = [c for c in gl.glTexParameteri.call_args_list
                        if c.args[1] == "GL_TEXTURE_MIN_FILTER"]
    assert len(min_filter_calls) == 1
    assert min_filter_calls[0].args[2] == "GL_LINEAR_MIPMAP_LINEAR"
    gl.glGenerateMipmap.assert_called_once_with("GL_TEXTURE_2D")


def test_upload_rgba_mipmaps_false_uses_plain_linear_and_skips_generate(mocked_gl):
    gl, textures_module = mocked_gl
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    textures_module.upload_rgba(array, mipmaps=False)
    min_filter_calls = [c for c in gl.glTexParameteri.call_args_list
                        if c.args[1] == "GL_TEXTURE_MIN_FILTER"]
    assert min_filter_calls[0].args[2] == "GL_LINEAR"
    gl.glGenerateMipmap.assert_not_called()


def test_upload_rgba_max_level_sets_the_gl_parameter(mocked_gl):
    gl, textures_module = mocked_gl
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    textures_module.upload_rgba(array, max_level=1)
    max_level_calls = [c for c in gl.glTexParameteri.call_args_list
                       if c.args[1] == "GL_TEXTURE_MAX_LEVEL"]
    assert len(max_level_calls) == 1
    assert max_level_calls[0].args[2] == 1


def test_upload_rgba_no_max_level_by_default(mocked_gl):
    gl, textures_module = mocked_gl
    array = np.zeros((4, 4, 4), dtype=np.uint8)
    textures_module.upload_rgba(array)
    max_level_calls = [c for c in gl.glTexParameteri.call_args_list
                       if c.args[1] == "GL_TEXTURE_MAX_LEVEL"]
    assert max_level_calls == []
