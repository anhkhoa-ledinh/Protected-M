"""
TASK_game_feel_1118 Phase 2: ``bleed_transparent`` (pure numpy/cv2, no GL)
and ``upload_rgba``'s filter parameters (mocked GL, no real context needed
— mirrors the mocked-GL pattern already used for other texture tests).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

import src.ui_album.textures as textures_module
from src.ui_album.textures import bleed_transparent


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
