"""
TASK_game_feel_fix_1118 Fix 2: photo layer textures skip
``bleed_transparent`` — their RGB under alpha=0 is already valid graded-
photo colour (``extract_layer_rgba``'s own docstring/implementation,
confirmed: full-frame graded RGB copied at every pixel, never zeroed;
``_pad_layer``'s border is ``cv2.BORDER_REPLICATE``, not zero-fill), so
the 6-iteration box blur was pure cost on the main render thread for no
visual effect. The layer-SHADOW texture is a different, small,
procedurally built array and keeps bleeding (the default).

``ContentPlaneRenderer.__new__`` (bypassing ``__init__``) + a real GL
context — ``_create_rgba_texture`` only touches its own argument and GL
global state, same pattern ``test_abandon_match.py`` uses for
``HcpdApplication``.
"""

from __future__ import annotations

import numpy as np

from src.renderer.content_plane import ContentPlaneRenderer


def _renderer() -> ContentPlaneRenderer:
    return ContentPlaneRenderer.__new__(ContentPlaneRenderer)


def test_apply_bleed_false_does_not_call_bleed_transparent(gl_context, monkeypatch):
    calls = []
    import src.renderer.content_plane as module

    monkeypatch.setattr(module, "bleed_transparent",
                        lambda arr: calls.append(arr) or arr)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    renderer = _renderer()
    renderer._create_rgba_texture(rgba, apply_bleed=False)
    assert calls == []


def test_apply_bleed_true_default_still_calls_bleed_transparent(gl_context, monkeypatch):
    calls = []
    import src.renderer.content_plane as module

    monkeypatch.setattr(module, "bleed_transparent",
                        lambda arr: calls.append(arr) or arr)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 3] = 255
    renderer = _renderer()
    renderer._create_rgba_texture(rgba)
    assert len(calls) == 1


def test_skipping_bleed_still_produces_a_valid_texture(gl_context):
    from OpenGL import GL

    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    rgba[..., 0] = 200
    rgba[..., 3] = 255
    renderer = _renderer()
    texture = renderer._create_rgba_texture(rgba, apply_bleed=False)
    assert texture > 0
    assert bool(GL.glIsTexture(texture))
    GL.glDeleteTextures(1, [texture])
