"""
TASK_game_feel_fix_1118 Fix 11: anti-alias every PIL ``ImageDraw``-built
shape in ``src/ui_album/v3.py`` — the user's own 18:24 report, a jagged
focus ring around the Password field. ``ImageDraw`` has no antialiasing
of its own; the fix is ``_aa_build``: draw 4x bigger, then
``Image.reduce(4)`` (box filter) back down to the final size.

Pure tests against ``_aa_build`` itself (no GL) cover the task's three
named checks: intermediate alpha on a curved edge, total alpha within
2% of the analytic area, and a pixel-aligned straight edge staying a
single-row transition. Real-GL tests confirm every shape method
(``rounded_frame``, ``rounded``, ``disc``, ``pip``, ``pill``) actually
routes through it, by intercepting the array just before GL upload.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import ImageDraw

from src.ui_album import v3
from src.ui_album.v3 import _AA_FACTOR, _aa_build

ROOT = Path(".").resolve()


# ---- _aa_build: the shared AA path, pure -----------------------------------

def _disc_draw(pen, w, h) -> None:
    pen.ellipse([0, 0, w - 1, h - 1], fill=(255, 255, 255, 255))


def test_curved_edge_has_intermediate_alpha_values():
    """A circle's boundary necessarily crosses pixel cells partially —
    a true AA path must produce alpha values strictly between 0 and
    255, not just a hard 0/255 staircase."""
    array = _aa_build(21, 21, _disc_draw)
    alpha = array[..., 3]
    intermediate = alpha[(alpha > 0) & (alpha < 255)]
    assert intermediate.size > 0


def test_disc_total_alpha_within_2_percent_of_analytic_area():
    size = 21
    array = _aa_build(size, size, _disc_draw)
    alpha = array[..., 3].astype(np.float64)
    total = alpha.sum() / 255.0
    analytic = math.pi * (size / 2.0) ** 2
    assert total == pytest.approx(analytic, rel=0.02)


@pytest.mark.parametrize("size", [11, 32, 47])
def test_disc_total_alpha_within_2_percent_for_several_sizes(size):
    array = _aa_build(size, size, _disc_draw)
    alpha = array[..., 3].astype(np.float64)
    total = alpha.sum() / 255.0
    analytic = math.pi * (size / 2.0) ** 2
    assert total == pytest.approx(analytic, rel=0.02)


def test_pixel_aligned_straight_edge_has_at_most_one_transition_row():
    """A horizontal edge that lands exactly on a final-pixel boundary
    (i.e. at a multiple of ``_AA_FACTOR`` in the supersampled canvas)
    must come out crisp — every big-pixel in a given output row is
    either fully inside or fully outside the filled rect, so the box
    filter introduces no blur at all beyond what's unavoidable."""
    width, height = 20, 20
    top_row, bottom_row = 8, 16   # final-pixel rows, both inside [0, height)

    def draw(pen, big_w, big_h):
        pen.rectangle([0, top_row * _AA_FACTOR, big_w - 1,
                       bottom_row * _AA_FACTOR - 1], fill=(255, 255, 255, 255))

    array = _aa_build(width, height, draw)
    col = array[:, 5, 3].astype(np.int32)
    transition_rows = np.sum((col > 0) & (col < 255))
    assert transition_rows <= 1
    # And, since the edges land exactly on the grid, it's actually crisp:
    assert transition_rows == 0
    assert col[top_row - 1] == 0
    assert col[top_row] == 255
    assert col[bottom_row - 1] == 255
    assert col[bottom_row] == 0


def test_aa_build_output_size_matches_final_size_not_the_supersampled_one():
    array = _aa_build(13, 27, _disc_draw)
    assert array.shape == (27, 13, 4)


def test_aa_build_draw_receives_the_supersampled_dimensions():
    seen = {}

    def draw(pen, big_w, big_h):
        seen["w"], seen["h"] = big_w, big_h

    _aa_build(10, 6, draw)
    assert seen == {"w": 10 * _AA_FACTOR, "h": 6 * _AA_FACTOR}


# ---- real painter methods route through _aa_build (real GL) --------------

@pytest.fixture
def painter(gl_context):
    manifest = v3.V3Manifest(ROOT)
    instance = v3.V3Painter(manifest, ROOT / "assets" / "shaders")
    instance.initialize()
    instance.set_frame((1280, 720))
    yield instance
    instance.release()


def _capture_upload(monkeypatch):
    """Intercept the array right before it reaches the GPU — lets a
    real-GL test inspect the exact pixels ``_aa_build`` produced for a
    real painter method, without needing to read a texture back."""
    captured: list = []
    real_upload = v3.upload_rgba

    def fake_upload(array, **kwargs):
        captured.append(array.copy())
        return real_upload(array, **kwargs)

    monkeypatch.setattr(v3, "upload_rgba", fake_upload)
    return captured


def test_disc_method_produces_antialiased_alpha(painter, monkeypatch):
    captured = _capture_upload(monkeypatch)
    painter.disc((100.0, 100.0, 40.0, 40.0), (1.0, 1.0, 1.0, 1.0))
    assert captured
    alpha = captured[0][..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_rounded_frame_produces_antialiased_alpha(painter, monkeypatch):
    captured = _capture_upload(monkeypatch)
    painter.rounded_frame((50.0, 50.0, 120.0, 40.0), (1.0, 1.0, 1.0, 1.0), 3.0, 10.0)
    assert captured
    alpha = captured[0][..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_rounded_cap_produces_antialiased_alpha(painter, monkeypatch):
    captured = _capture_upload(monkeypatch)
    painter.rounded((50.0, 50.0, 200.0, 24.0), 12.0, (1.0, 1.0, 1.0, 1.0))
    assert captured
    alpha = captured[0][..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_pip_ellipse_produces_antialiased_alpha(painter, monkeypatch):
    captured = _capture_upload(monkeypatch)
    style = {"fill": "#FFFFFF", "fill_op": 1.0}
    painter.pip((50.0, 50.0, 20.0, 20.0), "ellipse", 0.0, style)
    assert captured
    alpha = captured[0][..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_pip_rounded_rect_produces_antialiased_alpha(painter, monkeypatch):
    captured = _capture_upload(monkeypatch)
    style = {"fill": "#FFFFFF", "fill_op": 1.0}
    painter.pip((50.0, 50.0, 24.0, 16.0), "rect", 6.0, style)
    assert captured
    alpha = captured[0][..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_pill_background_produces_antialiased_alpha(painter, monkeypatch):
    captured = _capture_upload(monkeypatch)
    record = {"rect": (50.0, 50.0, 0.0, 28.0), "radius": 14.0, "pad": 12.0,
             "font": {"ff": "Nunito", "fw": 800, "fs": 13.0}}
    style = {"bg": "#FFFFFFEB", "fg": "#24415F"}
    painter.pill(record, "9/12", style)
    assert captured
    alpha = captured[0][..., 3]
    assert np.any((alpha > 0) & (alpha < 255))


def test_pill_cache_key_is_unaffected_by_aa_factor(painter):
    """Fix 11's own requirement: the cache key stays at the FINAL pixel
    size — calling the same pill twice must be a single build, not one
    per call (``_PilCache.builds`` only increments on an actual miss)."""
    record = {"rect": (50.0, 50.0, 0.0, 28.0), "radius": 14.0, "pad": 12.0,
             "font": {"ff": "Nunito", "fw": 800, "fs": 13.0}}
    style = {"bg": "#FFFFFFEB", "fg": "#24415F"}
    before = painter.pil.builds
    painter.pill(record, "9/12", style)
    after_first = painter.pil.builds
    painter.pill(record, "9/12", style)
    after_second = painter.pil.builds
    assert after_first == before + 1
    assert after_second == after_first
