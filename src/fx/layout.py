"""
Design-space layout helpers for placing fx relative to UI text — pure
Python, no GL. TASK_game_feel_1118 P3: the WIN screen's ``star_five``
anchor used to be the exact CENTRE of the ``score`` text rect (an old
copy-paste from the score text's own layout), so its "pon" word always
landed right on top of the score digits. These helpers let the chosen
anchor be checked by a rect-intersection test instead of eyeballing a
screenshot.
"""

from __future__ import annotations

import math

from src.fx.spec import FxSpec


def point_in_rect(x: float, y: float, rect) -> bool:
    rx, ry, rw, rh = rect
    return rx <= x <= rx + rw and ry <= y <= ry + rh


def rects_intersect(a, b) -> bool:
    """``True`` if two ``(x, y, w, h)`` rects overlap — touching edges
    (zero-area overlap) do NOT count."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and bx < ax + aw and ay < by + bh and by < ay + ah


def word_cue_aabb(spec: FxSpec, effect_name: str, cue_index: int,
                  anchor_x: float, anchor_y: float):
    """Axis-aligned bounding box, in design px, of a word CUE's rendered
    quad at its SETTLED (fully grown, ``WordAnim``'s spring at rest)
    scale — ``(x, y, w, h)``, matching every other rect in this codebase.

    Rotated by the cue's own ``rot`` first (``player.py``'s
    ``_word_command`` convention: positive angle reads clockwise in
    design's y-down space) — even the small angles real cues use shift
    the AABB a little, so this computes the rotated corners properly
    rather than assuming the box stays axis-aligned.
    """
    effect = spec.effects[effect_name]
    cue = (effect.get("words") or ())[cue_index]
    word_info = spec.words[cue["word"]]
    png_w = word_info["traced_w"] + 2 * word_info["pad"]
    png_h = word_info["traced_h"] + 2 * word_info["pad"]
    k0 = cue["size"] / word_info["traced_w"]
    half_w, half_h = (png_w * k0) / 2.0, (png_h * k0) / 2.0
    cx, cy = anchor_x + cue["dx"], anchor_y + cue["dy"]
    rot = cue.get("rot", 0.0)
    cos_a, sin_a = math.cos(rot), math.sin(rot)
    xs, ys = [], []
    for lx, ly in ((-half_w, -half_h), (half_w, -half_h),
                  (half_w, half_h), (-half_w, half_h)):
        xs.append(cx + lx * cos_a - ly * sin_a)
        ys.append(cy + lx * sin_a + ly * cos_a)
    x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    return (x0, y0, x1 - x0, y1 - y0)
