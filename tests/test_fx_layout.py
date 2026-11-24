"""TASK_game_feel_1118 Phase 3: ``src/fx/layout.py`` rect-intersection
helpers, and the WIN screen's ``star_five`` anchor chosen with them."""

from __future__ import annotations

from src.fx.layout import point_in_rect, rects_intersect, word_cue_aabb
from src.fx.spec import load_spec
from src.ui_album.v3 import V3Manifest
from src.ui_album.v3_consts import _WIN_SCORE_XY


def test_rects_intersect_overlapping():
    assert rects_intersect((0, 0, 10, 10), (5, 5, 10, 10)) is True


def test_rects_intersect_disjoint():
    assert rects_intersect((0, 0, 10, 10), (20, 20, 10, 10)) is False


def test_rects_intersect_touching_edges_do_not_count():
    assert rects_intersect((0, 0, 10, 10), (10, 0, 10, 10)) is False


def test_point_in_rect():
    assert point_in_rect(5, 5, (0, 0, 10, 10)) is True
    assert point_in_rect(15, 5, (0, 0, 10, 10)) is False


def test_word_cue_aabb_centred_on_anchor_plus_dx_dy():
    spec = load_spec(".")
    bbox = word_cue_aabb(spec, "star_five", 0, 0.0, 0.0)
    cue = spec.effects["star_five"]["words"][0]
    word_info = spec.words[cue["word"]]
    expected_w = (word_info["traced_w"] + 2 * word_info["pad"]) * (
        cue["size"] / word_info["traced_w"])
    x, y, w, h = bbox
    # Small rotation (cue's own "rot") only nudges the box a little —
    # width/height stay close to the unrotated size.
    assert abs(w - expected_w) < expected_w * 0.1
    cx = x + w / 2.0
    assert abs(cx - cue["dx"]) < 5.0


def test_win_star_five_anchor_clears_score_and_row_value_text():
    """The real bug TASK_game_feel_1118 found: the OLD anchor was the
    exact centre of the ``score`` rect, so the "pon" word always sat on
    top of the score digits (visible in 1117's own ``fx_win.png``)."""
    manifest = V3Manifest(".")
    spec = load_spec(".")
    record = manifest.screens["v3_09a_result_win"]
    texts = {t["key"]: tuple(t["rect"]) for t in record["texts"]}
    score_rect = texts["score"]

    assert point_in_rect(_WIN_SCORE_XY[0], _WIN_SCORE_XY[1], score_rect) is False

    bbox = word_cue_aabb(spec, "star_five", 0, *_WIN_SCORE_XY)
    assert rects_intersect(bbox, score_rect) is False
    # Also clear of the result-row text directly above the score.
    assert rects_intersect(bbox, texts["row_value#0"]) is False
    assert rects_intersect(bbox, texts["row_label#0"]) is False
