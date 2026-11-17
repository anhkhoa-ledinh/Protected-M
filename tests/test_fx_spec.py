"""TASK_ui_sakuga_1117 Phase 7: fx.json spec loading/validation."""

import json

import pytest

from src.fx.spec import load_spec


def test_load_spec_real_fx_json():
    spec = load_spec(".")
    assert spec.image_scale > 0
    assert spec.fps > 0
    assert "hit_kill" in spec.effects
    assert spec.sheet_sizes  # filled in while validating frame rects


def test_missing_sheet_raises(tmp_path):
    root = tmp_path / "assets" / "fx"
    root.mkdir(parents=True)
    (root / "words").mkdir()
    (root / "fx.json").write_text(json.dumps({
        "image_scale": 1.5, "fps": 30, "words": {},
        "effects": {"boom": {"variants": [
            {"sheet": "sheets/missing.png", "frames": []}]}},
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_spec(tmp_path)


def test_missing_word_png_raises(tmp_path):
    root = tmp_path / "assets" / "fx"
    (root / "sheets").mkdir(parents=True)
    (root / "words").mkdir()
    with open(root / "sheets" / "a.png", "wb") as f:
        _write_blank_png(f, 4, 4)
    (root / "fx.json").write_text(json.dumps({
        "image_scale": 1.5, "fps": 30,
        "words": {"gogo": {"traced_w": 10.0, "styles": ["ink"]}},
        "effects": {},
    }), encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        load_spec(tmp_path)


def test_frame_rect_outside_sheet_raises(tmp_path):
    root = tmp_path / "assets" / "fx"
    (root / "sheets").mkdir(parents=True)
    (root / "words").mkdir()
    with open(root / "sheets" / "a.png", "wb") as f:
        _write_blank_png(f, 4, 4)
    (root / "fx.json").write_text(json.dumps({
        "image_scale": 1.5, "fps": 30, "words": {},
        "effects": {"boom": {"variants": [
            {"sheet": "sheets/a.png",
             "frames": [{"rect": [0, 0, 100, 100], "offset": [0, 0]}]}]}},
    }), encoding="utf-8")
    with pytest.raises(ValueError):
        load_spec(tmp_path)


def _write_blank_png(fileobj, w, h):
    from PIL import Image
    Image.new("RGBA", (w, h), (0, 0, 0, 0)).save(fileobj, format="PNG")
