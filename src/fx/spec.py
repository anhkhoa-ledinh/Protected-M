"""
Loads and validates ``assets/fx/fx.json`` — the data spec documented in
``assets/fx/README.md``. Pure Python, no GL: this only reads JSON and
checks PNG file presence/dimensions with PIL, never decodes pixels.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

#: ``assets/fx/`` relative to the project root.
FX_DIR = Path("assets") / "fx"


@dataclass(frozen=True)
class FxSpec:
    """Validated contents of ``fx.json``. ``root`` is ``assets/fx/`` —
    every path in ``effects``/``words`` is relative to it.

    ``sheet_sizes``: ``relpath -> (w, h)`` in sheet px, filled in while
    validating frame rects — kept here so ``player.py`` can compute uv
    rects without opening every sheet PNG a second time."""

    root: Path
    image_scale: float
    fps: int
    effects: dict
    words: dict
    sheet_sizes: dict

    def sheet_path(self, relpath: str) -> Path:
        return self.root / relpath

    def word_path(self, word: str, style: str) -> Path:
        return self.root / "words" / f"{word}_{style}.png"

    def word_rel_path(self, word: str, style: str) -> str:
        return f"words/{word}_{style}.png"


def load_spec(project_root) -> FxSpec:
    """
    Read and validate ``<project_root>/assets/fx/fx.json``.

    Raises ``FileNotFoundError`` if a sheet or word PNG the spec refers to
    is missing, and ``ValueError`` if a frame's ``rect`` falls outside its
    own sheet's real pixel dimensions (opened with PIL, not just trusted
    off the variant's own declared ``size`` field — catches a stale
    ``size`` too).
    """
    root = Path(project_root) / FX_DIR
    data = json.loads((root / "fx.json").read_text(encoding="utf-8"))
    image_scale = float(data["image_scale"])
    fps = int(data["fps"])
    effects = data["effects"]
    words = data["words"]

    _validate_words(root, words)
    sheet_sizes = _validate_effects(root, effects)

    return FxSpec(root=root, image_scale=image_scale, fps=fps,
                  effects=effects, words=words, sheet_sizes=sheet_sizes)


def _validate_words(root: Path, words: dict) -> None:
    for word, info in words.items():
        for style in info.get("styles", ()):
            path = root / "words" / f"{word}_{style}.png"
            if not path.exists():
                raise FileNotFoundError(
                    f"fx.json: missing word PNG for {word!r}/{style!r}: {path}")


def _validate_effects(root: Path, effects: dict) -> dict:
    sheet_sizes: dict[str, tuple[int, int]] = {}
    for name, effect in effects.items():
        for variant_index, variant in enumerate(effect.get("variants", ())):
            sheet_rel = variant["sheet"]
            sheet_path = root / sheet_rel
            if sheet_rel not in sheet_sizes:
                if not sheet_path.exists():
                    raise FileNotFoundError(
                        f"fx.json: missing sheet PNG for effect {name!r} "
                        f"variant {variant_index}: {sheet_path}")
                with Image.open(sheet_path) as image:
                    sheet_sizes[sheet_rel] = image.size
            sheet_w, sheet_h = sheet_sizes[sheet_rel]
            for frame_index, frame in enumerate(variant.get("frames", ())):
                if frame is None:
                    continue
                x, y, w, h = frame["rect"]
                if x < 0 or y < 0 or x + w > sheet_w or y + h > sheet_h:
                    raise ValueError(
                        f"fx.json: effect {name!r} variant {variant_index} "
                        f"frame {frame_index} rect {frame['rect']} is "
                        f"outside its sheet {sheet_rel} ({sheet_w}x{sheet_h})")
    return sheet_sizes
