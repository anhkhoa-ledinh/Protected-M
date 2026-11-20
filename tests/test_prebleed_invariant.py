"""TASK_perf_textures_1119 Fix 1: every PNG ``tools/prebleed.py`` marks
``rgb_bled=1`` must be invisible — alpha identical to what git HEAD has
committed, and RGB identical everywhere alpha > 0. Only RGB under fully
transparent pixels may have changed.
"""
from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from tools.prebleed import RGB_BLED_KEY, discover_targets

ROOT = Path(__file__).resolve().parents[1]


def _git_head_bytes(path: Path) -> bytes | None:
    rel = path.relative_to(ROOT).as_posix()
    result = subprocess.run(
        ["git", "show", f"HEAD:{rel}"], cwd=ROOT, capture_output=True, check=False)
    if result.returncode != 0:
        return None
    return result.stdout


def _marked_targets() -> list[Path]:
    marked = []
    for path in discover_targets():
        with Image.open(path) as im:
            if getattr(im, "text", {}).get(RGB_BLED_KEY) == "1":
                marked.append(path)
    return marked


@pytest.mark.parametrize("path", _marked_targets(), ids=lambda p: str(p.relative_to(ROOT)))
def test_prebleed_only_changes_rgb_under_alpha_zero(path: Path) -> None:
    head_bytes = _git_head_bytes(path)
    if head_bytes is None:
        pytest.skip(f"{path} not committed yet (nothing to diff against)")
    before = np.asarray(Image.open(io.BytesIO(head_bytes)).convert("RGBA"))
    after = np.asarray(Image.open(path).convert("RGBA"))
    assert before.shape == after.shape, "size changed"
    np.testing.assert_array_equal(before[..., 3], after[..., 3], err_msg="alpha changed")
    opaque = before[..., 3] > 0
    np.testing.assert_array_equal(
        before[..., :3][opaque], after[..., :3][opaque],
        err_msg="RGB changed where alpha > 0")
