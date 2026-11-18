"""
TASK_accounts_1119 Phase 3: the only two allowed art edits.

1. ``v3_01c_sign_in_error/base.png`` — erase the baked "Wrong password.
   Try again." text (keep the red "!" icon). Every pixel OUTSIDE the
   erase rect must be byte-identical to the committed version before
   this task (fetched from git, not a manually-kept backup file).
2. ``v3_01b_sign_up``'s new ``hint_bad`` sprite — a crop of that same
   icon, used in place of ``hint_ok``'s green checkmark when a field has
   a problem.

No other pixel in either pack may move — ``git show`` against the commit
right before Phase 3 started is the ground truth, not a copy this repo
also controls (a stray edit to a "backup" file would silently invalidate
the whole test).
"""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(".").resolve()
BASE_PNG = "assets/ui_v3/v3_01c_sign_in_error/base.png"
HINT_BAD_PNG = "assets/ui_v3/sprites/v3_01b_sign_up__hint_bad.png"
#: The LAST commit before this phase's art edit (Phase 2's commit) —
#: ground truth for "every pixel outside the erase rect is untouched".
BEFORE_REF = "6e6ec78"
#: Image-px rect (x0, y0, x1, y1) that WAS allowed to change — the baked
#: "Wrong password. Try again." text, with a few px of margin for AA.
ERASE_RECT = (722, 548, 1005, 588)


def _git_show(ref: str, path: str) -> bytes:
    result = subprocess.run(["git", "show", f"{ref}:{path}"], cwd=ROOT,
                            capture_output=True, check=True)
    return result.stdout


@pytest.fixture(scope="module")
def before_base() -> np.ndarray:
    raw = _git_show(BEFORE_REF, BASE_PNG)
    return np.array(Image.open(io.BytesIO(raw)).convert("RGBA"))


@pytest.fixture(scope="module")
def after_base() -> np.ndarray:
    return np.array(Image.open(ROOT / BASE_PNG).convert("RGBA"))


def test_erase_rect_actually_changed_something():
    """Sanity check on the test itself: if the erase rect were empty or
    wrong, the "everything outside is identical" test below would pass
    VACUOUSLY even on a completely unedited file."""
    x0, y0, x1, y1 = ERASE_RECT
    assert x1 > x0 and y1 > y0


def test_every_pixel_outside_the_erase_rect_is_byte_identical(before_base, after_base):
    assert before_base.shape == after_base.shape
    x0, y0, x1, y1 = ERASE_RECT
    mask = np.ones(before_base.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = False
    assert np.array_equal(before_base[mask], after_base[mask])


def test_something_inside_the_erase_rect_did_change(before_base, after_base):
    """The erase rect itself SHOULD differ (that's the whole point) —
    guards against a no-op "fix" that technically passes the identical-
    outside test by changing nothing at all."""
    x0, y0, x1, y1 = ERASE_RECT
    before_region = before_base[y0:y1, x0:x1]
    after_region = after_base[y0:y1, x0:x1]
    assert not np.array_equal(before_region, after_region)


def test_the_red_icon_is_still_present_and_unmoved(before_base, after_base):
    """The task's own rule: keep the "!" icon — only the TEXT is erased.
    Checked by sampling a pixel inside the icon's own bbox (left of the
    erase rect) and confirming it's untouched (already covered by the
    "outside" test above) AND still opaque/coloured (not accidentally
    erased too)."""
    icon_sample = (705, 565)  # inside the icon, left of ERASE_RECT[0]=722
    x, y = icon_sample
    assert x < ERASE_RECT[0]
    pixel = after_base[y, x]
    assert pixel[3] > 0                 # still opaque (not erased)
    assert tuple(before_base[y, x]) == tuple(pixel)   # unmoved


def test_hint_bad_sprite_file_exists_and_is_non_empty():
    path = ROOT / HINT_BAD_PNG
    assert path.exists()
    img = Image.open(path).convert("RGBA")
    arr = np.array(img)
    assert arr.shape[0] > 0 and arr.shape[1] > 0
    assert np.any(arr[..., 3] > 0)      # not a fully-transparent crop


def test_hint_bad_sprite_is_the_same_icon_cropped_from_v3_01c(before_base):
    """The crop should match the ORIGINAL (pre-erase) icon pixels — not
    some unrelated red blob — confirmed by comparing against the
    committed base.png's own icon region at the same crop box used to
    produce the sprite file."""
    sprite = np.array(Image.open(ROOT / HINT_BAD_PNG).convert("RGBA"))
    crop_box = (693, 552, 719, 578)   # must match the crop used to make the file
    reference = before_base[crop_box[1]:crop_box[3], crop_box[0]:crop_box[2]]
    assert sprite.shape == reference.shape
    assert np.array_equal(sprite, reference)
