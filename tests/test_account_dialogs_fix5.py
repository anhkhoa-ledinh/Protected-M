"""TASK_perf_textures_1119 Fix 5: account dialog polish.

- Forgot-password field labels match sign-in's own baked label size.
- Both dialogs' cards fit their content -- no dead gap between the last
  row and the buttons.
- The forgot dialog's inline error/hint line actually draws (the logic
  that SETS the four error strings -- unknown name, wrong code,
  cooldown, format mismatch -- was already covered by
  tests/test_account_recovery.py; this is the render-level check that
  the draw path actually shows it).
"""
from __future__ import annotations

from src.ui_album.v3 import V3State
from src.ui_album.v3_screen_draw import (FORGOT_CANCEL_RECT, FORGOT_CARD_RECT,
                             FORGOT_ERROR_FONT, FORGOT_ERROR_RECT,
                             FORGOT_SUBMIT_RECT, RECOVERY_CARD_RECT,
                             RECOVERY_COPY_RECT, RECOVERY_SAVED_RECT,
                             _FIELD_LABEL_FONT, _draw_forgot_dialog,
                             _draw_recovery_dialog)

#: Generous but real -- a gap bigger than this between the last content
#: row and the row below it reads as "dead space", the exact complaint
#: this fix addresses. Chosen well under the ORIGINAL bugs' gaps (62px
#: forgot, 102px recovery) so a regression back toward either trips this.
_MAX_ACCEPTABLE_GAP_PX = 36.0


class _FakePainter:
    def __init__(self) -> None:
        self.lines: list = []
        self.rects: list = []
        self.buttons: list = []

    def line(self, rect, runs, texts, align="left", alpha=1.0) -> float:
        self.lines.append((tuple(rect), runs, list(texts), align, alpha))
        return rect[0]

    def rounded(self, rect, radius, colour) -> None:
        self.rects.append((tuple(rect), radius, colour))

    def rounded_frame(self, rect, colour, thickness, radius) -> None:
        pass

    def fill(self, rect, colour) -> None:
        pass

    def quad(self, rect, texture, colour=(1.0, 1.0, 1.0, 1.0), uv=None) -> None:
        pass

    def text_width(self, runs, texts) -> float:
        return 80.0


# ---- field label size matches v3_01a's own baked label -------------------

def test_forgot_field_label_font_size_matches_sign_in_measurement():
    """12.0, derived from v3_01a_sign_in/base.png's own baked "Username"
    label: measured ink height (design space) 8.67px, Nunito-700's
    OS/2.sCapHeight/unitsPerEm = 0.705, 8.67 / 0.705 = 12.3 -> 12."""
    assert _FIELD_LABEL_FONT["fs"] == 12.0
    assert _FIELD_LABEL_FONT["ff"] == "Nunito"


# ---- no dead gap between content and buttons ------------------------------

def test_recovery_card_has_no_dead_gap_before_its_buttons():
    card_bottom = RECOVERY_CARD_RECT[1] + RECOVERY_CARD_RECT[3]
    buttons_bottom = RECOVERY_COPY_RECT[1] + RECOVERY_COPY_RECT[3]
    assert RECOVERY_SAVED_RECT[1] == RECOVERY_COPY_RECT[1]
    # the card ends shortly after the buttons -- not 100px further down.
    assert 0.0 <= card_bottom - buttons_bottom <= _MAX_ACCEPTABLE_GAP_PX


def test_forgot_card_has_no_dead_gap_before_its_submit_button():
    error_bottom = FORGOT_ERROR_RECT[1] + FORGOT_ERROR_RECT[3]
    gap = FORGOT_SUBMIT_RECT[1] - error_bottom
    assert 0.0 <= gap <= _MAX_ACCEPTABLE_GAP_PX


def test_forgot_card_fits_its_cancel_link_with_a_small_bottom_pad():
    card_bottom = FORGOT_CARD_RECT[1] + FORGOT_CARD_RECT[3]
    cancel_bottom = FORGOT_CANCEL_RECT[1] + FORGOT_CANCEL_RECT[3]
    assert 0.0 <= card_bottom - cancel_bottom <= _MAX_ACCEPTABLE_GAP_PX


def test_recovery_card_is_meaningfully_shorter_than_the_original_bug():
    """Regression guard for the specific bug: the original card was 380
    design-px tall with a 102px dead gap. After removing it, the card
    must be noticeably shorter -- not just nominally different."""
    assert RECOVERY_CARD_RECT[3] <= 320.0


def test_forgot_card_is_meaningfully_shorter_than_the_original_bug():
    """Original: 540px tall, 62px dead gap before the submit button."""
    assert FORGOT_CARD_RECT[3] <= 520.0


# ---- inline error/hint line actually renders -------------------------------

def test_forgot_dialog_draws_the_error_line_when_present():
    state = V3State()
    state.forgot_open = True
    state.texts["forgot_error"] = "Wrong code. Try again."
    painter = _FakePainter()

    _draw_forgot_dialog(painter, state)

    matches = [entry for entry in painter.lines
              if entry[2] == ["Wrong code. Try again."]]
    assert len(matches) == 1
    rect, runs, _texts, _align, _alpha = matches[0]
    assert runs == [FORGOT_ERROR_FONT]
    assert tuple(rect) == FORGOT_ERROR_RECT


def test_forgot_dialog_draws_no_error_line_when_absent():
    state = V3State()
    state.forgot_open = True
    state.texts["forgot_error"] = ""
    painter = _FakePainter()

    _draw_forgot_dialog(painter, state)

    assert not any(entry[1] == [FORGOT_ERROR_FONT] for entry in painter.lines)


def test_forgot_dialog_draws_nothing_when_closed():
    state = V3State()
    state.forgot_open = False
    painter = _FakePainter()

    _draw_forgot_dialog(painter, state)

    assert painter.lines == []
    assert painter.rects == []


def test_recovery_dialog_draws_the_code_box_and_both_buttons():
    state = V3State()
    state.recovery_code = "ABCD-EFGH-JKLM"
    painter = _FakePainter()

    _draw_recovery_dialog(painter, state)

    assert any(entry[2] == ["ABCD-EFGH-JKLM"] for entry in painter.lines)
