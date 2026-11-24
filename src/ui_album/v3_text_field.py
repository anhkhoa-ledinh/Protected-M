"""
Text field editing: caret positioning/scroll and password masking for the
v3 UI's sign-in/up/forgot fields (TASK_arch_refactor_1120 R4: split out of
``src.ui_album.v3``, which had grown to 2479 lines).
"""

from __future__ import annotations

from OpenGL import GL

from src.ui_album.v3_hit import hotspot_rect
from src.ui_album.v3_painter import rgba
from src.ui_album import ids as ID

#: Con trỏ nhập của ô đang chọn (ghi trong ``screens[].notes``).
CARET_SIZE = (2.0, 20.0)
CARET_GAP = 2.0
CARET_COL = "#3B72B4"

#: Forgot-dialog field geometry (``src.ui_album.v3``'s own values,
#: duplicated here — same pattern as elsewhere in this split; only the
#: two numbers ``forgot_field_value_rect`` itself needs).
FORGOT_FIELD_X = 378.0
FORGOT_FIELD_W = 524.0


#: three sign-in/up fields whose caret must scroll within the FULL field
#: BOX, not the narrow rect Penpot exported around the box's SAMPLE value
#: (``user``'s sample "anhkhoa" measures 65px; the box itself is 356px).
_CARET_FIELD_HOTSPOT = {"user": ID.FIELD_USER, "pass": ID.FIELD_PASS,
                       "pass2": ID.FIELD_PASS2}


def _field_box_rect(manifest, screen: str, key: str, text_rect):
    """Widens ``text_rect`` (a ``texts`` entry's own rect, sized to its
    SAMPLE value) out to its field's real hotspot box, keeping the
    text's own left inset and y/height — the box is symmetric (left
    inset == right margin), matching the forgot dialog's own hand-placed
    16px margin and confirmed against the manifest's own numbers (both
    ``field_user`` and ``field_pass`` measure exactly 48px/260px)."""
    hotspot_name = _CARET_FIELD_HOTSPOT.get(key)
    if hotspot_name is None:
        return text_rect
    hotspot = next((h for h in manifest.screens[screen].get("hotspots", ())
                    if h["n"] == hotspot_name), None)
    if hotspot is None:
        return text_rect
    hx, hy, hw, hh = (float(v) for v in hotspot_rect(screen, hotspot))
    _ = hy, hh
    inset = float(text_rect[0]) - hx
    if inset <= 0.0:
        return text_rect
    width = (hx + hw) - inset - float(text_rect[0])
    return (text_rect[0], text_rect[1], max(1.0, width), text_rect[3])


def _draw_field_text(painter, rect, run: dict, text: str, caret_index,
                     alpha: float = 1.0) -> None:
    """Draws one field's text CLIPPED to ``rect`` and, if ``text`` is
    wider than ``rect``, scrolled so the caret stays inside the box —
    then draws the caret itself at ``caret_index`` (a character index;
    ``None`` draws plain, unfocused text, no scroll/cursor, clipped from
    the start — TASK_small_fixes_1120)."""
    offsets = painter.caret_offsets(run, text)
    total = offsets[-1] if offsets else 0.0
    box_w = float(rect[2])
    if caret_index is None:
        scroll = 0.0
    else:
        caret_index = max(0, min(len(text), int(caret_index)))
        scroll = max(0.0, offsets[caret_index] - box_w)
        scroll = min(scroll, max(0.0, total - box_w))
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glScissor(*painter.viewport_px(rect))
    shifted = (rect[0] - scroll, rect[1], rect[2] + scroll, rect[3])
    painter.line(shifted, [run], [text], "left", alpha)
    GL.glDisable(GL.GL_SCISSOR_TEST)
    if caret_index is not None:
        caret_x = rect[0] + offsets[caret_index] - scroll
        painter.fill((caret_x, rect[1] + (rect[3] - CARET_SIZE[1]) * 0.5,
                     CARET_SIZE[0], CARET_SIZE[1]), rgba(CARET_COL))




def _forgot_field_value(state, key: str, masked: bool) -> str:
    raw = str(state.texts.get(key) or "")
    return "•" * len(raw) if masked else raw


def forgot_field_value_rect(field_y: float):
    """The value text's own rect inside a forgot-dialog field box at
    ``field_y`` — shared by drawing and by ``V3UI``'s click-to-caret math
    (TASK_small_fixes_1120), so both measure from the exact same box."""
    return (FORGOT_FIELD_X + 16.0, field_y + 22.0, FORGOT_FIELD_W - 32.0, 24.0)
