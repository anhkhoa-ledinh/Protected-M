"""Part of ``V3UI``, moved out of ``v3_ui.py`` unchanged (B5): see ``ClickGeometryMixin``."""

from __future__ import annotations

from src.ui_album import v3_flow as F
from src.ui_album import ids as ID


class ClickGeometryMixin:
    # ---- TASK_small_fixes_1120: caret -------------------------------------
    def _sync_caret_after_click(self, name: str, pre_focus: str | None,
                                from_pointer: bool) -> None:
        """A click can move the caret two different ways: clicking the
        ALREADY-focused field repositions the caret under the mouse;
        clicking anything else that happens to leave a field focused
        (``link_forgot`` prefilling+focusing ``field_forgot_user``, Tab
        is handled separately in ``focus_next``) puts the caret at that
        field's END, same as a fresh focus always has."""
        focused = self.state.forgot_focus if self.state.forgot_open else self.state.focus
        if focused is None:
            return
        geometry = self._field_caret_geometry(focused)
        if geometry is None:
            return
        rect, run, text = geometry
        if from_pointer and name == focused:
            was_focused = (pre_focus == focused)
            old_caret = self.state.caret_index if was_focused else None
            self.state.caret_index = self._caret_index_for_click(rect, run, text,
                                                                 old_caret)
        else:
            self.state.caret_index = len(text)

    def _field_caret_geometry(self, name: str):
        """``(value_rect, run, display_text)`` for ``name``'s caret math —
        the SAME rect/run/text ``v3.py`` draws it with, so click
        positioning and rendering never disagree. ``None`` for anything
        that isn't one of the caret-bearing fields (``field_join`` keeps
        its own append/backspace-last model, out of this task's scope)."""
        from src.ui_album.v3_screen_draw import _FIELD_VALUE_FONT, FORGOT_FIELDS
        from src.ui_album.v3_text_field import _field_box_rect, forgot_field_value_rect

        if name in F.FORGOT_FIELD_ATTR:
            for fname, _label, field_y, masked in FORGOT_FIELDS:
                if fname != name:
                    continue
                raw = F.field_value(self.state, name)
                value = "•" * len(raw) if masked else raw
                return forgot_field_value_rect(field_y), _FIELD_VALUE_FONT, value
            return None
        key = {ID.FIELD_USER: "user", ID.FIELD_PASS: "pass",
              ID.FIELD_PASS2: "pass2"}.get(name)
        if key is None:
            return None
        spec = self.manifest.text(self.state.screen, key)
        if spec is None:
            return None
        raw = F.field_value(self.state, name)
        masked = name in (ID.FIELD_PASS, ID.FIELD_PASS2) and not self.state.show_password
        value = "•" * len(raw) if masked else raw
        rect = _field_box_rect(self.manifest, self.state.screen, key,
                              [float(v) for v in spec["rect"]])
        return rect, spec["runs"][0], value

    def _caret_index_for_click(self, rect, run: dict, text: str,
                               old_caret: int | None) -> int:
        """Nearest character boundary (0..len(text)) to ``self.mouse``'s
        x inside ``rect`` — accounting for the SAME scroll the field was
        drawn with a moment ago (``old_caret`` is this field's caret
        index from the frame BEFORE this click, or ``None`` if it wasn't
        the focused field yet, meaning it was drawn unscrolled)."""
        offsets = self.painter.caret_offsets(run, text)
        total = offsets[-1] if offsets else 0.0
        box_w = float(rect[2])
        if old_caret is None:
            scroll = 0.0
        else:
            caret_clamped = max(0, min(len(text), old_caret))
            scroll = max(0.0, offsets[caret_clamped] - box_w)
            scroll = min(scroll, max(0.0, total - box_w))
        click_x = float(self.mouse[0]) - float(rect[0]) + scroll
        best_i, best_d = 0, abs(offsets[0] - click_x)
        for index, offset in enumerate(offsets):
            dist = abs(offset - click_x)
            if dist < best_d:
                best_d, best_i = dist, index
        return best_i

    def _click_rect(self, screen: str, name: str, in_dialog: bool):
        if in_dialog:
            from src.ui_album.v3_screen_draw import (
                CONFIRM_NO_RECT, CONFIRM_YES_RECT,
                FORGOT_CANCEL_RECT, FORGOT_FIELD_H,
                FORGOT_FIELD_W, FORGOT_FIELD_X,
                FORGOT_FIELDS, FORGOT_SUBMIT_RECT,
                RECOVERY_COPY_RECT, RECOVERY_SAVED_RECT)

            rects = {ID.BTN_CONFIRM_YES: CONFIRM_YES_RECT, ID.BTN_CONFIRM_NO: CONFIRM_NO_RECT,
                    ID.BTN_RECOVERY_COPY: RECOVERY_COPY_RECT,
                    ID.BTN_RECOVERY_SAVED: RECOVERY_SAVED_RECT,
                    ID.BTN_FORGOT_SUBMIT: FORGOT_SUBMIT_RECT,
                    ID.BTN_FORGOT_CANCEL: FORGOT_CANCEL_RECT}
            for field_name, _label, field_y, _masked in FORGOT_FIELDS:
                rects[field_name] = (FORGOT_FIELD_X, field_y, FORGOT_FIELD_W, FORGOT_FIELD_H)
            return rects.get(name)
        from src.ui_album.v3_hit import hotspot_rect

        for hotspot in self.manifest.screens[screen].get("hotspots", ()):
            if hotspot["n"] == name:
                return hotspot_rect(screen, hotspot)
        return None

    def _slot_rect(self, screen: str, index: int):
        name = f"slot_empty_{index + 1}"
        for hotspot in self.manifest.screens[screen].get("hotspots", ()):
            if hotspot["n"] == name:
                return hotspot["rect"]
        return None

    def _dialog_hover(self, x: float, y: float) -> str | None:
        """Hộp xác nhận chia sẻ che cả màn — hai nút của nó, không gì khác."""
        from src.ui_album.v3_screen_draw import CONFIRM_NO_RECT, CONFIRM_YES_RECT

        for name, rect in ((ID.BTN_CONFIRM_YES, CONFIRM_YES_RECT),
                          (ID.BTN_CONFIRM_NO, CONFIRM_NO_RECT)):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return name
        return None

    def _recovery_hover(self, x: float, y: float) -> str | None:
        """TASK_accounts_1119 Phase 6: the recovery-code dialog's own two
        buttons, same "covers the whole screen, nothing else" rule as
        the share-confirm dialog."""
        from src.ui_album.v3_screen_draw import RECOVERY_COPY_RECT, RECOVERY_SAVED_RECT

        for name, rect in ((ID.BTN_RECOVERY_COPY, RECOVERY_COPY_RECT),
                          (ID.BTN_RECOVERY_SAVED, RECOVERY_SAVED_RECT)):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return name
        return None

    def _forgot_hover(self, x: float, y: float) -> str | None:
        """The forgot-password dialog's four fields + Reset/Cancel."""
        from src.ui_album.v3_screen_draw import (FORGOT_CANCEL_RECT, FORGOT_FIELD_H,
                                     FORGOT_FIELD_W, FORGOT_FIELD_X,
                                     FORGOT_FIELDS, FORGOT_SUBMIT_RECT)

        for name, _label, field_y, _masked in FORGOT_FIELDS:
            if (FORGOT_FIELD_X <= x <= FORGOT_FIELD_X + FORGOT_FIELD_W
                    and field_y <= y <= field_y + FORGOT_FIELD_H):
                return name
        for name, rect in ((ID.BTN_FORGOT_SUBMIT, FORGOT_SUBMIT_RECT),
                          (ID.BTN_FORGOT_CANCEL, FORGOT_CANCEL_RECT)):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return name
        return None
