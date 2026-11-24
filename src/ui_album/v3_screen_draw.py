"""
Per-screen content drawing for the v3 UI: HUD cards/bars/pips/pills,
dialogs (share-confirm, recovery code, forgot-password), aim/scope/hit-fx
overlays (TASK_arch_refactor_1120 R4: split out of ``src.ui_album.v3``,
which had grown to 2479 lines — a 4th file beyond the 3 the task named,
needed to keep every file under the ~900-line gate given that growth).
"""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import TYPE_CHECKING

from OpenGL import GL
import numpy as np

from src.ui_album.hud_cards import (cards_for_screen, element_card_map,
                                    item_slot_card_index)
from src.ui_album.scope import (RETICLE_ARM_PX, RETICLE_COL, RETICLE_GAP_PX,
                                RETICLE_THICK_PX, VIGNETTE_ALPHA,
                                VIGNETTE_COL, VIGNETTE_RADIUS_PX,
                                VIGNETTE_SOFT_PX, ZOOM_SETTLED_EPS,
                                screen_scope_transform)
from src.ui_album.shot_feel import (CRIT_IMPACT_COL,
                                    HITMARK_ANGLES_DEG, HITMARK_COL,
                                    HITMARK_CRIT_COL, HITMARK_CRIT_SCALE,
                                    HITMARK_INNER_PX, HITMARK_LIFE_S,
                                    HITMARK_OUTER_PX, HITMARK_OUTLINE_COL,
                                    HITMARK_OUTLINE_PX, HITMARK_WIDTH_PX)
from src.ui_album.v3_hit import hotspot_rect
from src.ui_album.v3_painter import rgba
from src.ui_album.v3_text_field import (CARET_COL, _draw_field_text,
                                        _field_box_rect, _forgot_field_value,
                                        forgot_field_value_rect)
from src.ui_album import ids as ID

if TYPE_CHECKING:
    from src.ui_album.v3 import V3Manifest, V3Painter, V3State

#: Design-space canvas size + ink colour (duplicated — same pattern as
#: ``v3_hit.py``/``v3_painter.py``).
DESIGN_SIZE = (1280.0, 720.0)
INK = "#24415F"

#: Vòng hover của hộp xác nhận chia sẻ (``_draw_confirm_dialog`` — hộp
#: thoại không đi qua ``v3_hover.py``, chỉ hai nút Yes/No của nó).
HOVER_PX = 2.0

#: Vùng vẽ 3D và webcam ở **chế độ chụp** — README mục "Shot mode".
SHOT_VIEWPORT_RGB = (60, 70, 80)
SHOT_CAMERA_RGB = (90, 90, 100)

#: Hover/focus button corner radius (duplicated — also needed by
#: ``v3_hit.py``'s hotspot-shape table; same numeric value, same
#: duplication pattern as elsewhere in this split).
BTN_CORNER_RADIUS_PX = 12.0


AIM_ARM = 7.0
AIM_THICK = 2.0
AIM_GAP = 2.0
AIM_EDGE = 1.0
AIM_COL, AIM_EDGE_COL = "#FFFFFF", "#24415F"

#: Bắn xong thì tâm ngắm **bung ra** và mờ đi, rồi khép lại theo nhịp súng.
#:
#: Đo trên một phiên chơi thật: 54 phát ra, 54 cú bấm bị chặn vì chưa tới
#: nhịp. Người chơi bấm nhanh gấp đôi súng, mà mỗi cú bị chặn thì màn hình
#: im lặng hoàn toàn — không tiếng (bản demo không có âm), không hình. Nhịp
#: bắn vô hình thì cảm giác như app đớp mất cú bấm. Cho tâm ngắm khép dần
#: lại là nói được "súng đang hồi" mà không phải đổi một con số cân bằng nào.
AIM_RECOIL = 6.0
AIM_DIM = 0.45

#: Lỗ đạn trên ảnh. Người chơi đang bắn vào một tấm ảnh, nên dấu để lại là
#: một cái **lỗ**, không phải chấm đỏ báo lỗi.
#:
#: Ba lớp tròn lồng nhau: quầng ngoài nhạt (mảnh vụn), vành sẫm, lõi gần
#: đen. Kích thước CỐ ĐỊNH nên cả vệt mờ dần chỉ tốn ba texture — xem
#: ``V3Painter.disc``.
HOLE_LIFE_S = 3.0
HOLE_FADE_S = 0.8               # chỉ mờ đi ở quãng cuối
HOLE_CORE, HOLE_RING, HOLE_DUST = 5.0, 9.0, 15.0
HOLE_CORE_COL, HOLE_RING_COL, HOLE_DUST_COL = "#141C28", "#2E3A4C", "#8DA3BA"

#: Trúng bọ thì thêm một chớp xanh ngắn đè lên lỗ — để biết là đã diệt được.
HIT_FLASH_S = 0.45
HIT_FLASH_SIZE = 26.0
HIT_FLASH_COL = "#3AA17A"

#: TASK_weapons_1119: thanh máu nổi trên đầu một con vừa bị bắn mà chưa
#: chết — 28×4 px thiết kế, xanh (``HIT_FLASH_COL``) → đỏ (``_DANGER_COL``,
#: khai báo dưới) theo phần máu còn. Độ mờ do ``AlbumMatch.hp_bars`` tính
#: sẵn (mờ dần hết ``HP_BAR_FADE_S`` giây không bị bắn tiếp) — vẽ ở đây chỉ
#: đọc lại, không tính tuổi của riêng nó.
HP_BAR_W_PX, HP_BAR_H_PX = 28.0, 4.0
HP_BAR_GAP_PX = 16.0            # khoảng cách từ chỗ bắn lên đáy thanh
HP_BAR_BG_COL = "#141C28"

#: Dòng báo nổi ở giữa dưới (toast của giao diện).
TOAST_RECT = (340.0, 626.0, 600.0, 44.0)
TOAST_RADIUS = 22.0
TOAST_BG, TOAST_FG = "#24415F", "#FFFFFF"
TOAST_FONT = {"ff": "Nunito", "fw": 800, "fs": 15.0, "col": TOAST_FG}

#: TASK_game_feel_fix_1118 Fix 10: screens with a gameplay HUD — the
#: per-card fade machinery (proximity, scope, H-hide) only applies
#: here (menus/results draw their ``top.png`` the old, single-quad way).
HUD_CARD_SCREENS = frozenset({ID.PLAY, ID.PVP_HUNT,
                              ID.PVP_HIDE})

#: Code-drawn album buttons, with fixed rectangles matching the manifest
#: hotspots. Share sits on the memory caption row; Random in the drop card.
CHOOSE_PHOTO_SCREEN = ID.PHOTOS
SHARE_BTN_RECT = (944.0, 384.0, 228.0, 30.0)
RANDOM_BTN_RECT = (264.0, 478.0, 240.0, 52.0)
SHARE_BTN_FONT = {"ff": "Nunito", "fw": 800, "fs": 14.0, "ls": 0.0,
                  "col": "#3B72B4"}
#: Màu ``primary`` của manifest — hợp đồng dùng hex trực tiếp (xem ``CARET_COL``).
BUTTON_PRIMARY_COL = "#3B72B4"

#: Hộp xác nhận trước lượt chia sẻ đầu tiên của mỗi ảnh (P2.3) — chữ ĐÚNG
#: nguyên văn đề bài, không một ký tự khác.
CONFIRM_TEXT = ("This photo will be public: anyone with the app can "
               "download it. Do not share photos of people without their "
               "consent. Upload?")
CONFIRM_CARD_RECT = (380.0, 230.0, 520.0, 260.0)
CONFIRM_YES_RECT = (766.0, 422.0, 110.0, 44.0)
CONFIRM_NO_RECT = (640.0, 422.0, 110.0, 44.0)
CONFIRM_TEXT_FONT = {"ff": "Nunito", "fw": 700, "fs": 15.0, "ls": 0.0,
                     "col": INK}
CONFIRM_BTN_FONT = {"ff": "Nunito", "fw": 800, "fs": 15.0, "ls": 0.0}

# ---- TASK_accounts_1119 Phase 6: recovery code + forgot-password dialogs --
#: Same style as the share-album confirm dialog above — a dimmed
#: full-screen backdrop, a white rounded card, Nunito buttons. Field
#: input-box fill sampled from the real sign-in design (base.png);
#: field-label colour reuses the "muted" secondary-text tone used
#: everywhere else in this pack.
_DIALOG_DIM_COL = "#0B1520"
_DIALOG_DIM_ALPHA = 0.45
_FIELD_BG_COL = "#F1F6FC"
_FIELD_LABEL_COL = "#8DA3BA"
_FIELD_VALUE_FONT = {"ff": "Nunito", "fw": 800, "fs": 16.0, "ls": 0.0, "col": INK}
#: TASK_perf_textures_1119 Fix 5: measured off v3_01a_sign_in/base.png's
#: own baked "Username"/"Password" labels — cropped the baked "U" glyph
#: (design-space ink height ~8.67px at image_scale 1.5) and solved
#: ``size = ink_height / (Nunito-700's OS/2.sCapHeight / unitsPerEm)``
#: = 8.67 / 0.705 ≈ 12.3, rounds to 12. The dialog's own labels were
#: fs=11 — one step smaller than the screen they overlay, not a huge
#: gap but a real, measurable one, and visible once side by side.
_FIELD_LABEL_FONT = {"ff": "Nunito", "fw": 700, "fs": 12.0, "ls": 0.0,
                     "col": _FIELD_LABEL_COL}
_DANGER_COL = "#E8676B"

#: TASK_perf_textures_1119 Fix 5: both dialogs below used to leave a
#: dead gap between their last content row and their buttons (recovery:
#: 102px between the code box and Copy/I-saved-it; forgot: 62px between
#: the error line and Reset password) — the card rect's own height was
#: just set too tall for what the content actually needs. Every y/height
#: below is now derived from the row above it plus one of these two
#: gaps, not an independent hand-picked number, so the two can't drift
#: apart again if a row's own height ever changes.
_DIALOG_ROW_GAP = 20.0
_DIALOG_BOTTOM_PAD = 24.0

RECOVERY_CARD_X, RECOVERY_CARD_Y, RECOVERY_CARD_W = 380.0, 170.0, 520.0
RECOVERY_TITLE_FONT = {"ff": "Baloo 2", "fw": 800, "fs": 22.0, "ls": 0.0, "col": INK}
RECOVERY_BODY_FONT = {"ff": "Nunito", "fw": 700, "fs": 14.0, "ls": 0.0, "col": INK}
RECOVERY_CODE_FONT = {"ff": "Nunito", "fw": 900, "fs": 26.0, "ls": 2.0,
                      "col": BUTTON_PRIMARY_COL}
#: Fixed-at-2-lines layout (the body string is a constant; verified at
#: this card width) — ``_draw_recovery_dialog`` still wraps it for real
#: and uses ``len(lines)`` for the code box, so a future body edit that
#: changes the line count shifts the code box correctly; only the
#: BUTTONS/card-height below stay keyed to this known-good line count.
_RECOVERY_BODY_LINES = 2
_recovery_code_y = (RECOVERY_CARD_Y + 28.0 + 44.0
                    + _RECOVERY_BODY_LINES * 20.0 + _DIALOG_ROW_GAP)
_recovery_code_bottom = _recovery_code_y + 56.0
RECOVERY_COPY_RECT = (420.0, _recovery_code_bottom + _DIALOG_ROW_GAP, 200.0, 48.0)
RECOVERY_SAVED_RECT = (660.0, _recovery_code_bottom + _DIALOG_ROW_GAP, 200.0, 48.0)
RECOVERY_CARD_RECT = (RECOVERY_CARD_X, RECOVERY_CARD_Y, RECOVERY_CARD_W,
                      RECOVERY_COPY_RECT[1] + RECOVERY_COPY_RECT[3]
                      + _DIALOG_BOTTOM_PAD - RECOVERY_CARD_Y)

FORGOT_CARD_X, FORGOT_CARD_Y, FORGOT_CARD_W = 330.0, 90.0, 620.0
FORGOT_TITLE_FONT = {"ff": "Baloo 2", "fw": 800, "fs": 22.0, "ls": 0.0, "col": INK}
FORGOT_FIELD_H = 56.0
FORGOT_FIELD_X = 378.0
FORGOT_FIELD_W = 524.0
#: ``(hotspot name, label, y, is_password)``, top to bottom.
FORGOT_FIELDS = (
    (ID.FIELD_FORGOT_USER, "Username", 198.0, False),
    (ID.FIELD_FORGOT_CODE, "Recovery code", 262.0, False),
    (ID.FIELD_FORGOT_PASS, "New password", 326.0, True),
    (ID.FIELD_FORGOT_PASS2, "Confirm new password", 390.0, True),
)
FORGOT_ERROR_RECT = (378.0, 458.0, 524.0, 20.0)
_forgot_error_bottom = FORGOT_ERROR_RECT[1] + FORGOT_ERROR_RECT[3]
FORGOT_SUBMIT_RECT = (378.0, _forgot_error_bottom + _DIALOG_ROW_GAP, 524.0, 48.0)
FORGOT_CANCEL_RECT = (378.0, FORGOT_SUBMIT_RECT[1] + FORGOT_SUBMIT_RECT[3] + 16.0,
                      524.0, 20.0)
FORGOT_CARD_RECT = (FORGOT_CARD_X, FORGOT_CARD_Y, FORGOT_CARD_W,
                    FORGOT_CANCEL_RECT[1] + FORGOT_CANCEL_RECT[3]
                    + _DIALOG_BOTTOM_PAD - FORGOT_CARD_Y)
FORGOT_ERROR_FONT = {"ff": "Nunito", "fw": 700, "fs": 13.0, "ls": 0.0,
                     "col": _DANGER_COL}
FORGOT_CANCEL_FONT = {"ff": "Nunito", "fw": 700, "fs": 13.0, "ls": 0.0,
                      "col": BUTTON_PRIMARY_COL}



def _texts_for(record: dict, state: V3State) -> list[str]:
    """Chuỗi của từng run sau khi thay giá trị sống."""
    runs = record["runs"]
    value = state.texts.get(record["key"])
    if value is None:
        return [str(r.get("c") or "") for r in runs]
    if isinstance(value, (list, tuple)):
        parts = list(value)
        # Nhiều giá trị hơn số run thì DỒN phần thừa vào run cuối. Cắt bỏ
        # chúng là mất chữ mà không báo gì: ô "LAYER 2/4" chỉ có một run, đưa
        # vào ``["2", "/4"]`` thì màn chơi hiện mỗi "2".
        if len(parts) > len(runs) and runs:
            parts = parts[:len(runs) - 1] + ["".join(
                "" if p is None else str(p) for p in parts[len(runs) - 1:])]
        return [str(r.get("c") or "") if v is None else str(v)
                for r, v in zip(runs, parts + [None] * len(runs))]
    # Một giá trị cho nhiều run: run đầu là con số sống, run sau giữ nguyên
    # ("7" + "/12" — README mục Draw order 8).
    return [str(value)] + [str(r.get("c") or "") for r in runs[1:]]


def _runs_for(record: dict, state: V3State) -> list[dict]:
    """Run sau khi thay màu sống (``key`` hoặc từng run)."""
    runs = record["runs"]
    colour = state.colours.get(record["key"])
    if colour is None:
        return runs
    values = colour if isinstance(colour, (list, tuple)) else [colour]
    out = []
    for index, run in enumerate(runs):
        new = values[index] if index < len(values) else None
        out.append({**run, "col": new} if new else run)
    return out


def _wrap_lines(painter: V3Painter, runs, texts, width: float) -> list[list[str]]:
    """Cắt dòng theo bề rộng ``rect.w`` (chỉ dùng cho text có ``wrap``)."""
    words = (texts[0] or "").split()
    if not words:
        return [texts]
    lines, current = [], ""
    for word in words:
        probe = f"{current} {word}".strip()
        if current and painter.text_width(runs[:1], [probe]) > width:
            lines.append(current)
            current = word
        else:
            current = probe
    lines.append(current)
    return [[line] + texts[1:] for line in lines]



def _draw_photos(painter, manifest, record, state) -> None:
    for photo in record.get("photos", ()):
        kind, rect = photo.get("kind"), photo["rect"]
        if kind == "photo":
            source = state.photo
            if source is not None and Path(source).exists():
                painter.cover(rect, source, photo.get("clip"))
            elif state.shot:
                painter.cover(rect, manifest.sample_photo(), photo.get("clip"))
        elif kind == "camera":
            if state.shot:
                painter.fill(rect, rgba("#%02X%02X%02X" % SHOT_CAMERA_RGB))
            elif state.camera_tex is not None:
                _cover_texture(painter, rect, state.camera_tex, mirror=True)
        elif kind == "viewport":
            if state.shot:
                painter.fill(rect, rgba("#%02X%02X%02X" % SHOT_VIEWPORT_RGB))
            elif state.viewport_draw is not None:
                # TASK_game_feel_1118 P4: screen shake moves the 3D scene
                # viewport (this rect) and the fx layer only — never the
                # HUD, so it stays readable while the scene shakes.
                dx, dy = state.scene_shake
                shaken = (rect[0] + dx, rect[1] + dy, rect[2], rect[3]) \
                    if (dx or dy) else rect
                state.viewport_draw(shaken)


def _cover_texture(painter, rect, entry, mirror: bool = False) -> None:
    """Texture cover-fit vào ``rect`` (khung webcam, lật ngang nếu cần)."""
    texture, tex_w, tex_h = entry
    factor = max(rect[2] / tex_w, rect[3] / tex_h)
    width, height = tex_w * factor, tex_h * factor
    uv = (1.0, 0.0, 0.0, 1.0) if mirror else (0.0, 0.0, 1.0, 1.0)
    GL.glEnable(GL.GL_SCISSOR_TEST)
    GL.glScissor(*painter.viewport_px(rect))
    painter.quad((rect[0] + (rect[2] - width) * 0.5,
                  rect[1] + (rect[3] - height) * 0.5, width, height),
                 texture, uv=uv)
    GL.glDisable(GL.GL_SCISSOR_TEST)


#: TASK_hud_fix_1119 Fix 2: these two sprite keys stay drawn at full
#: alpha even while H has hidden the HUD — everything else (every card,
#: its dynamic contents, and the "free" item-slot/head-tracking/effect
#: elements that have no card of their own) fades with
#: ``state.hud_hidden_alpha``.
_ALWAYS_VISIBLE_SPRITE_KEYS = frozenset({"crosshair", "bite"})


def _card_fade(card_map: dict, state: V3State, key: str) -> tuple[int | None, float]:
    """``(card index or None, current alpha)`` for ``key``.

    ``index=None`` ("free" — no card under this element) used to always
    mean alpha 1.0 (proximity/scope fade never touched it). Now a free
    element ALSO fades when the HUD is hidden (TASK_hud_fix_1119 Fix 2)
    — except the couple of sprite keys in ``_ALWAYS_VISIBLE_SPRITE_KEYS``,
    which stay visible either way.
    """
    index = card_map.get(key)
    base = 1.0 if index is None else float(state.card_alpha.get(index, 1.0))
    if key in _ALWAYS_VISIBLE_SPRITE_KEYS:
        return index, base
    return index, base * float(state.hud_hidden_alpha)


def _slot_card_alpha(manifest, screen: str, state: V3State) -> float:
    """Current alpha of the item-slot card (1.0 on a screen without one)."""
    index = item_slot_card_index(cards_for_screen(manifest, screen))
    return 1.0 if index is None else float(state.card_alpha.get(index, 1.0))


def _draw_hud_top_cards(painter, manifest, screen, record, state) -> None:
    """TASK_game_feel_fix_1118 Fix 10.1: draw ``top.png`` per CARD, each
    at its own current alpha, instead of one single full-canvas quad —
    this is what lets one card fade out while the rest of the HUD stays
    put.

    TASK_hud_fix_1119 Fix 2: this is the card BACKGROUND art itself, a
    separate draw from the dynamic bars/pips/pills/sprites/texts drawn
    over it (those go through ``_card_fade`` below) — missing this one
    left every card's opaque panel on screen while H had faded its
    CONTENTS to nothing (confirmed from a real ``hud_hidden.png``: Esc/
    Pause and every card panel still fully opaque). ``hud_hidden_alpha``
    applies to every card here, fadeable or not — unlike proximity/scope
    (which only ever targets ``card.fades`` cards), H hides the WHOLE
    HUD, Esc/Pause included.
    """
    top = record.get("top")
    for index, card in enumerate(cards_for_screen(manifest, screen)):
        if not card.art:
            continue
        base = float(state.card_alpha.get(index, 1.0)) if card.fades else 1.0
        alpha = base * float(state.hud_hidden_alpha)
        painter.layer_region(card.rect, top, alpha)


def _joined_text(value) -> str:
    if isinstance(value, (list, tuple)):
        return "".join(str(v) for v in value)
    return "" if value is None else str(value)


def _draw_covers(painter, state) -> None:
    """Flat shapes that hide sample text baked into the art."""
    for cover in state.covers:
        kind, rect = cover[0], cover[1]
        if kind == "rect":
            painter.fill(rect, rgba(cover[2]))
        elif kind == "rounded":
            painter.rounded(rect, cover[2], rgba(cover[3]))
        elif kind == "patch":
            image = cover[3]
            entry = painter.pil.get(cover[2], lambda: (image, None))
            if entry is not None:
                painter.quad(rect, entry[0])


def _draw_bars(painter, record, state, card_map) -> None:
    for bar in record.get("bars", ()):
        key = bar["key"]
        if key in state.hidden:
            continue
        _card_index, fade = _card_fade(card_map, state, key)
        x, y, w, h = (float(v) for v in bar["rect"])
        radius = float(bar.get("radius", 0.0))
        fraction = state.bars.get(key)
        if fraction is None:
            fraction = float(bar.get("sample_frac", 0.0))
        fraction = max(0.0, min(1.0, float(fraction)))
        if bar.get("rest_col"):
            painter.rounded((x, y, w, h), radius, rgba(bar["rest_col"], fade))
        # Không bao giờ hẹp hơn 2×radius (README): một viên bo góc bị bóp
        # dưới mức đó thành vệt nhòn nhọn, không còn là "thanh đầy 0 %".
        painter.rounded((x, y, max(w * fraction, 2.0 * radius), h), radius,
                        rgba(bar["col"], fade))


def _pip_flags(spec: dict, state: V3State) -> list[bool]:
    """Cờ bật/tắt từng pip: giá trị sống, nếu không thì ``sample_on``."""
    total = len(spec["rects"])
    value = state.pips.get(spec["key"])
    if value is None:
        flags = list(spec.get("sample_on", ()))
        return [bool(flags[i]) if i < len(flags) else False for i in range(total)]
    if isinstance(value, (list, tuple)):
        return [bool(value[i]) if i < len(value) else False for i in range(total)]
    return [i < int(value) for i in range(total)]


def _draw_pips(painter, record, state, card_map) -> None:
    for spec in record.get("pips", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        _card_index, fade = _card_fade(card_map, state, key)
        flags = _pip_flags(spec, state)
        limit = state.pips.get(f"{key}:limit")
        for index, rect in enumerate(spec["rects"]):
            if limit is not None and index >= int(limit):
                continue          # pip vượt quá N thì ẩn hẳn (HUD lớp/đạn)
            style = spec["on"] if flags[index] else spec["off"]
            painter.pip(rect, spec.get("shape", "rect"),
                        float(spec.get("radius", 0.0)), style, fade)


def _draw_pills(painter, record, state, card_map) -> None:
    for spec in record.get("pills", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        _card_index, fade = _card_fade(card_map, state, key)
        text, style_name = state.pills.get(key, (None, None))
        if text is None:
            text = spec.get("sample") or ""
        styles = spec["styles"]
        style = styles.get(style_name) if style_name else None
        if style is None:
            style = next(iter(styles.values()))
        painter.pill(spec, str(text), style, fade)


def _draw_sprites(painter, manifest, screen, record, state, card_map) -> None:
    for sprite in record.get("sprites", ()):
        key = sprite["key"]
        if key in state.hidden or key not in state.sprites:
            continue
        _card_index, fade = _card_fade(card_map, state, key)
        where = state.sprites[key]
        rect = sprite["rect"] if where is True else where
        painter.sprite(rect, sprite.get("file"),
                       float(state.alpha.get(key, 1.0)) * fade, key in state.mirror)
    # TASK_hud_fix_1119 Fix 2: borrowed sprites (item-slot icons/badges,
    # a non-Precision weapon's icon lent from the loadout screen) have
    # no card of their own, so they fade with ``hud_hidden_alpha``
    # directly instead of through ``_card_fade`` — none of them are in
    # the always-visible whitelist (that's only crosshair/bite, neither
    # ever borrowed).
    slot_fade = _slot_card_alpha(manifest, screen, state)
    for label, (other, key, rect) in state.borrowed.items():
        borrowed = manifest.sprite(other, key)
        if borrowed is not None:
            painter.sprite(rect, borrowed.get("file"),
                           float(state.alpha.get(label, 1.0))
                           * float(state.hud_hidden_alpha)
                           * (slot_fade if label in state.slot_labels else 1.0),
                           label in state.mirror)
    _ = screen


#: TASK_signin_toggles_1120: ``switch_remember`` has no OFF art — the
#: design mock (``base.png``) only ever drew it ON (green track, knob
#: right). Painting ON needs nothing extra; OFF is painted procedurally
#: over the baked ON look, grey track + knob pushed left, same colour as
#: the field placeholder text so it reads as "disabled" without a new
#: asset.
_SWITCH_OFF_TRACK_COL = _FIELD_LABEL_COL
_SWITCH_KNOB_COL = "#FFFFFF"
_SWITCH_KNOB_PAD = 2.0


def _draw_remember_switch(painter, screen, record, state) -> None:
    if state.remember:
        return
    hotspot = next((h for h in record.get("hotspots", ())
                    if h["n"] == ID.SWITCH_REMEMBER), None)
    if hotspot is None:
        return
    x, y, w, h = (float(v) for v in hotspot_rect(screen, hotspot))
    painter.rounded((x, y, w, h), h * 0.5, rgba(_SWITCH_OFF_TRACK_COL))
    knob_d = h - 2.0 * _SWITCH_KNOB_PAD
    painter.disc((x + _SWITCH_KNOB_PAD, y + _SWITCH_KNOB_PAD, knob_d, knob_d),
                rgba(_SWITCH_KNOB_COL))


def _draw_texts(painter, manifest, screen, record, state, card_map) -> None:
    for spec in record.get("texts", ()):
        key = spec["key"]
        if key in state.hidden:
            continue
        _card_index, fade = _card_fade(card_map, state, key)
        runs = _runs_for(spec, state)
        texts = _texts_for(spec, state)
        alpha = float(state.alpha.get(key, 1.0)) * fade
        rect = [float(v) for v in state.text_rect.get(key, spec["rect"])]
        if spec.get("wrap") and state.wrap:
            line_h = float(spec.get("line_h") or rect[3])
            lines = _wrap_lines(painter, runs, texts, rect[2])
            for index, parts in enumerate(lines):
                painter.line((rect[0], rect[1] + index * line_h, rect[2], line_h),
                             runs, parts, spec.get("al", "left"), alpha)
            continue
        if (state.caret == key and len(runs) == 1
                and spec.get("al", "left") == "left"):
            box = _field_box_rect(manifest, screen, key, rect)
            _draw_field_text(painter, box, runs[0], texts[0],
                             state.caret_index, alpha)
        else:
            painter.line(rect, runs, texts, spec.get("al", "left"), alpha)
    # TASK_hud_fix_1119 Fix 2: item-slot key/name/count text (no card of
    # its own, same reasoning as the borrowed-sprite loop above).
    for rect, runs, texts, align in state.extra:
        painter.line(rect, runs, texts, align, float(state.hud_hidden_alpha))
    slot_alpha = (float(state.hud_hidden_alpha)
                  * _slot_card_alpha(manifest, screen, state))
    for rect, runs, texts, align in state.slot_extra:
        painter.line(rect, runs, texts, align, slot_alpha)


def _hole(painter, x: float, y: float, size: float, colour) -> None:
    """Một lớp tròn của lỗ đạn, canh giữa ``(x, y)``."""
    painter.disc((x - size * 0.5, y - size * 0.5, size, size), colour)


def _scope_adjust(x: float, y: float, state) -> tuple[float, float, bool]:
    """TASK_game_feel_fix_1118 Fix 5: a scene-anchored overlay point
    (stored once, un-zoomed, at spawn time — the glass point ``g``) ->
    its CURRENT screen position while scoped, ``c_now + z_now·(g -
    c_now)``, or unchanged when not scoped. Third element: ``False`` if
    the (possibly moved) point now falls outside the scope's own vignette
    circle — "hide them outside the scope circle" — the caller should
    skip drawing in that case."""
    if state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS or state.aim is None:
        return x, y, True
    sx, sy = screen_scope_transform((x, y), state.aim, state.scope_zoom)
    if math.hypot(sx - state.aim[0], sy - state.aim[1]) > VIGNETTE_RADIUS_PX:
        return sx, sy, False
    return sx, sy, True


def _draw_shots(painter, state) -> None:
    """
    Lỗ đạn ở chỗ vừa bắn, cộng một chớp xanh khi trúng bọ.

    Giữ nguyên lỗ gần như suốt tuổi thọ rồi mới mờ hẳn ở quãng cuối: lỗ đạn
    trên ảnh thì không nhạt dần ngay, và người chơi cần nhìn lại được mình
    vừa bắn chệch về phía nào. Giữ vài phát gần nhất chứ không chỉ phát cuối.

    TASK_game_feel_fix_1118 Fix 5: each mark's stored ``(x, y)`` is the
    glass point it was fired at (unzoomed design px, by construction —
    see ``game_bridge.fire``'s own comment) — ``_scope_adjust`` re-derives
    where that point sits on screen THIS frame, scoped or not.
    """
    if state.shot:
        return
    for x, y, age, hit in state.shots:
        age = float(age)
        if age >= HOLE_LIFE_S:
            continue
        x, y, visible = _scope_adjust(x, y, state)
        if not visible:
            continue
        # Mờ dần CHỈ ở quãng cuối, trước đó giữ nguyên độ đậm.
        left = min(1.0, max(0.0, (HOLE_LIFE_S - age) / HOLE_FADE_S))
        _hole(painter, x, y, HOLE_DUST, rgba(HOLE_DUST_COL, 0.22 * left))
        _hole(painter, x, y, HOLE_RING, rgba(HOLE_RING_COL, 0.72 * left))
        _hole(painter, x, y, HOLE_CORE, rgba(HOLE_CORE_COL, 0.92 * left))
        if hit and age < HIT_FLASH_S and not state.fx_on:
            # Sakuga's own hit_kill/hit_crit sprite replaces this flash
            # when fx is on (TASK_ui_sakuga_1117) — off, it's unchanged.
            flash = 1.0 - age / HIT_FLASH_S
            _hole(painter, x, y, HIT_FLASH_SIZE * (1.0 - flash * 0.5),
                  rgba(HIT_FLASH_COL, 0.55 * flash))


def _hp_bar_colour(frac: float, alpha: float) -> tuple[float, float, float, float]:
    """Xanh (đầy) → đỏ (gần hết), nội suy tuyến tính theo ``frac`` (0–1)."""
    full, empty = rgba(HIT_FLASH_COL), rgba(_DANGER_COL)
    t = max(0.0, min(1.0, float(frac)))
    rgb = tuple(e + (f - e) * t for f, e in zip(full[:3], empty[:3]))
    return (*rgb, alpha)


def _draw_hp_bars(painter, state) -> None:
    """TASK_weapons_1119: thanh máu nổi trên đầu một con vừa bị bắn mà
    chưa chết — cùng lối "điểm cố định" với vết đạn/hit marker
    (``_scope_adjust``): gắn ở CHỖ PHÁT BẮN gần nhất găm vào nó, không
    truy lại vị trí chiếu 3D hiện tại của con bọ."""
    if state.shot:
        return
    for x, y, frac, alpha in state.hp_bars:
        x, y, visible = _scope_adjust(x, y, state)
        if not visible:
            continue
        left, top = x - HP_BAR_W_PX * 0.5, y - HP_BAR_GAP_PX
        painter.rounded((left, top, HP_BAR_W_PX, HP_BAR_H_PX),
                        HP_BAR_H_PX * 0.5, rgba(HP_BAR_BG_COL, 0.7 * alpha))
        fill_w = max(HP_BAR_H_PX, HP_BAR_W_PX * max(0.0, min(1.0, frac)))
        painter.rounded((left, top, fill_w, HP_BAR_H_PX), HP_BAR_H_PX * 0.5,
                        _hp_bar_colour(frac, alpha))


def _draw_toast(painter, state, screen=None) -> None:
    """
    Dòng báo nổi ở giữa dưới.

    Thiếu nó thì mọi lời ``v3_flow.say`` — "Code copied", "Night theme is not
    in v2 yet", "Reloading" — được đặt rồi **không ai vẽ**, nên người dùng bấm
    nút mà tưởng như không có gì xảy ra. Toast của renderer không thay được:
    nó vẽ trong pass cảnh 3D, mà màn không có ô viewport thì pass ấy bị cắt
    còn 1 pixel.
    """
    if not state.message or state.shot:
        return
    x, y, w, h = TOAST_RECT
    if screen == CHOOSE_PHOTO_SCREEN:
        y = 574.0
    text = str(state.message)
    width = min(w, painter.text_width([TOAST_FONT], [text]) + 2 * TOAST_RADIUS)
    left = x + (w - width) * 0.5
    painter.rounded((left, y, width, h), TOAST_RADIUS, rgba(TOAST_BG, 0.92))
    painter.line((left, y, width, h), [TOAST_FONT], [text], "center")


def _draw_choose_photo_extra(painter, screen, state) -> None:
    """Nút "Share to album" / "Random from album" — xem ``SHARE_BTN_RECT``."""
    if state.shot or screen != CHOOSE_PHOTO_SCREEN:
        return
    for rect, label_key, alpha_key, default in (
            (SHARE_BTN_RECT, "share_to_album_label", "share_to_album_btn",
             "Share to album"),
            (RANDOM_BTN_RECT, "random_from_album_label", "random_from_album_btn",
             "Random from album")):
        label = state.texts.get(label_key, default)
        alpha = float(state.alpha.get(alpha_key, 1.0))
        painter.rounded(rect, BTN_CORNER_RADIUS_PX, rgba("#FFFFFF"))
        painter.rounded_frame(rect, rgba(BUTTON_PRIMARY_COL, alpha), 1.5,
                              BTN_CORNER_RADIUS_PX)
        font = dict(SHARE_BTN_FONT)
        if rect == SHARE_BTN_RECT:
            font["fs"] = 12.0
        # The disabled label must fit even in the small caption-row button.
        width = painter.text_width([font], [str(label)])
        if width > rect[2] - 16:
            font["fs"] *= (rect[2] - 16) / width
        painter.line(rect, [font], [str(label)], "center", alpha)


def _draw_confirm_dialog(painter, state) -> None:
    """Hộp xác nhận chia sẻ — nổi trên mọi thứ, kể cả toast (P2.3)."""
    if not state.confirm or state.shot:
        return
    painter.fill((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1]), rgba("#0B1520", 0.45))
    x, y, w, h = CONFIRM_CARD_RECT
    painter.rounded((x, y, w, h), 18.0, rgba("#FFFFFF"))
    pad, line_h = 28.0, 22.0
    lines = _wrap_lines(painter, [CONFIRM_TEXT_FONT], [CONFIRM_TEXT], w - 2 * pad)
    for index, parts in enumerate(lines):
        painter.line((x + pad, y + pad + index * line_h, w - 2 * pad, line_h),
                     [CONFIRM_TEXT_FONT], parts, "left")
    for name, rect, label, bg in (
            (ID.BTN_CONFIRM_NO, CONFIRM_NO_RECT, "No", "#DCE9F7"),
            (ID.BTN_CONFIRM_YES, CONFIRM_YES_RECT, "Yes", BUTTON_PRIMARY_COL)):
        fg = INK if bg != BUTTON_PRIMARY_COL else "#FFFFFF"
        painter.rounded(rect, 10.0, rgba(bg))
        painter.line(rect, [{**CONFIRM_BTN_FONT, "col": fg}], [label], "center")
        if state.hover == name:
            painter.rounded_frame(rect, rgba(CARET_COL), HOVER_PX, 10.0)


def _draw_dialog_backdrop(painter) -> None:
    painter.fill((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1]),
                rgba(_DIALOG_DIM_COL, _DIALOG_DIM_ALPHA))


def _draw_dialog_button(painter, state, name: str, rect, label: str,
                        primary: bool) -> None:
    bg = BUTTON_PRIMARY_COL if primary else "#DCE9F7"
    fg = "#FFFFFF" if primary else INK
    painter.rounded(rect, 10.0, rgba(bg))
    painter.line(rect, [{**CONFIRM_BTN_FONT, "col": fg}], [label], "center")
    if state.hover == name:
        painter.rounded_frame(rect, rgba(CARET_COL), HOVER_PX, 10.0)


def _draw_recovery_dialog(painter, state) -> None:
    """TASK_accounts_1119 Phase 6: shown exactly once — right after
    sign-up, and again after a successful forgot-password reset (the old
    code is used up, so a fresh one is issued and shown the same way).
    Same style as the share-album confirm dialog: dimmed backdrop, white
    rounded card, Nunito buttons."""
    if not state.recovery_code or state.shot:
        return
    _draw_dialog_backdrop(painter)
    x, y, w, h = RECOVERY_CARD_RECT
    painter.rounded((x, y, w, h), 18.0, rgba("#FFFFFF"))
    pad = 28.0
    painter.line((x + pad, y + pad, w - 2 * pad, 30.0),
                 [RECOVERY_TITLE_FONT], ["Save your recovery code"], "left")
    body = ("This code is the ONLY way back into your album if you forget "
           "your password. It is shown once — write it down or copy it "
           "now.")
    line_h = 20.0
    lines = _wrap_lines(painter, [RECOVERY_BODY_FONT], [body], w - 2 * pad)
    body_y = y + pad + 44.0
    for index, parts in enumerate(lines):
        painter.line((x + pad, body_y + index * line_h, w - 2 * pad, line_h),
                     [RECOVERY_BODY_FONT], parts, "left")
    code_y = body_y + len(lines) * line_h + 20.0
    painter.rounded((x + pad, code_y, w - 2 * pad, 56.0), 10.0, rgba(_FIELD_BG_COL))
    painter.line((x + pad, code_y, w - 2 * pad, 56.0),
                 [RECOVERY_CODE_FONT], [state.recovery_code], "center")
    _draw_dialog_button(painter, state, ID.BTN_RECOVERY_COPY, RECOVERY_COPY_RECT,
                        "Copy", False)
    _draw_dialog_button(painter, state, ID.BTN_RECOVERY_SAVED, RECOVERY_SAVED_RECT,
                        "I saved it", True)





def _draw_forgot_dialog(painter, state) -> None:
    """TASK_accounts_1119 Phase 6: "Forgot password?" — four fields
    (username, recovery code, new password, confirm), an inline error
    line, Reset/Cancel. Its OWN focus (``state.forgot_focus``) and field
    values (``state.texts["forgot_*"]``) are separate from whatever
    screen it overlays (normally sign-in)."""
    if not state.forgot_open or state.shot:
        return
    _draw_dialog_backdrop(painter)
    x, y, w, h = FORGOT_CARD_RECT
    painter.rounded((x, y, w, h), 18.0, rgba("#FFFFFF"))
    painter.line((x + 28.0, y + 28.0, w - 56.0, 30.0),
                 [FORGOT_TITLE_FONT], ["Reset your password"], "left")
    for name, label, field_y, masked in FORGOT_FIELDS:
        rect = (FORGOT_FIELD_X, field_y, FORGOT_FIELD_W, FORGOT_FIELD_H)
        painter.rounded(rect, 10.0, rgba(_FIELD_BG_COL))
        painter.line((FORGOT_FIELD_X + 16.0, field_y + 6.0, FORGOT_FIELD_W - 32.0, 16.0),
                     [_FIELD_LABEL_FONT], [label], "left")
        value = _forgot_field_value(state, name.replace("field_", ""), masked)
        value_rect = forgot_field_value_rect(field_y)
        if state.forgot_focus == name:
            painter.rounded_frame(rect, rgba(CARET_COL), HOVER_PX, 10.0)
            _draw_field_text(painter, value_rect, _FIELD_VALUE_FONT, value,
                             state.caret_index)
        else:
            painter.line(value_rect, [_FIELD_VALUE_FONT], [value], "left")
    error = str(state.texts.get("forgot_error") or "")
    if error:
        painter.line(FORGOT_ERROR_RECT, [FORGOT_ERROR_FONT], [error], "left")
    _draw_dialog_button(painter, state, ID.BTN_FORGOT_SUBMIT, FORGOT_SUBMIT_RECT,
                        "Reset password", True)
    painter.line(FORGOT_CANCEL_RECT, [FORGOT_CANCEL_FONT], ["Cancel"], "center")
    if state.hover == ID.BTN_FORGOT_CANCEL:
        painter.rounded_frame(FORGOT_CANCEL_RECT, rgba(CARET_COL), HOVER_PX, 6.0)


def _draw_aim(painter, state) -> None:
    """
    Dấu ``+`` ở tâm ngắm: bốn nhánh, chừa một khoảng hở ở giữa.

    Chừa hở để chính cái tâm — chỗ viên đạn sẽ tới — không bị nét vẽ che.
    """
    if state.aim is None or state.shot:
        return
    if state.scope_zoom > 1.0 + ZOOM_SETTLED_EPS:
        return          # TASK_game_feel_1118 P6: the scope's own reticle instead
    x, y = float(state.aim[0]), float(state.aim[1])
    ready = max(0.0, min(1.0, float(state.aim_ready)))
    # Chưa hồi xong: bốn nhánh dạt ra và nhạt bớt, rồi khép lại dần.
    spread = AIM_RECOIL * (1.0 - ready)
    fade = AIM_DIM + (1.0 - AIM_DIM) * ready
    # TASK_game_feel_1118 P4: crosshair kick — a ``Spring`` kicked out to
    # the weapon's scale on every shot, settling back to 1.0. Scales the
    # whole cross (arm length AND thickness) around its own centre.
    kick = max(0.1, float(getattr(state, "aim_kick_scale", 1.0)))
    for base, grow in ((AIM_EDGE_COL, AIM_EDGE), (AIM_COL, 0.0)):
        alpha = (0.85 if grow else 1.0) * fade
        colour = rgba(base, alpha)
        half = (AIM_THICK * 0.5 + grow) * kick
        inner = (AIM_GAP + spread - grow) * kick
        outer = (AIM_ARM + spread + grow) * kick
        painter.fill((x - outer, y - half, outer - inner, half * 2.0), colour)
        painter.fill((x + inner, y - half, outer - inner, half * 2.0), colour)
        painter.fill((x - half, y - outer, half * 2.0, outer - inner), colour)
        painter.fill((x - half, y + inner, half * 2.0, outer - inner), colour)


def _flat_quad(painter, corners, colour) -> None:
    """A flat-coloured arbitrary quad — ``quad_corners`` with the painter's
    cached 1x1 white texture, same trick as ``fill()`` uses for axis-
    aligned rects."""
    entry = painter.cache.white()
    if entry:
        painter.quad_corners(corners, entry[0], colour)


def _draw_hit_markers(painter, state) -> None:
    """TASK_game_feel_1118 P4: four ticks at 45° around the crosshair on a
    hit, outlined, fading over 0.18 s — bigger and ``#E8676B`` on a crit.

    TASK_game_feel_fix_1118 Fix 5: a hit marker is spawned at the
    crosshair's position AT THE TIME of the shot — same scene-anchored
    treatment as shot marks (``_scope_adjust``) once the crosshair has
    since moved while still scoped.
    """
    if state.shot:
        return
    for x, y, age, crit in state.hit_markers:
        if age >= HITMARK_LIFE_S:
            continue
        x, y, visible = _scope_adjust(x, y, state)
        if not visible:
            continue
        fade = 1.0 - age / HITMARK_LIFE_S
        scale = HITMARK_CRIT_SCALE if crit else 1.0
        inner, outer = HITMARK_INNER_PX * scale, HITMARK_OUTER_PX * scale
        width, outline = HITMARK_WIDTH_PX * scale, HITMARK_OUTLINE_PX * scale
        core_col = HITMARK_CRIT_COL if crit else HITMARK_COL
        for col, half_w in ((HITMARK_OUTLINE_COL, (width + 2.0 * outline) * 0.5),
                            (core_col, width * 0.5)):
            colour = rgba(col, fade)
            for angle_deg in HITMARK_ANGLES_DEG:
                angle = math.radians(angle_deg)
                dx, dy = math.cos(angle), math.sin(angle)
                nx, ny = -dy, dx
                x0, y0 = x + dx * inner, y + dy * inner
                x1, y1 = x + dx * outer, y + dy * outer
                corners = ((x0 + nx * half_w, y0 + ny * half_w),
                          (x1 + nx * half_w, y1 + ny * half_w),
                          (x1 - nx * half_w, y1 - ny * half_w),
                          (x0 - nx * half_w, y0 - ny * half_w))
                _flat_quad(painter, corners, colour)


def _draw_crit_impact(painter, state) -> None:
    """TASK_game_feel_1118 P4: 0.05 s full-scene flash on a crit — placed
    (by ``draw_screen``'s call order) under the HUD and under the
    ``hit_crit`` Sakuga burst, so the burst pops against it."""
    if state.shot or state.crit_impact_alpha <= 0.0:
        return
    painter.fill((0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1]),
                rgba(CRIT_IMPACT_COL, state.crit_impact_alpha))


def _scope_vignette_texture(painter):
    """A square radial-gradient RGBA texture — transparent inside
    ``VIGNETTE_RADIUS_PX``, full ``VIGNETTE_COL``/``VIGNETTE_ALPHA``
    outside it, with a soft edge in between. Built once per framebuffer
    scale and cached (same ``pil`` cache ``rounded_frame`` uses) — the
    crosshair it's centred on moves every frame, but the gradient itself
    never changes shape, so only the DRAW rect needs to move."""
    scale = painter.scale
    half_px = max(8, int(round((VIGNETTE_RADIUS_PX + VIGNETTE_SOFT_PX) * scale)))
    size_px = half_px * 2
    radius_px = VIGNETTE_RADIUS_PX * scale
    soft_px = max(1.0, VIGNETTE_SOFT_PX * scale)
    key = ("scope_vignette", size_px, round(radius_px, 1), round(soft_px, 1))

    def build():
        yy, xx = np.mgrid[0:size_px, 0:size_px].astype(np.float64)
        dist = np.hypot(xx - half_px + 0.5, yy - half_px + 0.5)
        falloff = np.clip((dist - radius_px) / soft_px, 0.0, 1.0)
        r, g, b, _a = rgba(VIGNETTE_COL)
        image = np.empty((size_px, size_px, 4), dtype=np.uint8)
        image[..., 0] = int(round(r * 255.0))
        image[..., 1] = int(round(g * 255.0))
        image[..., 2] = int(round(b * 255.0))
        image[..., 3] = (falloff * VIGNETTE_ALPHA * 255.0).astype(np.uint8)
        return image, None

    return painter.pil.get(key, build)


def _draw_scope_vignette(painter, state) -> None:
    """TASK_game_feel_1118 P6: outside a 288 design-px circle centred on
    the crosshair, an ``#0F1C2B`` wash at alpha 0.94 with a soft 6 px
    edge — placed (by ``draw_screen``'s call order) above the scene, under
    the HUD, same slot as the crit impact frame."""
    if (state.shot or state.aim is None
            or state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS):
        return
    cx, cy = float(state.aim[0]), float(state.aim[1])
    half = VIGNETTE_RADIUS_PX + VIGNETTE_SOFT_PX
    design_w, design_h = DESIGN_SIZE
    col = rgba(VIGNETTE_COL, VIGNETTE_ALPHA)
    top = max(0.0, cy - half)
    bottom = min(design_h, cy + half)
    if top > 0.0:
        painter.fill((0.0, 0.0, design_w, top), col)
    if bottom < design_h:
        painter.fill((0.0, bottom, design_w, design_h - bottom), col)
    left = max(0.0, cx - half)
    right = min(design_w, cx + half)
    band_h = bottom - top
    if left > 0.0:
        painter.fill((0.0, top, left, band_h), col)
    if right < design_w:
        painter.fill((right, top, design_w - right, band_h), col)
    entry = _scope_vignette_texture(painter)
    if entry is not None:
        size = half * 2.0
        painter.quad((cx - half, cy - half, size, size), entry[0])


#: TASK_game_feel_fix_1118 Fix 10.3: scope mini readout — Nunito 800,
#: white, 2px #24415F outline, 18 design px, "just outside the ring".
_SCOPE_READOUT_FONT = {"ff": "Nunito", "fw": 800, "fs": 18.0}
_SCOPE_READOUT_FG = "#FFFFFF"
_SCOPE_READOUT_OUTLINE_COL = "#24415F"
_SCOPE_READOUT_OUTLINE_PX = 2.0
#: Clear of the ring's own soft edge (``VIGNETTE_SOFT_PX``) plus room for
#: the label's own half-height, so it never overlaps the vignette.
_SCOPE_READOUT_MARGIN_PX = 34.0
_SCOPE_READOUT_RADIUS_PX = VIGNETTE_RADIUS_PX + _SCOPE_READOUT_MARGIN_PX
#: "lower-right ~40°" (task's own wording) — mirrored for the lower-left
#: kills readout; "top" for the timer is the vertical case (90°).
_SCOPE_READOUT_SIDE_DEG = 40.0


def _draw_scope_mini_readout(painter, state) -> None:
    """TASK_game_feel_fix_1118 Fix 10.3: while scoped, the cards have
    faded to 0 (``_card_fade``/``state.card_alpha``, driven by
    ``v3_ui.py``'s Springs) — this is what replaces them: ammo "9/12"
    lower-right, timer top, kills lower-left, all just outside the scope
    ring, so the player still has the three numbers that matter without
    the cards blocking their aim."""
    if (state.shot or state.aim is None
            or state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS):
        return
    cx, cy = float(state.aim[0]), float(state.aim[1])
    r = _SCOPE_READOUT_RADIUS_PX
    rad = math.radians(_SCOPE_READOUT_SIDE_DEG)
    dx, dy = math.cos(rad), math.sin(rad)

    ammo = state.texts.get("ammo")
    if ammo is not None:
        _draw_scope_readout_label(painter, (cx + r * dx, cy + r * dy),
                                  _joined_text(ammo))
    timer = state.texts.get("timer")
    if timer is not None:
        _draw_scope_readout_label(painter, (cx, cy - r), str(timer))
    kills = state.texts.get("kills")
    if kills is None:
        kills = state.texts.get("kills_you")
    if kills is not None:
        _draw_scope_readout_label(painter, (cx - r * dx, cy + r * dy),
                                  _joined_text(kills))


def _draw_scope_readout_label(painter, centre, text: str) -> None:
    painter.outlined_label(centre, text, _SCOPE_READOUT_FONT,
                           _SCOPE_READOUT_FG, _SCOPE_READOUT_OUTLINE_COL,
                           _SCOPE_READOUT_OUTLINE_PX)


def _draw_scope_reticle(painter, state) -> None:
    """TASK_game_feel_1118 P6: a thin 1.5 px cross with a centre gap, in
    ``#24415F`` — the scope's OWN reticle, drawn instead of the normal
    crosshair while scoped (``_draw_aim`` skips itself in that case)."""
    if (state.shot or state.aim is None
            or state.scope_zoom <= 1.0 + ZOOM_SETTLED_EPS):
        return
    x, y = float(state.aim[0]), float(state.aim[1])
    half = RETICLE_THICK_PX * 0.5
    colour = rgba(RETICLE_COL)
    painter.fill((x - RETICLE_ARM_PX, y - half,
                 RETICLE_ARM_PX - RETICLE_GAP_PX, RETICLE_THICK_PX), colour)
    painter.fill((x + RETICLE_GAP_PX, y - half,
                 RETICLE_ARM_PX - RETICLE_GAP_PX, RETICLE_THICK_PX), colour)
    painter.fill((x - half, y - RETICLE_ARM_PX,
                 RETICLE_THICK_PX, RETICLE_ARM_PX - RETICLE_GAP_PX), colour)
    painter.fill((x - half, y + RETICLE_GAP_PX,
                 RETICLE_THICK_PX, RETICLE_ARM_PX - RETICLE_GAP_PX), colour)


def _draw_rings(painter, manifest, screen, record, state) -> None:
    """Vòng hover và vòng focus — lớp cuối, sau mọi thứ (README mục 9).

    TASK_ui_sakuga_1117 Phase 5: chỉ còn một lệnh gọi mỏng vào
    ``v3_hover.HoverFx.draw`` — spring, màu/alpha theo hình thật
    (``rounded``/``pill``/``circle``/``underline``/``thumb``/``none``),
    con trỏ OS, và vòng focus bàn phím đều ở đó. Nhập trong hàm (không ở
    đầu file) để tránh vòng nhập: ``v3_hover`` đọc ngược lại các hằng/hàm
    của module này.
    """
    if state.shot or state.hover_fx is None:
        return
    state.hover_fx.draw(painter, manifest, screen, record, state)


