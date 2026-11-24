"""
Giá trị sống của giao diện v3: trạng thái app/ván chơi → ``V3State``.

Một hàm cho mỗi màn. Khoá nào không được đặt ở đây thì lớp vẽ giữ nguyên chữ
``sample`` trong manifest, nên màn nào cũng đủ nội dung để chụp ảnh luận văn
(và lượt ``--ui-shots`` đi hẳn đường khác: ``v3.shot_state``).

Hai luật xuyên suốt:
- **Số lớp thật.** Bản vẽ in cứng "4 layers" / 4 pip; ở đây mọi chỗ ấy lấy
  ``flow.layer_count`` của ảnh đang mở. Nói 4 khi ảnh ra 3 lớp là nói sai.
- **Chỗ lặp lại không nằm trong manifest.** Penpot xuất MỘT thực thể cho mỗi
  kiểu ô vật phẩm và mỗi dòng điểm; ba ô và n dòng dựng ở đây qua
  ``V3State.extra``, lấy kiểu chữ của chính thực thể ấy làm khuôn.
"""

from __future__ import annotations

import time

from src.account import crypto as account_crypto
from src.content import shared_album
from src.ui_album import v3_flow as F
from src.game.match import WEAPON_LABEL
from src.ui_album.v3 import V3State

#: Màu lấy từ bảng ``colors`` của manifest, gọi tên cho dễ đọc.
INK, SAFE, DANGER = "#24415F", "#3AA17A", "#E8676B"
MUTED, PRIMARY, AMBER = "#8DA3BA", "#3B72B4", "#E08A1E"
SLOT_WARM = "#9A5A1E"

#: Bốn bước của màn tách lớp, theo phần trăm (U2 — Loading).
PIPELINE_STEPS = ((0, 24, "Reading photo"), (25, 59, "Estimating depth"),
                  (60, 84, "Splitting into {n} layers"),
                  (85, 100, "Filling hidden areas"))

#: Đồng hồ đổi sang màu hổ phách từ ngần này giây trở xuống.
TIMER_WARN_S = 30.0

#: ``bite`` hiện khi ảnh đã bị gặm từ ngần này phần trăm.
BITE_PCT = 10.0

#: Dòng "Miss!" sống ngần này giây sau một phát hụt.
MISS_SHOW_S = 0.8

#: Vệt loáng trên màn thử nghiêng dịch theo đầu: 6 px thiết kế mỗi độ, chặn ±120.
GLARE_PX_PER_DEG, GLARE_MAX_PX = 6.0, 120.0


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _initial(name: str) -> str:
    text = (name or "").strip()
    return text[:1].upper() if text else "?"


#: TASK_accounts_1119: ellipsis for a username slot that doesn't fit —
#: one character, so it costs almost nothing of the (narrow) rect it's
#: cut INTO.
_NAME_ELLIPSIS = "…"


def _fit_name(manifest, painter, screen: str, key: str, text: str) -> str:
    """Cuts ``text`` with "…" so it fits ``key``'s own rect on ``screen``,
    measured with the REAL font (``painter.text_width`` — FreeType glyph
    advances, not a character-count guess). Every text slot that shows a
    username (``player_name``, the PvP ``you_name``/``host_name``/
    ``opp_name`` — the LAST one is NETWORK-SUPPLIED, another player's
    display name, so it must never be trusted to already fit) goes
    through this.

    ``painter`` may be ``None`` (``--ui-shots``' ``shot_state`` path, and
    a couple of tests that don't need a real one) — returns ``text``
    unfitted rather than crash; the manifest's own baked layout already
    assumes its SAMPLE name fits, so this only matters for live data.
    """
    if painter is None or manifest is None:
        return text
    record = manifest.text(screen, key)
    if record is None:
        return text
    run = record["runs"][0]
    max_width = float(record["rect"][2])
    if painter.text_width([run], [text]) <= max_width:
        return text
    if painter.text_width([run], [_NAME_ELLIPSIS]) > max_width:
        return _NAME_ELLIPSIS       # pathological: rect too narrow even for "…"
    lo, hi, best = 0, len(text), _NAME_ELLIPSIS
    while lo <= hi:
        mid = (lo + hi) // 2
        candidate = text[:mid] + _NAME_ELLIPSIS
        if painter.text_width([run], [candidate]) <= max_width:
            best = candidate
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def _stem(path) -> str:
    from pathlib import Path

    return Path(path).stem.replace("_", " ").strip() if path else "Living room"


def _layers(flow, game=None) -> int:
    """
    Số lớp THẬT của ảnh đang mở.

    Thứ tự tin cậy: số lớp ván đang chơi đặt bọ lên → số lớp renderer vừa
    dựng → số lớp cấu hình nhắm tới. **Không** có hằng 4 ở đây: 4 là con số
    bản vẽ in sẵn, mà mỗi ảnh tách ra một số lớp khác nhau (adaptive K 2–4),
    nên in 4 cho mọi ảnh là nói sai ba phần tư số lần.
    """
    total = int(getattr(game, "layer_total", 0) or 0)
    if total <= 0:
        total = int(flow.layer_count or 0)
    if total <= 0:
        total = int(flow.layer_target or 0)
    return max(1, total)


def _history(flow, photo):
    return flow.history.get(str(photo)) if photo else None


def _spec(manifest, screen: str, key: str):
    """Bản ghi text ``key`` của màn — dùng làm khuôn cho dòng chữ app tự đặt."""
    for record in manifest.screens[screen].get("texts", ()):
        if record["key"] == key:
            return record
    return None


def _pip_count(manifest, screen: str, key: str) -> int:
    """Số ô pip bản vẽ xuất cho dãy ``key`` — trần vẽ được của dãy đó."""
    for record in manifest.screens[screen].get("pips", ()):
        if record["key"] == key:
            return len(record["rects"])
    return 0


def _row_pitch(manifest, screen: str, prefix: str) -> float:
    """Khoảng cách dòng, lấy từ nhịp của chính hai dòng #0 → #1 (U2)."""
    first = _spec(manifest, screen, f"{prefix}#0")
    second = _spec(manifest, screen, f"{prefix}#1")
    if first is None or second is None:
        return 0.0
    return float(second["rect"][1]) - float(first["rect"][1])


def _extra_row(state, manifest, screen, key, index, text, colour=None) -> bool:
    """
    Đặt một dòng chữ ở nhịp thứ ``index`` của nhóm ``key``.

    Trả ``True`` nếu dòng này dùng ô có sẵn trong manifest (caller không phải
    ẩn nó), ``False`` nếu là dòng thêm vẽ qua ``extra``.
    """
    prefix = key.rsplit("#", 1)[0]
    native = _spec(manifest, screen, f"{prefix}#{index}")
    if native is not None:
        state.texts[native["key"]] = text
        if colour:
            state.colours[native["key"]] = colour
        return True
    template = _spec(manifest, screen, f"{prefix}#0")
    if template is None:
        return False
    pitch = _row_pitch(manifest, screen, prefix)
    rect = list(template["rect"])
    rect[1] = float(rect[1]) + pitch * index
    runs = [{**template["runs"][0], **({"col": colour} if colour else {})}]
    state.extra.append((rect, runs, [str(text)], template.get("al", "left")))
    return False


def extra_files(manifest, screen: str, state: V3State) -> tuple:
    """File sprite vay của màn khác — ``bind_screen`` phải nạp kèm."""
    out = set()
    for other, key, _rect in state.borrowed.values():
        if other == screen:
            continue
        record = manifest.sprite(other, key)
        if record is not None:
            out.add(record["file"])
    return tuple(sorted(out))


# ---- TASK_accounts_1119 Phase 6: recovery code + forgot-password dialogs --
def _dialogs(state, flow) -> None:
    """Populates both overlay dialogs' state — called UNCONDITIONALLY
    (regardless of ``flow.screen``), since either can be open on top of
    whatever screen is underneath (the recovery dialog after sign-up
    lands on MENU; the forgot dialog opens on SIGN_IN/SIGN_IN_ERR)."""
    state.recovery_code = flow.recovery_code_shown
    state.forgot_open = flow.forgot_open
    state.forgot_focus = flow.forgot_focus
    if not flow.forgot_open:
        return
    state.caret_index = flow.caret_index
    state.texts["forgot_user"] = flow.forgot_user
    state.texts["forgot_code"] = flow.forgot_code
    state.texts["forgot_pass"] = flow.forgot_new_pass
    state.texts["forgot_pass2"] = flow.forgot_new_pass2
    state.texts["forgot_error"] = flow.forgot_error


# ---- màn đăng nhập / đăng ký -------------------------------------------
def _signup_problem(flow, accounts: dict | None) -> str | None:
    """Vấn đề ĐẦU TIÊN (trên xuống: tên → mật khẩu → nhập lại) của một ô
    người dùng đã THẬT SỰ gõ vào (``flow.touched``) — ``None`` nếu không
    ô nào đã động tới có vấn đề. ``accounts`` là dữ liệu kho tài khoản
    (để kiểm tên đã bị lấy chưa); ``None`` bỏ qua việc kiểm đó (vd. khi
    gọi mà chưa có kho — không nên xảy ra ở đường chạy thật, nhưng tránh
    văng thay vì giả định)."""
    if "field_user" in flow.touched:
        display, reason = account_crypto.validate_username(flow.player)
        if display is None:
            return reason
        if accounts is not None and account_crypto.username_key(display) in accounts.get(
                "accounts", {}):
            return "Username already taken"
    if "field_pass" in flow.touched:
        ok, reason = account_crypto.validate_password(flow.password)
        if not ok:
            return reason
    if "field_pass2" in flow.touched and flow.password2 != flow.password:
        return "Passwords don't match"
    return None


def _sign_in(state, flow, accounts: dict | None = None) -> None:
    dots = flow.password if flow.show_password else "•" * len(flow.password)
    dots2 = flow.password2 if flow.show_password else "•" * len(flow.password2)
    state.texts["user"] = flow.player
    state.texts["pass"] = dots
    state.texts["pass2"] = dots2
    state.remember = flow.remember
    state.caret = {"field_user": "user", "field_pass": "pass",
                   "field_pass2": "pass2"}.get(flow.focus or "")
    state.caret_index = flow.caret_index
    state.focus = flow.focus
    if flow.screen == F.SIGN_IN_ERR:
        state.texts["error"] = flow.sign_in_error
    if flow.screen == F.SIGN_UP:
        # TASK_accounts_1119: "valid_user" chỉ bật khi tên VỪA hợp lệ VỪA
        # còn trống — không còn là "có gõ gì đó" như bản mô phỏng cũ.
        display, _reason = account_crypto.validate_username(flow.player)
        taken = (display is not None and accounts is not None
                and account_crypto.username_key(display) in accounts.get("accounts", {}))
        if display is not None and not taken:
            state.sprites["valid_user"] = True
        pass_ok, _reason2 = account_crypto.validate_password(flow.password)
        if pass_ok and flow.password == flow.password2:
            state.sprites["valid_pass2"] = True
        problem = _signup_problem(flow, accounts)
        if problem is None:
            state.sprites["hint_ok"] = True
            # Chưa đặt ``state.texts["hint"]`` KHÔNG có nghĩa là ẩn —
            # lớp vẽ coi khoá vắng mặt là "giữ nguyên chữ mẫu trong
            # manifest" (để màn nào cũng đủ chữ cho --ui-shots), nên
            # phải ẩn THẬT bằng ``state.hidden`` ở đây, nếu không chữ
            # mẫu "Username already taken" đè lên đúng lúc hint_ok xanh
            # đang hiện.
            state.hidden.add("hint")
        else:
            state.sprites["hint_bad"] = True
            state.texts["hint"] = problem


# ---- màn chính ---------------------------------------------------------
def _wallet_line(state, manifest, screen, rect, template_key, text) -> None:
    """A text the art has no slot for (wallet, price): draw it with the
    style of an existing text ``template_key`` of the same screen."""
    template = _spec(manifest, screen, template_key) if manifest else None
    if template is None:
        return
    state.extra.append((list(rect), template["runs"], [text],
                        template.get("al", "left")))


def _menu(state, flow, manifest=None, painter=None) -> None:
    name = flow.player or "guest"
    state.texts["player_name"] = _fit_name(manifest, painter, F.MENU, "player_name", name)
    state.texts["player_initial"] = _initial(name)
    photo = flow.photo or (flow.recent[0] if flow.recent else None)
    state.photo = photo
    index = (len(flow.history) % 99) + 1
    state.texts["photo_caption"] = f"Memory #{index:02d} · {_stem(photo)}"
    record = _history(flow, photo)
    if record:
        state.texts["photo_damage"] = (
            f"{record['date']} · {record['eaten']:.0f}% eaten")
    else:
        state.texts["photo_damage"] = "not played yet"
    state.focus = flow.focus if flow.focus in F.MENU_BUTTONS else None
    if flow.wallet is not None:
        _wallet_line(state, manifest, F.MENU, (144, 596, 360, 22),
                     "photo_caption", f"Points  {flow.wallet}")
    reason = F.pvp_lock_reason(flow)
    if reason is not None:
        _wallet_line(state, manifest, F.MENU, (144, 622, 600, 22),
                     "photo_caption", reason)


# ---- màn cài đặt -------------------------------------------------------
def _settings(state, flow) -> None:
    state.texts["webcam_name"] = flow.webcam_name
    state.texts["volume_value"] = f"{int(flow.volume)}%"
    state.bars["volume"] = flow.volume / 100.0
    state.sprites["seg_theme:day"] = True       # v3 chỉ có một bảng màu
    key = "seg_display:window" if flow.display == "window" else "seg_display:full"
    state.sprites[key] = True
    state.sprites["face:ok" if flow.face_seen else "face:none"] = True
    # Núm âm lượng: tâm nằm ở cuối phần đã đầy của rãnh ``volume``.
    state.sprites["volume_knob"] = _knob_rect(flow)


#: Rãnh âm lượng trong không gian thiết kế (``bars[].rect`` của màn cài đặt).
VOLUME_TRACK = (520.0, 563.0, 170.0)
#: ``rect`` và ``anchor`` của sprite núm — lệch nhau đúng phần viền mềm.
KNOB_RECT = (617.3, 544.0, 44.0, 44.0)
KNOB_ANCHOR_CX = 627.0 + 12.0


def _knob_rect(flow) -> tuple:
    start, _y, width = VOLUME_TRACK
    centre = start + width * (flow.volume / 100.0)
    return (KNOB_RECT[0] + (centre - KNOB_ANCHOR_CX), KNOB_RECT[1],
            KNOB_RECT[2], KNOB_RECT[3])


# ---- mất kết nối -------------------------------------------------------
def _offline(state, flow, now) -> None:
    left = max(0, int(round(F.RECONNECT_S - (now - flow.offline_since_s))))
    state.texts["reconnect_left"] = f"{left} seconds left"


# ---- chọn cấp ----------------------------------------------------------
def _levels(state, flow, manifest) -> None:
    level = flow.level
    state.texts["level_name"] = flow.level_name()
    state.texts["level_desc"] = F.LEVEL_DESC[level]
    state.texts["level_tag"] = f"LEVEL {level}"
    state.texts["stat_bugs"] = str(flow.bugs())
    state.texts["stat_time"] = mmss(flow.seconds())
    state.texts["stat_slots"] = str(flow.level_slots())
    state.pips["stat_slots_pips"] = flow.level_slots()
    state.pips["difficulty"] = level
    record = manifest.sprite(F.LEVELS, "sel:level")
    if record is not None:
        state.sprites["sel:level"] = _on_target(record, f"level_row_{level}")
    if level == F.NEW_BUG_LEVEL:
        state.sprites["new_bug"] = True
        state.texts["new_bug_name"] = F.NEW_BUG_NAME
    else:
        state.hidden |= {"new_bug", "new_bug_name"}


def _on_target(record: dict, target: str):
    """``rect + (T.x − anchor.x, T.y − anchor.y)`` — README mục Draw order 7."""
    rect, anchor = record["rect"], record["anchor"]
    goal = (record.get("targets") or {}).get(target)
    if goal is None:
        return True
    return (rect[0] + goal[0] - anchor[0], rect[1] + goal[1] - anchor[1],
            rect[2], rect[3])


# ---- chọn ảnh ----------------------------------------------------------
def _photos(state, flow) -> None:
    level = flow.level
    state.texts["eyebrow"] = (f"SOLO · STEP 2 OF 3 · LEVEL {level} "
                              f"{flow.level_name().upper()}")
    photo = flow.photo or (flow.recent[0] if flow.recent else None)
    state.photo = photo
    index = (len(flow.history) % 99) + 1
    state.texts["photo_caption"] = f"Memory #{index:02d} · {_stem(photo)}"
    record = _history(flow, photo)
    if record:
        state.texts["photo_date"] = f"last played {record['date'][:5]}"
    else:
        state.texts["photo_date"] = "new photo"
    if record and record.get("eaten", 0.0) > 0.0:
        state.sprites["damage_badge"] = True
        state.texts["damage_text"] = f"{record['eaten']:.0f}% eaten"
    else:
        state.hidden |= {"damage_badge", "damage_text"}
    can_upload = shared_album.can_upload()
    state.texts["share_to_album_label"] = ("Share to album" if can_upload
                                           else shared_album.NOT_CONFIGURED)
    state.alpha["share_to_album_btn"] = 1.0 if can_upload else 0.55
    # P3: list/random chi can cloud_name+tag, khong can upload_preset.
    can_list = shared_album.load_config() is not None
    state.texts["random_from_album_label"] = ("Random from album" if can_list
                                               else shared_album.NOT_CONFIGURED)
    state.alpha["random_from_album_btn"] = 1.0 if can_list else 0.55


# ---- màn tách lớp ------------------------------------------------------
def _loading(state, flow) -> None:
    pct = max(0.0, min(100.0, float(flow.pipeline_pct)))
    layers = _layers(flow)
    # Trước 60 % thì lớp chưa dựng xong nên con số còn là ước lượng; nói "about"
    # thay vì khẳng định một con số rồi lát nữa đổi sang con số khác.
    known = bool(flow.layer_count)
    state.bars["progress"] = pct / 100.0
    state.texts["progress_value"] = f"{pct:.0f}%"
    step = 0
    for index, (low, high, _name) in enumerate(PIPELINE_STEPS):
        if low <= pct <= high:
            step = index
    count = f"{layers}" if known else f"~{layers}"
    name = PIPELINE_STEPS[step][2].format(n=count)
    # "of 4" ở đây là 4 BƯỚC của đường ống, không phải 4 lớp — đừng thay.
    state.texts["progress_step"] = f"Step {step + 1} of 4 · {name}"
    state.texts["step_3"] = f"Split into {count} layers"
    state.pips["step_dots"] = [index <= step for index in range(4)]
    # Màu nhãn bốn bước theo ``screens[].notes``: xong xanh lá w700, đang làm
    # mực w800, chưa tới xám w700. Độ đậm nằm trong manifest nên chỉ đổi màu.
    for index in range(4):
        if index < step:
            state.colours[f"step_{index + 1}"] = SAFE
        elif index == step:
            state.colours[f"step_{index + 1}"] = INK
        else:
            state.colours[f"step_{index + 1}"] = MUTED


# ---- màn trang bị ------------------------------------------------------
def _loadout(state, flow, manifest, painter) -> None:
    screen = F.LOADOUT
    level, items = flow.level, flow.items
    state.texts["note"] = F.LEVEL_NOTE[level]
    state.texts["weight_total"] = str(flow.weight_total())
    if flow.wallet is not None:
        left = flow.wallet - flow.price_total()
        _wallet_line(state, manifest, screen, (912, 540, 304, 24), "weight_total",
                     f"Points {flow.wallet} · items {flow.price_total()}"
                     f" · left {left}")
    used, total = len(items), flow.slots()
    state.texts["slots_label"] = f"ITEM SLOTS · {used} / {total}"
    state.colours["slots_label"] = MUTED if used == 0 else SLOT_WARM
    if flow.weapon is None:
        state.sprites["tip:need_weapon"] = True
        state.sprites["btn_start:off"] = True
        state.hidden.add("tip_text")
    else:
        state.sprites["tip:level"] = True
        state.sprites["btn_start:on"] = True
        state.texts["tip_text"] = F.LEVEL_TIP[level]
        record = manifest.sprite(screen, "sel:weapon")
        if record is not None:
            state.sprites["sel:weapon"] = _on_target(
                record, f"weapon_{flow.weapon_name()}")
    # Nhãn trong manifest là một thực thể mẫu; ba ô thật dựng ở dưới.
    state.hidden |= {"slot_key", "slot_name", "slot_sub",
                     "sticker_placed_label"}
    _loadout_stickers(state, flow, manifest)
    _loadout_slots(state, flow, manifest, painter)


def _loadout_stickers(state, flow, manifest) -> None:
    """Mỗi sticker: hoặc chính nó, hoặc ``sticker_placed`` + "In slot n"."""
    screen = F.LOADOUT
    placed = manifest.sprite(screen, "sticker_placed")
    label = _spec(manifest, screen, "sticker_placed_label")
    rects = manifest.screens[screen].get("sticker_rects", {})
    for item in F.ITEM_WEIGHT:
        if item not in flow.items:
            state.sprites[f"sticker:{item}"] = True
            continue
        slot = flow.items.index(item) + 1
        anchor = rects.get(item)
        if placed is None or anchor is None:
            continue
        offset = (anchor[0] - placed["anchor"][0],
                  anchor[1] - placed["anchor"][1])
        # ``state.sprites`` đánh khoá theo TÊN sprite, nên gắn hai vật phẩm là
        # lần gán sau đè lần trước và chỉ một cái được vẽ. Chỗ lặp lại phải đi
        # qua ``state.borrowed`` — nhãn ở đó là duy nhất cho từng thực thể.
        state.borrowed[f"sticker_placed#{slot}"] = (
            screen, "sticker_placed",
            (placed["rect"][0] + offset[0], placed["rect"][1] + offset[1],
             placed["rect"][2], placed["rect"][3]))
        if label is not None:
            rect = list(label["rect"])
            rect[0] = float(rect[0]) + offset[0]
            rect[1] = float(rect[1]) + offset[1]
            state.extra.append((rect, label["runs"], [f"In slot {slot}"],
                                label.get("al", "center")))


def _loadout_slots(state, flow, manifest, painter) -> None:
    """Ba ô trang bị: trống, hoặc ``slot_filled`` + tile + ba dòng chữ."""
    screen = F.LOADOUT
    record = manifest.screens[screen]
    rects = record.get("slot_rects", ())
    layout = record.get("slot_layout", {})
    filled = manifest.sprite(screen, "slot_filled")
    usable = flow.slots()
    for index, slot_rect in enumerate(rects):
        number = index + 1
        if index >= len(flow.items):
            state.sprites[f"slot_empty:{number}"] = True
            # Cấp 1 chỉ có hai ô dùng được: ô thứ ba vẫn vẽ nhưng mờ 45 %, để
            # người chơi thấy cấp sau sẽ mở thêm chỗ (U2 — Loadout).
            if number > usable:
                state.alpha[f"slot_empty:{number}"] = 0.45
            continue
        item = flow.items[index]
        if filled is not None:
            offset = (slot_rect[0] - filled["anchor"][0],
                      slot_rect[1] - filled["anchor"][1])
            # Cùng bẫy với ``sticker_placed``: một khoá trong ``state.sprites``
            # chỉ giữ được MỘT ô, nên ô 1 mất trắng cả thẻ lẫn số khi ô 2 ghi
            # đè. ``borrowed`` giữ thứ tự chèn, nên thẻ nằm dưới icon.
            state.borrowed[f"slot_filled#{index}"] = (
                screen, "slot_filled",
                (filled["rect"][0] + offset[0], filled["rect"][1] + offset[1],
                 filled["rect"][2], filled["rect"][3]))
        icon = layout.get("icon")
        if icon:
            state.borrowed[f"slot_icon#{index}"] = (
                screen, f"item_tile:{item}",
                (slot_rect[0] + icon[0], slot_rect[1] + icon[1],
                 icon[2], icon[3]))
        weight = F.ITEM_WEIGHT.get(item, 0)
        _slot_line(state, manifest, screen, "slot_key", slot_rect,
                   layout.get("key"), str(number))
        _slot_line(state, manifest, screen, "slot_name", slot_rect,
                   layout.get("name"), item)
        _slot_line(state, manifest, screen, "slot_sub", slot_rect,
                   layout.get("sub"),
                   f"key {number} · {F.item_price(item)} pts"
                   + ("" if F.ITEMS_HAVE_EFFECT else " · no effect yet"))
    _ = painter


def _slot_line(state, manifest, screen, key, slot_rect, offset, text) -> None:
    """Một dòng chữ trong ô trang bị, đặt theo ``slot_layout``."""
    template = _spec(manifest, screen, key)
    if template is None or offset is None:
        return
    rect = list(template["rect"])
    if len(offset) == 4:
        rect = [slot_rect[0] + offset[0], slot_rect[1] + offset[1],
                offset[2], offset[3]]
    else:
        rect = [slot_rect[0] + offset[0], slot_rect[1] + offset[1],
                rect[2], rect[3]]
    state.extra.append((rect, template["runs"], [str(text)],
                        template.get("al", "left")))


# ---- màn thử nghiêng ---------------------------------------------------
def _tilt(state, flow, manifest) -> None:
    screen = F.TILT
    angle = float(flow.tilt_deg)
    state.texts["tilt_value"] = f"Tilt {angle:+.0f}°".replace("+", "+")
    indicator = manifest.sprite(screen, "tilt_indicator")
    if indicator is not None:
        state.sprites["tilt_indicator"] = True
        # Mũi tên phải chỉ về phía đầu đang nghiêng. Sprite vẽ sẵn hướng
        # TRÁI, nên góc dương thì lật ngang — chỉ lật hình, chữ "Tilt +11°"
        # giữ nguyên (lật cả chữ thì đọc ngược).
        if angle > 0.0:
            state.mirror.add("tilt_indicator")
    glare = manifest.sprite(screen, "glare")
    if glare is not None:
        shift = max(-GLARE_MAX_PX, min(GLARE_MAX_PX, angle * GLARE_PX_PER_DEG))
        rect = glare["rect"]
        state.sprites["glare"] = (rect[0] + shift, rect[1], rect[2], rect[3])
    checks = flow.tilt_checks()
    current = next((i for i, done in enumerate(checks) if not done), None)
    marker = manifest.sprite(screen, "check:done")
    labels = ("check_face", "check_left", "check_right")
    hints = ("look at the camera", "tilt your head to the left",
             "tilt your head to the right")
    for index, done in enumerate(checks):
        kind = "done" if done else ("current" if index == current else "todo")
        _check_row(state, manifest, screen, index, kind, marker)
        key = labels[index]
        if kind == "current":
            state.colours[key] = PRIMARY
            state.texts["check_hint"] = hints[index]
            # TASK_ui_sakuga_1117 bug A: bản vẽ khoá "check_hint" ở y=549
            # (hàng "Tilt right"); dời theo hàng ĐANG hỏi, giữ nguyên x.
            hint_spec = _spec(manifest, screen, "check_hint")
            if hint_spec is not None:
                hx, _hy, hw, hh = hint_spec["rect"]
                state.text_rect["check_hint"] = (hx, CHECK_ROWS_Y[index], hw, hh)
        elif kind == "todo":
            state.colours[key] = MUTED
        else:
            state.colours[key] = INK
    if current is None:
        state.hidden.add("check_hint")
    state.sprites["btn_enter:on" if flow.tilt_ready() else "btn_enter:off"] = True


#: Ba hàng dấu kiểm (``screens[].notes`` của màn thử nghiêng).
CHECK_ROWS_Y = (486.0, 518.0, 550.0)

#: TASK_ui_sakuga_1117 bug A — mỗi biến thể (``check:done``/``current``/
#: ``todo``) được XUẤT ở một hàng gốc khác nhau trong bản vẽ, không phải cả
#: ba ở hàng 0 như code cũ giả định (``done`` ở hàng 0 = 486, còn
#: ``current``/``todo`` ở hàng 2 = 550). Dời sai thì ``current``/``todo``
#: lệch xuống dưới đúng khoảng ``550 - CHECK_ROWS_Y[hàng đích]``.
CHECK_VARIANT_ANCHOR_Y = {"done": 486.0, "current": 550.0, "todo": 550.0}


def _check_row(state, manifest, screen, index, kind, marker) -> None:
    """
    Dấu kiểm của hàng ``index``: một trong ``done|current|todo``.

    Đặt theo CHÍNH gốc của biến thể đang dùng
    (``CHECK_VARIANT_ANCHOR_Y[kind]``), không phải luôn tính từ hàng 0 —
    ``done`` neo ở 486, ``current``/``todo`` neo ở 550.
    """
    record = manifest.sprite(screen, f"check:{kind}")
    if record is None:
        return
    rect = record["rect"]
    y = rect[1] + (CHECK_ROWS_Y[index] - CHECK_VARIANT_ANCHOR_Y[kind])
    state.borrowed[f"check#{index}"] = (
        screen, f"check:{kind}", (rect[0], y, rect[2], rect[3]))
    _ = marker


# ---- HUD ---------------------------------------------------------------
def _hud(state, flow, game, manifest, now) -> None:
    screen = flow.screen
    level, layers = flow.level, _layers(flow, game)
    pvp = screen in (F.PVP_HUNT, F.PVP_HIDE)
    if screen == F.PVP_HIDE:
        state.texts["level_tab"] = "PVP · HUNT & HIDE · YOU ARE THE BUG"
    elif pvp:
        state.texts["level_tab"] = f"PVP · {flow.mode.upper()}"
    else:
        state.texts["level_tab"] = f"LEVEL {level} · {flow.level_name().upper()}"
    current = 1 if game is None else max(1, min(layers, game.layer_index))
    # ``layer`` là MỘT run ("LAYER 2/4"), không phải hai như ``ammo``/``kills``
    # — đưa danh sách hai phần tử vào đây thì phần "/N" rơi mất không kêu.
    state.texts["layer"] = f"LAYER {current}/{layers}"
    # Bản vẽ chỉ xuất 4 ô pip. Ảnh ra NHIỀU hơn 4 lớp thì dãy pip không nói
    # hết được — chữ "a / N" mới là chỗ nói con số thật, còn dãy pip chỉ vẽ
    # được tới chừng nào nó có ô. Ảnh ra ÍT hơn thì ẩn bớt ô thừa.
    shown = min(layers, _pip_count(manifest, screen, "layer_pips"))
    state.pips["layer_pips"] = min(current, shown)
    state.pips["layer_pips:limit"] = shown
    left = flow.seconds() if game is None else game.time_left_s
    state.texts["timer"] = mmss(abs(left))
    overtime = left < 0.0
    state.colours["timer"] = AMBER if (left <= TIMER_WARN_S or overtime) else INK
    eaten = 0.0 if game is None else game.eaten_pct
    intact = max(0.0, 100.0 - eaten)
    state.texts["eaten"] = f"{eaten:.0f}% eaten"
    state.bars["intact"] = intact / 100.0
    state.texts["intact_value"] = [f"{intact:.0f}", "%"]
    if eaten >= BITE_PCT:
        state.sprites["bite"] = True
    else:
        state.hidden.add("bite")
    hp = 100.0 if game is None else game.hp
    state.texts["hp_value"] = f"{hp:.0f}"
    state.bars["hp"] = hp / 100.0
    if level == 5:
        stamina = 100.0 if game is None else game.stamina
        state.sprites["st_block"] = True
        state.texts["st_value"] = f"{stamina:.0f}%"
        state.bars["st"] = stamina / 100.0
    else:
        state.hidden |= {"st_block", "st_value", "st"}
    state.sprites["head:good" if (flow.face_seen and not flow.tracker_simulated)
                  else "head:lost"] = True
    _hud_shot(state, flow, game, now, level)
    _hud_weapon(state, flow, game, manifest, screen)
    if game is not None:
        state.shots = [(x, y, now - fired, hit)
                       for x, y, fired, hit in getattr(game, "shots", ())]
        state.hp_bars = game.hp_bars(now)
        if game.reloading(now):
            # Không có ô nào trên bản vẽ dành cho việc nạp đạn, nên nói bằng
            # dòng báo — nếu không, bấm chuột mà súng im là khó hiểu.
            state.message = "Reloading…"
    if screen == F.PLAY:
        _hud_kills(state, flow, game)
        _hud_items(state, flow, game, manifest, screen)
    elif screen == F.PVP_HUNT:
        _hud_pvp_hunter(state, flow, game)
        _hud_items(state, flow, game, manifest, screen)
    elif screen == F.PVP_HIDE:
        _hud_pvp_bug(state, flow, game)
    _hud_effects(state, flow, game, manifest, screen)


def _aim_ready(game, now: float) -> float:
    """
    Súng đã hồi tới đâu, 0–1. Đang nạp thì 0 suốt.

    Đây là cách duy nhất người chơi **nhìn thấy** nhịp bắn: không có âm thanh
    trong bản demo, nên cú bấm bị chặn vì chưa tới nhịp sẽ im lặng tuyệt đối
    nếu tâm ngắm không nói gì.
    """
    if game is None:
        return 1.0
    if game.reloading(now):
        return 0.0
    gap = float(getattr(game, "fire_interval_s", 0.0) or 0.0)
    if gap <= 0.0:
        return 1.0
    left = float(getattr(game, "next_shot_s", 0.0)) - now
    if left <= 0.0:
        return 1.0
    return max(0.0, 1.0 - left / gap)


def _hud_shot(state, flow, game, now, level) -> None:
    """Tâm ngắm và dòng "Miss!" — cả hai đi theo chỗ vừa bắn."""
    missed = game is not None and getattr(game, "missed_until_s", 0.0) > now
    if not missed:
        state.hidden |= {"hit", "hit_sub"}
        return
    state.texts["hit"] = "Miss!"
    if level == 5:
        state.texts["hit_sub"] = "-5 stamina"
    else:
        state.hidden.add("hit_sub")
    _ = flow


def _hud_weapon(state, flow, game, manifest, screen) -> None:
    name = flow.weapon_name() or (game.weapon_name if game else F.WEAPONS[0])
    state.texts["weapon_name"] = WEAPON_LABEL.get(name, name).upper()
    rounds, total = (10, 10) if game is None else game.rounds
    state.texts["ammo"] = [str(rounds), f"/{total}"]
    # Bản vẽ chỉ xuất 10 ô pip; băng 16 viên thì dãy pip kịch trần còn con số
    # "a/b" mới nói đủ — cùng lối với layer_pips.
    shown = min(total, _pip_count(manifest, screen, "ammo_pips"))
    state.pips["ammo_pips"] = min(rounds, shown)
    state.pips["ammo_pips:limit"] = shown
    icon = manifest.sprite(screen, f"weapon_icon:{name}")
    if icon is not None:
        state.sprites[f"weapon_icon:{name}"] = True
        return
    # Hai súng kia không có icon riêng: vẽ tile của màn trang bị vào cùng ô.
    slot = manifest.sprite(screen, "weapon_icon:Precision gun")
    tile = manifest.sprite(F.LOADOUT, f"weapon_tile:{name}")
    if slot is not None and tile is not None:
        state.borrowed["weapon_icon"] = (F.LOADOUT, f"weapon_tile:{name}",
                                         slot["rect"])
    state.hidden.add("weapon_icon:Precision gun")


def _bug_note(state, holds, promised, wide: bool = False) -> None:
    """Say it when the photo holds fewer bugs than the level promises.
    ``wide`` = the loading screen, which has room for a full sentence; the
    HUD card only fits the short form."""
    if holds is not None and holds < promised:
        state.texts["bug_note"] = (
            f"This photo only holds {holds} of the {promised} bugs this "
            f"level asks for" if wide else f"photo holds {holds} of {promised}")


def _hud_kills(state, flow, game) -> None:
    total = flow.bugs() if game is None else game.bugs_total
    found = 0 if game is None else game.bugs_found
    state.texts["kills"] = [str(found), f"/{total}"]
    _bug_note(state, total, flow.bugs())


def _hud_pvp_hunter(state, flow, game) -> None:
    total = flow.bugs() if game is None else game.bugs_total
    found = 0 if game is None else getattr(game, "my_kills", 0)
    opponent = 0 if game is None else getattr(game, "bot_found", 0)
    state.texts["kills_you"] = [str(found), f"/{total}"]
    state.texts["kills_opp"] = [str(opponent), f"/{total}"]
    state.bars["progress_you"] = found / max(1, total)
    state.bars["progress_opp"] = opponent / max(1, total)
    items = 0 if game is None else getattr(game, "bot_items", 0)
    plural = "item" if items == 1 else "items"
    opp_name = flow.pvp_opponent_name or "Opponent"
    state.texts["opponent_activity"] = f"{opp_name} used {items} {plural}"


def _hud_pvp_bug(state, flow, game) -> None:
    total = flow.bugs() if game is None else game.bugs_total
    hunter = 0 if game is None else getattr(game, "bot_found", 0)
    state.texts["hunter_progress"] = f"Hunter has killed {hunter}/{total} bugs"
    exposure = 0.0 if game is None else getattr(game, "exposure_pct", 0.0)
    state.texts["exposure_value"] = f"{exposure:.0f}%"
    state.bars["exposure"] = exposure / 100.0


def _hud_items(state, flow, game, manifest, screen) -> None:
    """Ba ô vật phẩm của HUD, dựng từ ``item_slots`` + ``item_slot_layout``.

    TASK_hud_fix_1119 Fix 1: these always draw at full, designed size —
    the alternate small-layout mode that used to shrink them (removed
    this task) is gone. Fix 2's H-hide fades them via
    ``state.hud_hidden_alpha`` instead (``v3.py``'s ``_draw_sprites``/
    ``_draw_texts`` apply it to ``state.borrowed``/``state.extra``
    directly, since these have no card of their own).
    """
    record = manifest.screens[screen]
    slots = record.get("item_slots", ())
    layout = record.get("item_slot_layout", {})
    if not slots or not layout:
        return
    state.hidden |= {"islot_key", "islot_key_sel", "islot_name",
                     "islot_name_used", "islot_count", "islot_count_used"}
    used = getattr(game, "used", {}) if game is not None else {}
    selected = int(flow.slot_selected)
    for index, slot in enumerate(slots):
        number = index + 1
        if index >= len(flow.items):
            state.hidden |= {f"islot:{k}" for k in ("normal", "selected")}
            continue
        item = flow.items[index]
        left = max(0, 1 - int(used.get(item, 0)))
        mark = "selected" if number == selected else "normal"
        state.slot_labels |= {f"islot#{index}", f"ikey#{index}",
                              f"islot_icon#{index}", f"iused#{index}"}
        _place(state, manifest, screen, f"islot:{mark}", slot, (0.0, 0.0),
               f"islot#{index}")
        key_off = layout.get("key") or (0, 0, 20, 20)
        _place(state, manifest, screen, f"ikey:{mark}", slot,
               (key_off[0], key_off[1]), f"ikey#{index}")
        icon = layout.get("icon")
        if icon:
            state.borrowed[f"islot_icon#{index}"] = (
                F.LOADOUT, f"item_tile:{item}",
                (slot[0] + icon[0], slot[1] + icon[1], icon[2], icon[3]))
        if left == 0:
            badge = layout.get("used_badge")
            if badge:
                _place(state, manifest, screen, "iused", slot,
                       (badge[0], badge[1]), f"iused#{index}")
        _slot_text(state, manifest, screen,
                   "islot_key_sel" if mark == "selected" else "islot_key",
                   slot, layout.get("key"), str(number))
        _slot_text(state, manifest, screen,
                   "islot_name_used" if left == 0 else "islot_name",
                   slot, layout.get("name"), item)
        count = layout.get("count_right")
        if count:
            _slot_text(state, manifest, screen,
                       "islot_count_used" if left == 0 else "islot_count",
                       slot, (count[0] - 40.0, count[1], 40.0, 17.0),
                       f"×{left}", align="right")


def _place(state, manifest, screen, key, slot, offset, label) -> None:
    """Đặt một sprite ô vật phẩm ở góc ``slot`` + ``offset``."""
    record = manifest.sprite(screen, key)
    if record is None:
        return
    anchor, rect = record["anchor"], record["rect"]
    shift = (slot[0] + offset[0] - anchor[0], slot[1] + offset[1] - anchor[1])
    state.borrowed[label] = (screen, key,
                             (rect[0] + shift[0], rect[1] + shift[1],
                              rect[2], rect[3]))


def _slot_text(state, manifest, screen, key, slot, offset, text,
               align=None) -> None:
    template = _spec(manifest, screen, key)
    if template is None or offset is None:
        return
    base = template["rect"]
    if len(offset) == 4:
        rect = [slot[0] + offset[0], slot[1] + offset[1], offset[2], offset[3]]
    else:
        rect = [slot[0] + offset[0], slot[1] + offset[1], base[2], base[3]]
    state.slot_extra.append((rect, template["runs"], [str(text)],
                             align or template.get("al", "left")))


#: Hiệu ứng xếp từ x = 52, cách nhau 8 px thiết kế (README mục Key rules).
EFFECT_X0, EFFECT_GAP = 52.0, 8.0


def _hud_effects(state, flow, game, manifest, screen) -> None:
    active = list(getattr(game, "effects", ()) or ()) if game is not None else []
    names = [name for name, _left in active]
    for record in manifest.screens[screen].get("sprites", ()):
        key = record["key"]
        if key.startswith("effect:") and key[len("effect:"):] not in names:
            state.hidden.add(key)
            state.hidden.add(f"effect_{key[len('effect:'):]}_left")
    cursor = EFFECT_X0
    for name, left in active:
        record = manifest.sprite(screen, f"effect:{name}")
        if record is None:
            continue
        rect = record["rect"]
        state.sprites[f"effect:{name}"] = (cursor, rect[1], rect[2], rect[3])
        label = _spec(manifest, screen, f"effect_{name}_left")
        if label is not None:
            shift = cursor - rect[0]
            state.extra.append(([label["rect"][0] + shift, label["rect"][1],
                                 label["rect"][2], label["rect"][3]],
                                label["runs"], [f"{left:.0f}s"],
                                label.get("al", "left")))
            state.hidden.add(f"effect_{name}_left")
        cursor += rect[2] + EFFECT_GAP
    _ = flow


# ---- kết quả đơn -------------------------------------------------------
def _result(state, flow, manifest, win: bool, accounts=None) -> None:
    screen = F.WIN if win else F.LOSE
    level = flow.level
    result = flow.last_result or {}
    found = int(result.get("found", 0))
    total = int(result.get("total", flow.bugs()))
    score = int(result.get("score", 0))
    state.pills["level_pill"] = (f"LEVEL {level} · {flow.level_name().upper()}",
                                 "asian" if level == 5 else "normal")
    if win:
        state.texts["stamp_line"] = (f"LEVEL {level} · {flow.level_name().upper()}"
                                     f" · {time.strftime('%d.%m.%Y')}")
        state.texts["caption"] = f"{found}/{total} bugs turned to dust"
    rows = []
    if not win:
        rows.append(("Bugs left", f"{total - found} / {total}", False))
    # Số 0 không mang dấu: "+0" đọc như một khoản cộng, mà hết giờ thì không
    # cộng được gì cả.
    left = int(result.get("time_left", 0))
    points = left * level
    rows.append((f"Time left {left} s × level {level}",
                 f"+{points}" if points > 0 else str(points), False))
    for item, count in (result.get("used") or {}).items():
        weight = F.ITEM_WEIGHT.get(item, 0) * int(count)
        rows.append((f"{item} ×{count} · weight {F.ITEM_WEIGHT.get(item, 0)}",
                     f"-{weight}", True))
    if level == 5 and result.get("overtime_s"):
        penalty = int(round(result["overtime_s"] * 5))
        rows.append((f"Overtime {mmss(result['overtime_s'])} · penalty",
                     f"-{penalty}", True))
    for index, (label, value, negative) in enumerate(rows):
        _extra_row(state, manifest, screen, "row_label#0", index, label)
        _extra_row(state, manifest, screen, "row_value#0", index, value,
                   DANGER if negative else None)
    _hide_spare_rows(state, manifest, screen, ("row_label", "row_value"),
                     len(rows))
    state.texts["score"] = str(score)
    if win:
        state.colours["score"] = SAFE
    else:
        state.colours["score"] = DANGER if score < 0 else INK
    from src.ui_album import board

    rank = board.rank_of(accounts, flow.account_key)
    if win and rank is not None and rank <= 10:
        state.sprites["rank_up"] = True
        state.texts["rank_up_text"] = f"You're now #{rank} on the Solo board"
    else:
        state.hidden |= {"rank_up", "rank_up_text"}


def _hide_spare_rows(state, manifest, screen, prefixes, count) -> None:
    """Ẩn những ô dòng có trong manifest mà ván này không dùng tới."""
    for record in manifest.screens[screen].get("texts", ()):
        key = record["key"]
        for prefix in prefixes:
            if key.startswith(f"{prefix}#") and int(key.split("#")[1]) >= count:
                state.hidden.add(key)


# ---- PvP ---------------------------------------------------------------
def _pvp_lobby(state, flow, now, manifest=None, painter=None,
               accounts=None) -> None:
    screen = flow.screen
    state.texts["eyebrow"] = (f"PVP · {flow.mode.upper()}"
                              if screen == F.PVP_INVITE else
                              f"PVP · {flow.mode.upper()} · LEVEL {flow.level}")
    name = flow.player or "guest"
    for key in ("you_name", "host_name"):
        state.texts[key] = _fit_name(manifest, painter, screen, key, name)
    for key in ("you_initial", "host_initial"):
        state.texts[key] = _initial(name)
    if screen == F.PVP_SEARCH:
        state.texts["search_time"] = mmss(now - flow.search_started_s)
        items = ", ".join(flow.items) or "no items"
        from src.ui_album import board

        rank = board.rank_of(accounts, flow.account_key)
        state.texts["you_sub"] = (f"Solo rank #{rank} · {items}" if rank
                                  else f"Solo unranked · {items}")
    elif screen == F.PVP_INVITE:
        state.texts["room_code"] = f"PM-{(flow.room_code or '0000')[:4]}"
        state.texts["join_typed"] = flow.join_typed.ljust(4, "_")
        state.focus = "field_join" if flow.focus == "field_join" else None
    elif screen == F.PVP_FOUND:
        state.texts["countdown"] = str(max(1, 3 - int(now - flow.found_at_s)))
        # TASK_accounts_1119: NETWORK-SUPPLIED — the opponent's own typed
        # display name, arrives over the wire with no length guarantee
        # from this side. Fitting it is the one case here that is not
        # just cosmetic.
        opp_name = flow.pvp_opponent_name or "Opponent"
        state.texts["opp_name"] = _fit_name(manifest, painter, screen, "opp_name", opp_name)
        state.texts["opp_initial"] = _initial(opp_name)
        # TASK_pvp_fix_1116: tính lại mỗi khung (app._pvp_lobby_message),
        # không phải một toast F.say 30 s — nếu không thì "đè" lên giá trị
        # chung ở đầu state_for() cũng không giúp gì, vì chuỗi đã tính SẴN
        # là rỗng khi không còn gì để nói (xong rồi/chưa bắt đầu).
        if flow.pvp_lobby_message:
            state.message = flow.pvp_lobby_message


def _pvp_result(state, flow, manifest=None, painter=None) -> None:
    """
    TASK_pvp_lan_1116: dữ liệu THẬT — ``flow.last_result`` được ``app.py``
    đổ từ thông điệp ``end`` của server (``winner``/``scores``/``rows``/
    ``opponent_rows``/``opp_name``), không còn số mô phỏng.
    """
    screen = F.PVP_RESULT
    result = flow.last_result or {}
    you = int(round(float(result.get("score", 0))))
    opponent = int(round(float(result.get("opponent_score", 0))))
    winner = result.get("winner")
    draw = winner == "draw"
    won = winner == "you" if winner is not None else you >= opponent
    state.texts["title"] = "Draw" if draw else ("Victory!" if won else "Defeat")
    state.colours["title"] = SAFE if won else DANGER
    state.texts["eyebrow"] = f"PVP · {flow.mode.upper()} · LEVEL {flow.level}"
    name = flow.player or "guest"
    state.texts["you_name"] = _fit_name(manifest, painter, screen, "you_name", name)
    state.texts["you_initial"] = _initial(name)
    state.texts["you_score"] = str(you)
    # TASK_accounts_1119: NETWORK-SUPPLIED, same as _pvp_lobby's opp_name.
    opp_name = result.get("opp_name") or "Opponent"
    state.texts["opp_name"] = _fit_name(manifest, painter, screen, "opp_name", opp_name)
    state.texts["opp_initial"] = _initial(opp_name)
    state.texts["opp_score"] = str(opponent)
    state.pills["you_pill"] = ("won this match" if won else "runner-up",
                               "won" if won else "lost")
    state.pills["opp_pill"] = ("runner-up" if won else "won this match",
                               "lost" if won else "won")
    if won:
        state.sprites["crown"] = True
    else:
        state.hidden.add("crown")
    record = manifest.sprite(screen, "sel:winner")
    if record is not None:
        state.sprites["sel:winner"] = _on_target(record,
                                                 "you" if won else "opponent")
    for side, rows in (("you", _pvp_rows(result.get("rows"))),
                       ("opp", _pvp_rows(result.get("opponent_rows")))):
        for index, (label, value, negative) in enumerate(rows):
            _extra_row(state, manifest, screen, f"{side}_row_label#0", index,
                       label)
            _extra_row(state, manifest, screen, f"{side}_row_value#0", index,
                       value, DANGER if negative else None)
        _hide_spare_rows(state, manifest, screen,
                         (f"{side}_row_label", f"{side}_row_value"), len(rows))


#: Hai dòng đầu server luôn gửi trước — "Bugs killed", "Time left"/"Bugs
#: alive"/"Hunt lasted (s)" — không phải vật phẩm nên không tô đỏ.
_PVP_FIXED_ROWS = 2


def _pvp_rows(rows) -> list:
    """``result["rows"]``/``["opponent_rows"]`` đã là (nhãn, giá trị) do
    server tính sẵn (``src/net/server.py:_end_match``) — ở đây chỉ quyết
    định dòng nào tô đỏ (vật phẩm trừ điểm)."""
    return [(str(label), str(value), index >= _PVP_FIXED_ROWS)
            for index, (label, value) in enumerate(rows or ())]


# ---- bảng xếp hạng -----------------------------------------------------
def _board(state, flow, manifest=None, painter=None, accounts=None) -> None:
    """Real leaderboard: every account on this computer, ranked by lifetime
    points (``board.py``). Without a manifest (static shots) it keeps the
    baked sample art."""
    if manifest is None:
        return
    from src.ui_album import board

    board.fill_state(state, flow, manifest, painter, accounts)


#: TASK_ui_sakuga_1117 Phase 5: ``tab_Solo`` targets ``F.LEVELS``
#: (``F.TABS``), but the nav bar also renders — still highlighted — on
#: ``PHOTOS``/``LOADOUT``/``TILT``, the rest of the solo flow upstream of
#: a match. The other three tabs' screens are a straight 1:1 with
#: ``F.TABS``, so only "Solo" needs its own wider set.
_SOLO_TAB_SCREENS = frozenset({F.LEVELS, F.PHOTOS, F.LOADOUT, F.TILT})


def _selected_hotspots(screen: str, flow) -> set:
    """Hotspot names that are the CURRENT selection on ``screen`` — the
    active tab/level/weapon/display/theme — for ``V3State.selected``
    (Phase 5's halved hover-ring alpha on an already-selected row)."""
    selected: set = set()
    for name in F.TABS:
        in_section = (screen in _SOLO_TAB_SCREENS if name == "tab_Solo"
                     else screen == F.TABS[name])
        if in_section:
            selected.add(name)
    if screen == F.LEVELS:
        selected.add(f"level_row_{flow.level}")
    if screen == F.LOADOUT:
        selected.add(f"weapon_{flow.weapon_name()}")
    if screen == F.SETTINGS:
        selected.add("display_window" if flow.display == "window" else "display_full")
        selected.add("theme_night" if flow.theme == "night" else "theme_day")
    return selected


# ---- điều phối ---------------------------------------------------------
def state_for(manifest, painter, flow, game=None, aim=None, accounts=None) -> V3State:
    """Bộ giá trị sống cho màn đang mở. ``aim`` là tâm ngắm (toạ độ thiết kế).
    ``accounts`` (TASK_accounts_1119): dữ liệu kho tài khoản hiện có, chỉ
    màn đăng nhập/đăng ký đọc tới (kiểm tên đã bị lấy chưa)."""
    now = time.perf_counter()
    state = V3State()
    screen = flow.screen
    state.message = (flow.message if now < flow.message_until_s else "")
    state.confirm = flow.shared_album_pending
    _dialogs(state, flow)
    state.selected = _selected_hotspots(screen, flow)
    if screen in F.LOCKED:
        _sign_in(state, flow, accounts)
    elif screen == F.MENU:
        _menu(state, flow, manifest, painter)
    elif screen == F.SETTINGS:
        _settings(state, flow)
    elif screen == F.OFFLINE:
        _offline(state, flow, now)
    elif screen == F.LEVELS:
        _levels(state, flow, manifest)
    elif screen == F.PHOTOS:
        _photos(state, flow)
    elif screen == F.LOADING:
        _loading(state, flow)
        state.photo = flow.photo
    elif screen == F.LOADING_READY:
        _bug_note(state, flow.bug_hold, flow.bugs(), wide=True)
    elif screen == F.LOADOUT:
        _loadout(state, flow, manifest, painter)
    elif screen == F.TILT:
        _tilt(state, flow, manifest)
    elif screen in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE):
        _hud(state, flow, game, manifest, now)
        # Chỉ màn nào BẮN được mới có tâm ngắm; màn con bọ (PVP_HIDE) thì
        # người chơi đang trốn, không có súng.
        if screen in (F.PLAY, F.PVP_HUNT):
            state.aim = aim
            state.aim_ready = _aim_ready(game, now)
    elif screen == F.WIN:
        _result(state, flow, manifest, win=True, accounts=accounts)
        state.photo = flow.photo
    elif screen == F.LOSE:
        _result(state, flow, manifest, win=False, accounts=accounts)
        state.photo = flow.photo
    elif screen in (F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND):
        _pvp_lobby(state, flow, now, manifest, painter, accounts)
    elif screen == F.PVP_RESULT:
        _pvp_result(state, flow, manifest, painter)
    elif screen == F.BOARD:
        _board(state, flow, manifest, painter, accounts)
    return state
