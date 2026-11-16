"""
Giá trị sống của các ô động: trạng thái trò chơi / giao diện → ``key: giá trị``.

Khoá ``<tên>:frac`` là phần của thanh (0–1) — ``screen.draw_record`` nhân bề
rộng với nó. Khoá nào không có ở đây thì giữ nguyên chữ mẫu trong manifest,
nên mọi màn vẫn đầy đủ nội dung tĩnh để chụp ảnh luận văn.
"""

from __future__ import annotations

import time

from src.ui_album import flow as F


def mmss(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def _head_text(state) -> str:
    """Nhãn trạng thái đầu: mô phỏng / tốt / mất dấu."""
    if state.tracker_simulated:
        return "● Đầu: mô phỏng"
    return "● Đầu: tốt" if state.face_seen else "● Mất dấu đầu"


def _head_colour(state) -> str:
    if state.tracker_simulated:
        return "warning"
    return "success" if state.face_seen else "danger"


def _level_tab(state) -> str:
    """Nhãn góc trái: đơn thì "CẤP n · TÊN", PvP thì tên chế độ."""
    if state.screen in (F.PVP_HUNT, F.PVP_HIDE):
        return f"PVP · {state.mode.upper()}"
    return f"CẤP {state.level} · {F.LEVEL_TABLE[state.level][0].upper()}"


def _item_values(state, game) -> dict:
    """Ô vật phẩm: số phím · TÊN ×số, lấy từ trang bị đã chọn."""
    used = getattr(game, "used", {}) if game else {}
    out = {}
    for slot in range(3):
        key = f"items#{slot * 2 + 1}"
        if slot < len(state.stickers):
            name = state.stickers[slot]
            left = max(0, 1 - int(used.get(name, 0)))
            out[key] = f"{slot + 1} · {name.upper()} ×{left}"
        else:
            out[key] = ""
    return out


def _play_values(state, game) -> dict:
    """Màn chơi: đồng hồ, số bọ, máu, stamina, lớp, đạn, đầu."""
    name, bugs, seconds, _slots, _attack = F.LEVEL_TABLE[state.level]
    left = seconds if game is None else game.time_left_s
    total = bugs if game is None else game.bugs_total
    found = 0 if game is None else game.bugs_found
    values = {
        "timer": mmss(left),
        "kills": str(found), "kills_total": f"/{total}",
        "hp_label": f"MÁU {int(round(game.hp))}" if game else "MÁU 100",
        "hp_bar:frac": (game.hp / 100.0) if game else 1.0,
        "st_label": f"STAMINA {int(round(game.stamina))}%" if game else "STAMINA 100%",
        "st_bar:frac": (game.stamina / 100.0) if game else 1.0,
        "layer": game.layer_label if game else "– / 3",
        "ammo": f"{game.weapon_name.upper()} · ĐẠN {game.ammo}" if game
                else "SÚNG CHÍNH XÁC · ĐẠN 10/10",
        # Ba trạng thái, không phải hai: tracker chạy mô phỏng thì con số
        # không đến từ đầu người nào cả, nói "tốt" là nói sai.
        "head": _head_text(state), "head:col": _head_colour(state),
        "level_tab": _level_tab(state),
        "level_tab#0": _level_tab(state),
        "kills_bar:frac": (found / total) if total else 0.0,
    }
    values.update(_item_values(state, game))
    eaten = 0 if game is None else game.eaten_pct
    # Chú thích ảnh là một nhóm hai dòng: ngày, rồi phần ảnh đã bị gặm.
    values["photo_caption#0"] = time.strftime("%d/%m/%Y")
    values["photo_caption#1"] = f"đã bị gặm {eaten:.0f}%"
    return values


#: Ván giả cho lượt chụp ảnh (``--album-shots``): một ván THẮNG, còn 42 giây,
#: đã dùng Thuốc giải ×1 (trọng số 5) → 42 − 5 = **37 điểm**. Con số này là
#: **dữ liệu chụp ảnh**, không phải kết quả đo — README nói rõ.
SHOT_MATCH = {"time_left_s": 42.0, "bugs_found": 6, "bugs_total": 6,
              "hp": 78.0, "stamina": 55.0, "score": 37,
              "items": ["Thuốc giải"], "used": {"Thuốc giải": 1}}


def shot_values(state) -> dict:
    """Giá trị cho các màn cần trạng thái ván khi chụp ảnh."""
    return {"timer": mmss(SHOT_MATCH["time_left_s"]),
            "kills": str(SHOT_MATCH["bugs_found"]),
            "kills_total": f"/{SHOT_MATCH['bugs_total']}",
            "kills_bar:frac": 1.0,
            "hp_label": f"MÁU {SHOT_MATCH['hp']:.0f}",
            "hp_bar:frac": SHOT_MATCH["hp"] / 100.0,
            "st_label": f"STAMINA {SHOT_MATCH['stamina']:.0f}%",
            "st_bar:frac": SHOT_MATCH["stamina"] / 100.0}


def values_for(state, game=None) -> dict:
    """Bộ giá trị cho màn đang mở."""
    now = time.perf_counter()
    screen = state.screen
    if screen in (F.LOGIN, F.SIGNUP, F.LOGIN_ERR):
        # Con trỏ "|" ở ô đang chọn: không có nó thì gõ xong không biết chữ
        # vào ô nào (người dùng 06/11: "nhập tên vô cùng khó chịu").
        def field(name: str, text: str) -> str:
            caret = "|" if state.focus == name else ""
            return (text + caret) or " "

        return {"field:TÊN NGƯỜI CHƠI": field("o_TÊN NGƯỜI CHƠI", state.player),
                "field:MẬT KHẨU": field("o_MẬT KHẨU",
                                        "•" * len(state.password)),
                "field:NHẬP LẠI MẬT KHẨU": field("o_NHẬP LẠI MẬT KHẨU",
                                                 "•" * len(state.password2))}
    if screen in (F.PIPE0, F.PIPE, F.PIPE100):
        pct = 0.0 if screen == F.PIPE0 else (
            100.0 if screen == F.PIPE100 else state.pipeline_pct)
        return {"pct": f"{pct:.0f}", "pct_bar:frac": pct / 100.0}
    if screen == F.PVP_SEARCH:
        return {"search_time": mmss(now - state.search_started_s)}
    if screen == F.PVP_FOUND:
        left = max(1, 3 - int(now - state.found_at_s))
        return {"countdown": str(left)}
    if screen == F.PVP_INVITE:
        return {"room_code": f"PM-{state.room_code[:4] or '0000'}"}
    if screen in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE):
        return _play_values(state, game)
    return {}


def hidden_for(state, game=None) -> set:
    """Khoá phải **ẩn** ở màn đang mở (README: stamina chỉ có ở Châu Á)."""
    hidden = set()
    if state.screen in (F.PLAY, F.PVP_HUNT) and state.level != 5:
        # v2: giấu cả RÃNH, không chỉ nhãn — rãnh stamina vẫn hiện ở cấp 1.
        hidden |= {"st_label", "st_bar", "st_track"}
    if state.screen in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE):
        if not state.stickers:
            hidden.add("items")
        else:
            for slot in range(len(state.stickers), 3):
                hidden |= {f"items#{slot * 2}", f"items#{slot * 2 + 1}"}
    if game is None or not getattr(game, "effects", None):
        hidden.add("effects")
    if game is None or not getattr(game, "missed_until_s", 0.0) > time.perf_counter():
        hidden.add("miss")
    return hidden
