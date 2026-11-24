"""
Giao diện v3 ("v2 · Redesign"): nạp ``assets/ui_v3/manifest.json`` và vẽ.

Thứ tự vẽ là **hợp đồng** của bộ asset (README mục "Draw order"):
``base`` → ``photos`` → ``top`` → ``bars`` → ``pips`` → ``pills`` →
``sprites`` → ``texts`` → hover/focus. ``assets/ui_v3/tools/preview.py`` là
bản tham chiếu bằng PIL của đúng thứ tự ấy, và ``expected/*.jpg`` là đầu ra
của nó; mọi phép tính chữ/thanh/viên ở đây **chép theo** preview.py, không tự
nghĩ lại — lệch một pixel là trượt ngưỡng 24 của ``tools/compare.py``.

Khác với lớp vẽ cũ (``draw.py``), module này không làm việc PIL nào trong
đường chạy mỗi khung: chữ, viên thuốc, đầu thanh bo góc và pip đều dựng một
lần rồi nằm trong bộ đệm texture (luật F4). ``base``, ``top`` và sprite của
một màn nạp **một lần cho mỗi màn**.

TASK_arch_refactor_1120 R4: this module used to hold the painter, hotspot/
hit-testing, text-field editing and every per-screen draw helper too
(2479 lines) — those moved to ``v3_painter.py``, ``v3_hit.py``,
``v3_text_field.py`` and ``v3_screen_draw.py``. What is left: the manifest
loader, the per-frame draw STATE, and ``draw_screen`` itself (the
dispatcher that calls into the other four, in the draw-order above).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from src.ui_album.hud_cards import element_card_map
from src.ui_album.v3_screen_draw import (
    HUD_CARD_SCREENS,
    _draw_aim,
    _draw_bars,
    _draw_choose_photo_extra,
    _draw_confirm_dialog,
    _draw_covers,
    _draw_crit_impact,
    _draw_forgot_dialog,
    _draw_hit_markers,
    _draw_hp_bars,
    _draw_hud_top_cards,
    _draw_pills,
    _draw_pips,
    _draw_photos,
    _draw_recovery_dialog,
    _draw_remember_switch,
    _draw_rings,
    _draw_scope_mini_readout,
    _draw_scope_reticle,
    _draw_scope_vignette,
    _draw_shots,
    _draw_sprites,
    _draw_texts,
    _draw_toast,
)

if TYPE_CHECKING:
    from src.ui_album.v3_painter import V3Painter

logger = logging.getLogger(__name__)

#: Thư mục bộ asset v3, tính từ gốc project.
ASSET_DIR = Path("assets") / "ui_v3"

#: Không gian thiết kế của bộ v3. Mọi ``rect`` trong manifest nằm trong hệ này.
DESIGN_SIZE = (1280.0, 720.0)

#: Màu chữ mặc định khi ``runs[].col`` rỗng (preview.py dùng đúng hằng này).
INK = "#24415F"

#: Vòng focus bàn phím: 2,5 px, lùi ra 3 px (``v3_hover.py`` dùng cả hai).
FOCUS_PX, FOCUS_INSET = 2.5, 3.0


class V3Manifest:
    """Bản kê v3 đã nạp. Chỉ đọc; không giữ trạng thái vẽ nào."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root) / ASSET_DIR
        path = self.root / "manifest.json"
        self.data = json.loads(path.read_text(encoding="utf-8"))
        version = int(self.data.get("version", 0))
        if version != 3:
            raise ValueError(f"{path}: can manifest version 3, doc duoc {version}")
        self.image_scale = float(self.data.get("image_scale", 1.5))
        self.design_size = tuple(self.data.get("design_size", DESIGN_SIZE))
        self.screens = self.data["screens"]
        self.colors = self.data.get("colors", {})
        # ``FontBook`` đọc đúng hai thuộc tính này của bản kê.
        self.fonts = self.data.get("fonts", {})
        self.sprites = {}
        missing = self.validate()
        logger.info("UI v3: %d man, %d sprite, thieu %d file",
                    len(self.screens), self.sprite_count(), len(missing))
        if missing:
            logger.warning("UI v3: thieu file %s", missing[:5])

    def screen_names(self) -> list[str]:
        """Tên 23 màn, theo thứ tự trong manifest."""
        return list(self.screens)

    def sprite_count(self) -> int:
        return sum(len(s.get("sprites", ())) for s in self.screens.values())

    def path(self, relative: str | None) -> Path | None:
        return None if not relative else self.root / relative

    def sample_photo(self) -> Path | None:
        return self.path(self.data.get("sample_photo"))

    def text(self, screen: str, key: str) -> dict | None:
        """Bản ghi text ``key`` của màn ``screen``; ``None`` nếu không có."""
        for record in self.screens.get(screen, {}).get("texts", ()):
            if record["key"] == key:
                return record
        return None

    def sprite(self, screen: str, key: str) -> dict | None:
        """Bản ghi sprite ``key`` của màn ``screen``; ``None`` nếu không có."""
        for record in self.screens.get(screen, {}).get("sprites", ()):
            if record["key"] == key:
                return record
        return None

    def validate(self) -> list[str]:
        """File khai trong manifest mà không có trên đĩa."""
        missing = []
        for name, screen in self.screens.items():
            for layer in ("base", "top"):
                relative = screen.get(layer)
                path = self.path(relative)
                if path is not None and not path.exists():
                    missing.append(relative)
            for record in screen.get("sprites", ()):
                path = self.path(record.get("file"))
                if path is not None and not path.exists():
                    missing.append(record["file"])
            _ = name
        return missing




class V3State:
    """
    Mọi thứ đổi theo thời gian của một màn v3.

    Lớp vẽ cố tình **không biết** luật nào: ``v3_values.py`` quyết định sprite
    nào bật, chữ nào thay, thanh đầy bao nhiêu, rồi đổ vào đây. Nhờ vậy lượt
    chụp (``--ui-shots``) chỉ là một bộ giá trị khác, không phải một nhánh mã
    khác — mà README nói rõ ``expected/`` chính là chế độ chụp.
    """

    __slots__ = ("texts", "colours", "hidden", "sprites", "borrowed", "bars",
                 "pips", "pills", "photo", "camera_tex", "viewport_draw",
                 "hover", "focus", "caret", "alpha", "shot", "message", "wrap",
                 "extra", "mirror", "aim", "shots", "aim_ready", "confirm",
                 "text_rect", "fx_on", "fx_draw", "selected", "pressed",
                 "hover_fx", "aim_kick_scale", "hit_markers", "hp_bars",
                 "crit_impact_alpha", "scene_shake", "scope_zoom",
                 "card_alpha", "hud_hidden_alpha", "recovery_code", "forgot_open",
                 "forgot_focus", "remember", "caret_index", "covers")

    def __init__(self) -> None:
        #: ``key`` → chuỗi (thay run đầu) hoặc danh sách chuỗi (thay từng run).
        self.texts: dict = {}
        #: ``key`` → mã màu, hoặc danh sách mã màu theo từng run.
        self.colours: dict = {}
        self.hidden: set = set()
        #: ``key`` → ``True`` (vẽ ở ``rect`` của nó) hoặc ``rect`` đặt chỗ khác.
        self.sprites: dict = {}
        #: Sprite app tự đặt chỗ: ``nhãn`` → ``(màn, key, rect)``. Nhãn là duy
        #: nhất nên **một** sprite vẽ được nhiều lần (ba ô vật phẩm, ba hàng
        #: dấu kiểm), và ``màn`` khác màn đang mở thì là sprite vay — HUD lấy
        #: ``item_tile:*``/``weapon_tile:*`` của màn trang bị (README).
        self.borrowed: dict = {}
        #: ``key`` → tỉ lệ đầy 0–1.
        self.bars: dict = {}
        #: ``key`` → số pip đang bật, hoặc danh sách cờ, hoặc ``("n", trần)``.
        self.pips: dict = {}
        #: ``key`` → (chuỗi, tên kiểu).
        self.pills: dict = {}
        self.photo = None
        self.camera_tex = None
        self.viewport_draw = None
        self.hover: str | None = None
        self.focus: str | None = None
        #: ``key`` của ô chữ đang nhận chữ gõ — vẽ con trỏ sau nó.
        self.caret: str | None = None
        #: TASK_small_fixes_1120: character index of the caret INSIDE
        #: ``self.caret``'s field (0 = before the first character) —
        #: unused while ``self.caret`` is ``None``.
        self.caret_index: int = 0
        #: ``key`` → độ mờ 0–1 (ô vật phẩm cấp 1 vẽ 45 %).
        self.alpha: dict = {}
        self.shot = False
        self.message: str = ""
        self.wrap = True
        #: TASK_shared_album_1111: đường dẫn ảnh đang chờ xác nhận chia sẻ;
        #: "" (mặc định của ``str``) = không có hộp thoại nào đang mở.
        self.confirm: str = ""
        #: Nhãn sprite cần lật ngang (mũi tên nghiêng đầu khi góc > 0).
        self.mirror: set = set()
        #: Tâm ngắm ``(x, y)`` trong toạ độ thiết kế; ``None`` = không vẽ.
        self.aim = None
        #: Vết đạn ``(x, y, tuổi giây, trúng?)``, mới nhất ở cuối.
        self.shots: list = []
        #: Súng đã hồi tới đâu, 0–1. Dưới 1 thì tâm ngắm bung ra và mờ đi.
        self.aim_ready: float = 1.0
        #: Dòng chữ app tự đặt chỗ: ``(rect, runs, texts, align)``. Bản vẽ chỉ
        #: xuất MỘT thực thể cho mỗi kiểu ô vật phẩm / dòng điểm, còn ván chơi
        #: cần ba ô và số dòng thay đổi — chỗ lặp lại nằm ở đây, không phải
        #: trong manifest.
        self.extra: list = []
        #: Flat shapes drawn right after the art and before every dynamic
        #: element, to hide sample text baked into ``base.png`` (see
        #: ``board.py``): ``("rect", rect, hex)``, ``("rounded", rect, radius,
        #: hex)`` or ``("patch", rect, cache key, RGBA array)``. Design pixels.
        self.covers: list = []
        #: TASK_ui_sakuga_1117 bug A: ``key`` → ``rect`` đặt chỗ khác cho MỘT
        #: chữ sẵn có trong bản vẽ (khác ``extra``, vốn dùng khi cần NHIỀU
        #: bản của cùng một kiểu chữ). ``check_hint`` cần đúng việc này: dời
        #: theo hàng đang hỏi, giữ nguyên kiểu chữ/x.
        self.text_rect: dict = {}
        #: TASK_ui_sakuga_1117: Sakuga VFX on for this frame? Only gates
        #: the old green ``HIT_FLASH`` (shot marks always draw either way)
        #: — the real on/off switch for spawning/drawing effects lives in
        #: ``fx_draw`` below and in the caller that decides whether to
        #: spawn anything at all.
        self.fx_on: bool = False
        #: Zero-arg callback that draws the live Sakuga effects for this
        #: frame, or ``None``. Kept as a callback (not a direct dependency
        #: on ``FxLibrary``) so this module stays GL-texture-agnostic;
        #: ``app.py`` wires it to ``FxLibrary.draw(player)``.
        self.fx_draw = None
        #: TASK_ui_sakuga_1117 Phase 5: hotspot names that are the CURRENT
        #: selection (active tab/level/weapon/display/theme) — their hover
        #: ring draws at half alpha even while hovered, so hovering an
        #: already-selected row doesn't look identical to hovering a
        #: plain unselected one.
        self.selected: set = set()
        #: Left mouse button held down on ``hover`` right now (not just
        #: hovered) — the pressed tint/inset ring in ``v3_hover.py``.
        self.pressed: bool = False
        #: The persistent ``v3_hover.HoverFx`` instance (springs survive
        #: across frames; ``V3State`` itself does not) — ``None`` means
        #: "no hover feedback this frame" (``shot_state``'s screenshots).
        self.hover_fx = None
        #: TASK_game_feel_1118 P4 — all five default to "nothing kicked/
        #: shaking/marked right now"; ``v3_ui.py`` fills them in on
        #: PLAY/PVP_HUNT from its persistent ``ShotFeel``.
        self.aim_kick_scale: float = 1.0
        #: ``[(x, y, age_s, crit), ...]``.
        self.hit_markers: list = []
        #: TASK_weapons_1119: ``[(x, y, phần máu còn, độ mờ), ...]`` của mỗi
        #: con vừa bị bắn mà chưa chết — ``v3_values._hud`` đổ từ
        #: ``AlbumMatch.hp_bars``.
        self.hp_bars: list = []
        self.crit_impact_alpha: float = 0.0
        #: Design-px ``(dx, dy)`` the 3D scene viewport and the fx layer
        #: are shifted by — HUD panels/text never move with it.
        self.scene_shake: tuple = (0.0, 0.0)
        #: TASK_game_feel_1118 P6: Precision-gun scope zoom, 1.0 = not
        #: scoped (no vignette/reticle drawn) up to 2.5 at full zoom.
        self.scope_zoom: float = 1.0
        #: TASK_game_feel_fix_1118 Fix 10: card index (``hud_cards``) ->
        #: current alpha 0-1 — ``v3_ui.py`` owns the Springs across
        #: frames and fills this in each ``draw()`` for PLAY/PVP_HUNT/
        #: PVP_HIDE; missing index = 1.0 (not faded).
        self.card_alpha: dict = {}
        #: TASK_hud_fix_1119 Fix 2: H toggles the TARGET this eases
        #: toward (1.0 visible, 0.0 hidden) — session-only, never
        #: persisted. ``_card_fade`` multiplies it into every HUD
        #: card/dynamic-element's alpha except the always-visible
        #: sprites (crosshair, bite); ``_draw_sprites``'s borrowed loop
        #: and ``_draw_texts``'s ``state.extra`` loop apply it directly
        #: for item slots, which have no card of their own.
        self.hud_hidden_alpha: float = 1.0
        #: TASK_accounts_1119 Phase 6: the one-time recovery code dialog
        #: (``recovery_code`` non-empty = open, same ``""``-means-closed
        #: convention as ``confirm`` above) and the forgot-password
        #: dialog (``forgot_open`` + which of its OWN four fields has
        #: focus — separate from ``focus`` above, which stays the
        #: underlying screen's, same as the share-album confirm dialog
        #: leaves the screen under it alone). Both dialogs' field VALUES
        #: and any error message live in ``texts`` (keys "forgot_user"/
        #: "forgot_code"/"forgot_pass"/"forgot_pass2"/"forgot_error") —
        #: no new slot needed, nothing in the manifest ever uses those
        #: key names so the generic manifest-driven text draw never
        #: touches them; only the dialog's own draw function reads them.
        self.recovery_code: str = ""
        self.forgot_open: bool = False
        self.forgot_focus: str | None = None
        #: TASK_signin_toggles_1120: ``switch_remember``'s current value —
        #: ``base.png`` only ever drew the ON look, so this is read by
        #: ``_draw_remember_switch`` to paint the OFF look over it; ON
        #: itself needs no extra draw.
        self.remember: bool = False


def shot_state(manifest: V3Manifest, screen: str) -> V3State:
    """
    Trạng thái của **chế độ chụp** = đúng luật ``expected/`` trong README.

    Mọi chữ lấy ``sample``, mọi sprite ``default: true`` vẽ ở ``rect`` của nó,
    pip theo ``sample_on``, thanh theo ``sample_frac``, viên thuốc lấy kiểu
    đầu tiên, ô ảnh lấy ``sample/sample_room.png``, không hover/focus/caret.
    Riêng ``wrap`` phải TẮT: ``preview.py`` bỏ qua cờ đó, nên ``expected/``
    vẽ ``tip_text`` thành một dòng.
    """
    record = manifest.screens[screen]
    state = V3State()
    state.shot = True
    state.wrap = False
    state.photo = manifest.sample_photo()
    # TASK_signin_toggles_1120: ``switch_remember`` has no "sample"/
    # "default" field of its own (it isn't a sprite/pill) — ON is the
    # baked look in ``base.png``, so pin it ON here the same way every
    # other control's sample value is pinned, instead of falling through
    # to ``V3State``'s real-gameplay default (OFF) and drawing an overlay
    # ``expected/`` was never captured with.
    state.remember = True
    state.sprites = {s["key"]: True for s in record.get("sprites", ())
                     if s.get("default")}
    state.bars = {b["key"]: float(b.get("sample_frac", 0.0))
                  for b in record.get("bars", ())}
    state.pips = {p["key"]: list(p.get("sample_on", ()))
                  for p in record.get("pips", ())}
    state.pills = {p["key"]: (p.get("sample") or "",
                              next(iter(p["styles"]), None))
                   for p in record.get("pills", ())}
    return state



def draw_screen(painter: V3Painter, manifest: V3Manifest, screen: str,
                state: V3State) -> None:
    """Vẽ trọn một màn theo thứ tự lớp của README."""
    record = manifest.screens[screen]
    full = (0.0, 0.0, DESIGN_SIZE[0], DESIGN_SIZE[1])
    painter.layer(full, record.get("base"))
    _draw_photos(painter, manifest, record, state)
    # TASK_game_feel_fix_1118 Fix 4: the crit impact frame draws here —
    # BEFORE ``top.png``, not after. ``top.png`` is where the HUD CARD
    # BACKGROUNDS are actually baked (the dynamic numbers/bars/pips drawn
    # further below were always on top and stayed readable); drawing the
    # wash AFTER ``top.png`` (the 1118 version) painted it straight over
    # those opaque card backgrounds, visibly darkening them — exactly
    # what ``hud_layer_check.png``/G3 catches. Now the card art (drawn
    # next) covers the wash wherever a card sits, leaving it visible only
    # in the open scene area.
    #
    # ``_draw_scope_vignette`` below is left as-is (still after
    # ``top.png``, so it has the SAME card-darkening issue today) — out
    # of Fix 4's own scope, which names only the crit impact frame; the
    # scope's own HUD handling is Fix 10's "every card fades to 0 while
    # scoped", a bigger, deliberate redesign rather than a one-line
    # reorder like this one.
    _draw_crit_impact(painter, state)
    # TASK_game_feel_fix_1118 Fix 10: on a gameplay HUD screen, top.png
    # draws per-CARD (its own alpha per card — proximity/scope fade,
    # Fix 10.2/10.3) instead of one single full-canvas quad. The
    # designed HUD (cards + every dynamic element) is the only gameplay
    # layout now — TASK_hud_fix_1119 Fix 1 removed the small-pills
    # alternative entirely.
    is_hud = screen in HUD_CARD_SCREENS
    card_map = element_card_map(manifest, screen) if is_hud else {}
    if is_hud:
        _draw_hud_top_cards(painter, manifest, screen, record, state)
    else:
        painter.layer(full, record.get("top"))
    _draw_scope_vignette(painter, state)
    _draw_covers(painter, state)
    _draw_bars(painter, record, state, card_map)
    _draw_pips(painter, record, state, card_map)
    _draw_pills(painter, record, state, card_map)
    _draw_sprites(painter, manifest, screen, record, state, card_map)
    _draw_remember_switch(painter, screen, record, state)
    _draw_texts(painter, manifest, screen, record, state, card_map)
    # Fix 10.3: drawn LAST of the HUD layer, on top of the (now-fading)
    # cards — ammo/timer/kills must stay legible for the whole fade, not
    # just once alpha hits exactly 0.
    _draw_scope_mini_readout(painter, state)
    _draw_choose_photo_extra(painter, screen, state)
    _draw_shots(painter, state)
    _draw_hp_bars(painter, state)
    _draw_aim(painter, state)
    _draw_scope_reticle(painter, state)
    _draw_hit_markers(painter, state)
    _draw_rings(painter, manifest, screen, record, state)
    _draw_toast(painter, state, screen)
    # TASK_ui_sakuga_1117: above HUD/toast (so a hit's burst reads clearly
    # over gameplay) but below the share-album confirm dialog (so an open
    # dialog's own ui_tap effect, drawn separately by the dialog's own
    # click handling, is never occluded by an unrelated effect still
    # playing underneath it).
    if state.fx_draw is not None:
        state.fx_draw()
    _draw_confirm_dialog(painter, state)
    _draw_recovery_dialog(painter, state)
    _draw_forgot_dialog(painter, state)


