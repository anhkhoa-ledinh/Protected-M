"""
Cầu nối giữa app và giao diện v3: một đối tượng, ba việc.

``handle_input`` (chuột/phím) → ``tick`` (nhịp mô phỏng) → ``draw`` (vẽ màn
đang mở) — y lối của ``ui.py`` cho bộ asset cũ, để ``app.py`` chỉ phải chọn
một trong hai đối tượng chứ không phải hai đường mã.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

from OpenGL import GL

from src.account import crypto as account_crypto
from src.account import store as account_store
from src.config import prefs as _prefs
from src.fx.anim import Spring
from src.fx.library import FxLibrary
from src.fx.player import FxPlayer
from src.fx.spec import load_spec
from src.ui_album import v3_routes, v3_values
from src.ui_album import v3_flow as F
from src.ui_album.hud_cards import card_index_for_point, cards_for_screen
from src.ui_album.scope import ScopeState
from src.ui_album.shot_feel import ShotFeel
from src.ui_album.v3 import V3Manifest, V3Painter, draw_screen, hit_test, shot_state
from src.ui_album.v3_hover import HoverFx, cursor_for

logger = logging.getLogger(__name__)

#: TASK_ui_sakuga_1117 Phase 6 — hotspot names/prefixes that get a plain
#: ``ui_tap`` on click (the PRIMARY button of the screen gets
#: ``ui_confirm`` instead; anything not listed here gets nothing —
#: fields/slider/dropdown/dropzone/switch/links/icons/recent_memory
#: already have Phase 5's hover/press feedback, a tap on top would be
#: noise on a text-entry/drag control).
_UI_TAP_PREFIXES = ("btn_", "tab_", "seg_", "mode_", "display_", "theme_",
                    "level_row_", "weapon_", "sticker_", "slot_")
_UI_TAP_EXACT = frozenset({"leave", "continue"})
#: A tap on something this tall/wide shows at the CLICK point, not the
#: hotspot centre (a 300px-wide ``mode_`` card, or the full-screen
#: ``continue`` hotspot) — centred would land far from where the player
#: actually looked.
_UI_TAP_BIG_HEIGHT_PX, _UI_TAP_BIG_WIDTH_PX = 100.0, 360.0

#: "kills" counter text per screen, the wordless star_pop's anchor on
#: every kill. ``PVP_HIDE`` has none — the hider doesn't shoot.
_KILL_STAR_TEXT = {F.PLAY: "kills", F.PVP_HUNT: "kills_you"}

#: Loading screen (``_step_loading_fx``): the progress bar's own design-px
#: span, so the edge/word effects track the fill exactly.
_LOAD_BAR_X0, _LOAD_BAR_W, _LOAD_BAR_Y = 48.0, 1184.0, 665.0
_LOAD_EDGE_INTERVAL_S = 0.25
_LOAD_DONE_XY = (1232.0, 665.0)
_GOGO_X_RANGE = (200.0, 1080.0)
_GOGO_Y = 600.0
_STEP_DOT_X = (53.0, 349.0, 645.0, 941.0)
_STEP_DOT_Y = 687.0

#: Tilt checklist marker centres (Phase 2's ``CHECK_ROWS_Y`` is the row's
#: TEXT baseline; this is the task's own, slightly different, marker-glyph
#: centre — kept as its own constant rather than reusing one for the other).
_TILT_STAR_X, _TILT_STAR_Y0, _TILT_STAR_DY = 783.0, 497.0, 32.0

#: TASK_game_feel_1118 P3: the OLD anchor, (1084, 370), was the exact
#: centre of the ``score`` text rect ((984, 308, 200, 125) on
#: v3_09a_result_win) — the burst's "pon" word always landed right on
#: the score digits. Moved straight up, clear of both the score rect AND
#: the row_value text just above it — checked with
#: ``src/fx/layout.py``'s rect-intersection helpers (see
#: ``tests/test_fx_layout.py``), not eyeballed.
_WIN_SCORE_XY = (1084.0, 220.0)
_PVP_CROWN_XY = (640.0, 50.0)
_RANK_UP_DELAY_S = 0.5

_TOAST_XY = (640.0, 648.0)
_TOAST_PHOTOS_XY = (640.0, 596.0)

#: TASK_game_feel_fix_1118 Fix 10.2/10.3: HUD card fade — unscoped
#: proximity (crosshair within a card's rect, grown by this much) drops
#: that one card to this alpha; scoped drops EVERY fadeable card to 0.
#: Spring constants are the task's own numbers, shared by both targets —
#: only ``.to`` differs.
_HUD_FADE_GROW_PX = 48.0
_HUD_FADE_PROXIMITY_ALPHA = 0.22
_HUD_FADE_SPRING_K, _HUD_FADE_SPRING_D = 420.0, 26.0


class V3UI:
    """Giao diện v3 gắn vào một cửa sổ GLFW đã có context."""

    def __init__(self, root: Path, shader_dir: Path, options, window=None) -> None:
        self.manifest = V3Manifest(root)
        self.painter = V3Painter(self.manifest, shader_dir)
        self.state = F.V3Flow()
        #: TASK_game_feel_fix_1118 Fix 10.4: loaded once at startup, like
        #: ``screen_shake``/``fx`` below — ``toggle_hud_compact`` and
        #: ``app.py`` own writing it back to disk from then on.
        self.state.hud_compact = _prefs.hud_compact_enabled(root)
        self.options = options
        self.game = None
        self.mouse = (0.0, 0.0)
        #: TASK_ui_sakuga_1117 Phase 5: left button held RIGHT NOW (press
        #: feedback needs hold state, not the edge-triggered clicks
        #: ``app.py`` otherwise drains via ``pop_clicks()``).
        self.mouse_down = False
        self.pending = ""
        #: OS cursor shape changes (``window.set_cursor_shape``) — optional,
        #: so standalone/test construction without a real window still works.
        self._window = window
        self._hover_fx = HoverFx()
        self._hover_last_t: float | None = None
        #: TASK_game_feel_1118 P4: crosshair kick/shake/hit marker/crit
        #: impact — survives the whole session, like ``_hover_fx``.
        self._shot_feel = ShotFeel()
        self._shot_feel.shake_enabled = _prefs.screen_shake_enabled(root)
        #: TASK_game_feel_1118 P6: Precision-gun scope — zoom spring,
        #: right-mouse hold, finer-aim crosshair. ``app.py``'s
        #: ``_tick_scope`` drives it via the methods below.
        self._scope = ScopeState()
        #: TASK_game_feel_fix_1118 Fix 10.2/10.3: one Spring per fadeable
        #: HUD card on the CURRENT gun screen (``_tick_hud_fade``) — reset
        #: whenever the screen changes, since PLAY/PVP_HUNT can have
        #: different card layouts.
        self._hud_springs: dict[int, Spring] = {}
        self._hud_springs_screen: str | None = None
        #: Khung webcam đã lên GPU: ô ``camera`` của màn cài đặt / thử nghiêng.
        self._camera_entry = None
        self._camera_token = None
        # TASK_accounts_1119: kho tài khoản cục bộ — nạp MỘT LẦN ở đây,
        # giống `_fx_enabled`/`_shot_feel`: thuộc phiên giao diện, không
        # phải trạng thái màn (``V3Flow``). ``app.py`` không tự đọc/ghi
        # file tài khoản — chỉ gọi các phương thức công khai dưới đây.
        self._account_root = root
        account_load = account_store.load(root)
        self._account_data = account_load.data
        #: True đúng MỘT lần sau khi nạp, nếu cả ``accounts.json`` lẫn
        #: ``.bak`` đều hỏng — ``app.py`` đọc cờ này ở khung đầu để báo
        #: toast, rồi tự xoá cờ (xem ``pop_account_corrupt_toast``).
        self._account_corrupt = account_load.reset
        self._account_throttle = account_store.FailThrottle()
        #: ``(kind, holder)`` trong lúc một việc băm đang chạy trên luồng
        #: riêng — ``holder`` là dict mà luồng đó tự điền ``done``/
        #: ``result``/``error`` vào, đọc lại ở ``_poll_account_job``
        #: (gọi mỗi khung từ ``tick()``). Nút chính bỏ qua cú bấm lặp lại
        #: trong lúc ``account_busy`` là True.
        self._account_job: tuple | None = None
        #: TASK_accounts_1119 Phase 5: the guest-only toast fires once
        #: per PROCESS (not persisted — guest never persists anything),
        #: the first time a guest match actually finishes.
        self._guest_toast_shown = False
        if options.guest:
            self.state.player = "guest"
            self.state.account_key = account_crypto.GUEST_KEY
            self.state.go(F.MENU)
        else:
            # TASK_accounts_1119 Phase 4: a valid "remember me" session
            # signs in STRAIGHT to the menu, no sign-in screen — and
            # refreshes the expiry (a session that's actively used should
            # not quietly expire mid-way through a long play session).
            resumed_key = account_store.resolve_session(root, self._account_data)
            if resumed_key is not None:
                account_store.start_session(root, self._account_data, resumed_key)
                account_store.save(root, self._account_data)
                self.state.account_key = resumed_key
                self.state.player = self._account_data["accounts"][resumed_key]["display"]
                self._load_progress(resumed_key)
                self.state.go(F.MENU)
            else:
                # No session: pre-fill the username field with the last
                # account that signed in on this machine (nothing to
                # pre-fill if none ever has).
                last_key = self._account_data.get("last_account")
                last_account = (self._account_data["accounts"].get(last_key)
                                if last_key else None)
                if last_account is not None:
                    self.state.player = last_account["display"]
        # Móc đo: mở thẳng một màn để đếm FPS trên đúng màn đó (vd. màn chơi).
        forced = os.environ.get("HCPD_V3_SCREEN", "").strip()
        if forced and forced in self.manifest.screens:
            self.state.go(forced)

        # TASK_ui_sakuga_1117: tắt qua HCPD_NO_FX hoặc "fx": false trong
        # user_prefs.json (G6) — tắt thì KHÔNG đọc fx.json luôn, không chỉ
        # không vẽ, nên không texture nào của bộ này lên GPU.
        self._fx_enabled = _prefs.fx_enabled(root)
        self._fx_player = None
        self._fx_library = None
        self._fx_screen: str | None = None
        self._fx_last_t: float | None = None
        if self._fx_enabled:
            try:
                spec = load_spec(root)
            except (OSError, ValueError) as err:
                logger.warning("V3UI: khong nap duoc fx.json (%s) — tat hieu ung", err)
                self._fx_enabled = False
            else:
                self._fx_player = FxPlayer(spec)
                self._fx_library = FxLibrary(spec)

        # TASK_ui_sakuga_1117 Phase 6: bookkeeping for the per-tick/
        # per-screen fx triggers (loading bar, tilt checklist, toast,
        # WIN/PVP_RESULT entry) — see ``_step_fx_triggers``.
        self._fx_trigger_last_t: float | None = None
        self._fx_trigger_screen: str | None = None
        self._fx_load_edge_elapsed = 0.0
        self._fx_gogo_spawned = False
        self._fx_load_done_spawned = False
        self._fx_last_step_done = 0
        self._fx_last_tilt_checks: tuple | None = None
        self._fx_last_message_until = 0.0
        self._fx_win_entered_t: float | None = None
        self._fx_win_star_spawned = False
        self._fx_rank_up_spawned = False
        self._fx_pvp_result_spawned = False

    def initialize(self) -> None:
        self.painter.initialize()
        if self._fx_enabled and not self._fx_library.check_capability():
            self._fx_enabled = False
        if self._fx_enabled:
            # TASK_game_feel_1118 P3: decode+upload EVERY fx sheet/word
            # starting now, not lazily per-screen — see
            # ``FxLibrary.start_preload``'s docstring for why.
            self._fx_library.start_preload()

    #: Xem ``ui.AlbumUI.flow``: ``app.py`` dùng chung một bộ hàm cho hai bộ
    #: giao diện, phân biệt nhau đúng ở module này.
    flow = F

    @property
    def bug_theme(self) -> str:
        return F.BUG_THEME

    # ---- nhập liệu -------------------------------------------------------
    def design_mouse(self, x: float, y: float) -> tuple[float, float]:
        return self.painter.unmap(x, y)

    def hover(self) -> str | None:
        if self.state.shared_album_pending:
            return self._dialog_hover(*self.mouse)
        return hit_test(self.manifest, self.state.screen, *self.mouse)

    def _dialog_hover(self, x: float, y: float) -> str | None:
        """Hộp xác nhận chia sẻ che cả màn — hai nút của nó, không gì khác."""
        from src.ui_album.v3 import CONFIRM_NO_RECT, CONFIRM_YES_RECT

        for name, rect in (("btn_confirm_yes", CONFIRM_YES_RECT),
                          ("btn_confirm_no", CONFIRM_NO_RECT)):
            rx, ry, rw, rh = rect
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return name
        return None

    def click(self) -> str:
        """Bấm chuột trái tại vị trí hiện tại; trả hành động cho app."""
        return self._click(self.hover())

    def activate(self, name: str) -> str:
        """Kích hoạt một hotspot CỤ THỂ (phím tắt U/R…), bỏ qua vị trí chuột."""
        return self._click(name)

    def _click(self, name: str | None) -> str:
        """``on_click`` + (TASK_ui_sakuga_1117 Phase 6) a ui_tap/ui_confirm
        burst at the hotspot's PRE-click position — captured before
        ``on_click`` runs, since it may change ``self.state.screen`` out
        from under a post-click lookup."""
        if not name:
            self.pending = ""
            return ""
        screen = self.state.screen
        in_dialog = bool(self.state.shared_album_pending)
        rect = self._click_rect(screen, name, in_dialog)
        is_primary = (not in_dialog and
                     name == v3_routes.primary_button(self.manifest, self.state))
        pre_items = list(self.state.items) if hasattr(self.state, "items") else None
        action = v3_routes.on_click(self.state, name, time.perf_counter(),
                                    self.mouse)
        self.pending = action
        self._spawn_click_fx(name, screen, rect, in_dialog, is_primary, pre_items)
        return action

    def _click_rect(self, screen: str, name: str, in_dialog: bool):
        if in_dialog:
            from src.ui_album.v3 import CONFIRM_NO_RECT, CONFIRM_YES_RECT

            return {"btn_confirm_yes": CONFIRM_YES_RECT,
                   "btn_confirm_no": CONFIRM_NO_RECT}.get(name)
        from src.ui_album.v3 import hotspot_rect

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

    def _spawn_click_fx(self, name: str, screen: str, rect, in_dialog: bool,
                        is_primary: bool, pre_items) -> None:
        if not self._fx_enabled:
            return
        if in_dialog:
            if name in ("btn_confirm_yes", "btn_confirm_no") and rect is not None:
                self.spawn_fx("ui_tap", rect[0] + rect[2] * 0.5,
                             rect[1] + rect[3] * 0.5)
            return
        if rect is not None:
            cx, cy = rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5
            if is_primary:
                self.spawn_fx("ui_confirm", cx, cy)
            elif name in _UI_TAP_EXACT or any(
                    name.startswith(p) for p in _UI_TAP_PREFIXES):
                big = (rect[3] > _UI_TAP_BIG_HEIGHT_PX or
                      rect[2] > _UI_TAP_BIG_WIDTH_PX)
                x, y = self.mouse if big else (cx, cy)
                self.spawn_fx("ui_tap", x, y)
        # Vật phẩm vừa được GẮN (không phải BỎ RA) vào một ô loadout — sao
        # bật ở Ô NÓ VỪA VÀO, không phải ở chính con sticker đã bấm.
        if name.startswith("sticker_") and pre_items is not None:
            item = name[len("sticker_"):]
            if item not in pre_items and item in self.state.items:
                index = self.state.items.index(item)
                slot_rect = self._slot_rect(screen, index)
                if slot_rect is not None:
                    sx = slot_rect[0] + slot_rect[2] * 0.5
                    sy = slot_rect[1] + slot_rect[3] * 0.5
                    self.spawn_fx("star_pop", sx, sy, words=False)

    def key(self, char: str) -> None:
        F.type_char(self.state, char)

    def backspace(self) -> None:
        F.backspace(self.state)

    def focus_next(self) -> None:
        fields = {h["n"] for h in
                  self.manifest.screens[self.state.screen].get("hotspots", ())}
        F.focus_next(self.state, fields)

    def submit(self) -> str:
        """Enter = nút chính của màn đang mở (U2: "Enter = the primary button")."""
        name = v3_routes.primary_button(self.manifest, self.state)
        if not name:
            return ""
        return self._click(name)

    def back(self) -> str:
        """Esc = lùi một bước."""
        return v3_routes.on_back(self.state, time.perf_counter())

    def move_focus(self, delta: int) -> None:
        """↑/↓ trên màn chính: dịch vòng focus giữa bốn nút."""
        F.move_menu_focus(self.state, delta)

    # ---- nhịp ------------------------------------------------------------
    def tick(self) -> None:
        # TASK_accounts_1119: đọc kết quả một việc băm/kiểm chạy nền, nếu
        # có — rẻ (không có việc nào đang chạy thì chỉ là một so sánh
        # ``is None``), nên chạy TRƯỚC cả cổng ``ui_shots`` dưới đây.
        self._poll_account_job()
        # Khi chụp, KHÔNG chạy đồng hồ mô phỏng: nó đẩy "đang tìm trận" sang
        # "đã tìm thấy" ngay, nên hai ảnh ra y hệt nhau (bẫy của lượt album).
        if self.options.ui_shots:
            return
        F.update(self.state, time.perf_counter())
        now = time.perf_counter()
        dt = 0.0 if self._hover_last_t is None else max(0.0, now - self._hover_last_t)
        self._hover_last_t = now
        self._hover_fx.step(self.state.screen, self.hover(), dt)
        self._shot_feel.step(dt)
        self._tick_hud_fade(dt)
        if self._fx_enabled:
            fx_dt = 0.0 if self._fx_last_t is None else max(0.0, now - self._fx_last_t)
            self._fx_last_t = now
            self._fx_player.step(fx_dt)

    def _tick_hud_fade(self, dt: float) -> None:
        """TASK_game_feel_fix_1118 Fix 10.2/10.3: one Spring per fadeable
        card on the current gun screen (PLAY/PVP_HUNT — PVP_HIDE has no
        gun/crosshair, same scope as the kick/shake/crit-impact block in
        ``draw()``). Unscoped: the card under the (grown) crosshair fades
        to ``_HUD_FADE_PROXIMITY_ALPHA``, every other card eases back to
        1.0. Scoped: EVERY fadeable card fades to 0, regardless of
        proximity. Stepped every ``tick()`` regardless of which screen
        ``draw()`` ends up painting, so a spring already mid-flight keeps
        converging even across a one-frame screen flicker."""
        name = self.state.screen
        if name not in (F.PLAY, F.PVP_HUNT):
            self._hud_springs = {}
            self._hud_springs_screen = None
            return
        cards = cards_for_screen(self.manifest, name)
        if self._hud_springs_screen != name:
            self._hud_springs = {index: Spring(1.0, _HUD_FADE_SPRING_K, _HUD_FADE_SPRING_D)
                                 for index, card in enumerate(cards) if card.fades}
            self._hud_springs_screen = name
        if self.scope_visually_active:
            for spring in self._hud_springs.values():
                spring.to = 0.0
        else:
            near = card_index_for_point(cards, self.mouse, grow=_HUD_FADE_GROW_PX)
            for index, spring in self._hud_springs.items():
                spring.to = _HUD_FADE_PROXIMITY_ALPHA if index == near else 1.0
        for spring in self._hud_springs.values():
            spring.step(dt)

    @property
    def hud_card_alpha(self) -> dict:
        """``{card index: current alpha}`` for ``v3.py``'s per-card draw —
        empty on any screen without card-fade Springs (``state.card_alpha``
        then defaults every index to 1.0, i.e. no fade)."""
        return {index: spring.x for index, spring in self._hud_springs.items()}

    def toggle_hud_compact(self) -> bool:
        """TASK_game_feel_fix_1118 Fix 10.4: H key — flip in-memory state;
        ``app.py`` owns persisting it to ``user_prefs.json`` (same split
        as ``toggle_display``/``_apply_display_mode``). Returns the new
        value."""
        self.state.hud_compact = not self.state.hud_compact
        return self.state.hud_compact

    # ---- TASK_accounts_1119: accounts — local store, no server -----------
    @property
    def account_busy(self) -> bool:
        """True while a sign-in/sign-up hash job is running on its own
        thread — the primary button must ignore repeat clicks during this
        window (scrypt takes tens to hundreds of ms; a second click while
        one is in flight would start a SECOND concurrent hash for no
        reason, and could race the first one's result application)."""
        return self._account_job is not None

    def pop_account_corrupt_toast(self) -> bool:
        """True exactly once, the first time it's read after a launch
        where ``accounts.json`` AND its ``.bak`` were both unreadable —
        ``app.py`` shows a toast for this, then it never fires again this
        process (the store already started empty and renamed the bad
        file aside; nothing more to say after the first toast)."""
        was = self._account_corrupt
        self._account_corrupt = False
        return was

    def _start_account_job(self, kind: str, fn, *args) -> None:
        if self.account_busy:
            return
        holder: dict = {}
        self._account_job = (kind, holder)

        def worker() -> None:
            try:
                holder["result"] = fn(*args)
            except Exception as exc:                          # noqa: BLE001
                holder["error"] = exc
            finally:
                holder["done"] = True

        threading.Thread(target=worker, name=f"account-{kind}", daemon=True).start()

    def _poll_account_job(self) -> None:
        if self._account_job is None:
            return
        kind, holder = self._account_job
        if not holder.get("done"):
            return
        self._account_job = None
        if "error" in holder:
            logger.exception("TASK_accounts_1119: viec '%s' loi", kind,
                             exc_info=holder["error"])
            F.say(self.state, "Something went wrong. Try again.")
            return
        {
            "sign_in": self._apply_sign_in_result,
            "sign_up": self._apply_sign_up_result,
        }[kind](holder["result"])

    def sign_in_submit(self) -> None:
        """``ACT_SIGN_IN`` (``app.py``): the cooldown check is cheap and
        synchronous (in-memory, no hashing) and happens HERE, before even
        starting a thread — a name mid-cooldown must never spend a scrypt
        call just to be told to wait."""
        username, password = self.state.player, self.state.password
        key_guess = account_crypto.username_key(account_crypto.normalize_username(username))
        now = time.perf_counter()
        remaining = self._account_throttle.cooldown_remaining(key_guess, now)
        if remaining > 0.0:
            self.state.sign_in_error = f"Too many tries. Wait {int(round(remaining))} s."
            self.state.go(F.SIGN_IN_ERR)
            return
        self._start_account_job("sign_in", self._verify_credentials_job,
                                username, password)

    def _verify_credentials_job(self, username: str, password: str):
        result = account_store.verify_credentials(self._account_data, username, password)
        if result[0] == account_store.RESULT_OK:
            # Only a successful sign-in ever mutates ``data`` (last-signed-
            # in time, and a rehash if the stored scrypt params were
            # stale) — no point writing the file back out after a no-op
            # failed attempt.
            account_store.save(self._account_root, self._account_data)
        return result

    def _apply_sign_in_result(self, result) -> None:
        outcome, key = result
        now = time.perf_counter()
        if outcome == account_store.RESULT_OK:
            self._account_throttle.reset(key)
            self.state.account_key = key
            self.state.player = self._account_data["accounts"][key]["display"]
            self.state.password = self.state.password2 = ""
            self.state.focus = None
            self._load_progress(key)
            # TASK_accounts_1119 Phase 4: the switch's CURRENT state wins
            # on every sign-in — on means create/refresh a session, off
            # means make sure no stale one from an EARLIER sign-in (with
            # the switch on, since changed) survives this one.
            if self.state.remember:
                account_store.start_session(self._account_root, self._account_data, key)
            else:
                account_store.clear_session(self._account_root, self._account_data, key)
            account_store.save(self._account_root, self._account_data)
            self.state.go(F.MENU)
            return
        # RESULT_UNKNOWN has no key (no such account) — throttle on the
        # NAME the user actually typed, same as RESULT_WRONG's real key
        # (an unknown name still burns cooldown attempts, matching the
        # task's own "5 failures in a row for one name", not "one
        # account").
        throttle_key = key or account_crypto.username_key(
            account_crypto.normalize_username(self.state.player))
        self._account_throttle.record_failure(throttle_key, now)
        remaining = self._account_throttle.cooldown_remaining(throttle_key, now)
        if remaining > 0.0:
            self.state.sign_in_error = f"Too many tries. Wait {int(round(remaining))} s."
        elif outcome == account_store.RESULT_UNKNOWN:
            self.state.sign_in_error = f"No album named {self.state.player}. Create one below."
        else:
            self.state.sign_in_error = "Wrong password. Try again."
        self.state.go(F.SIGN_IN_ERR)

    def sign_up_submit(self) -> None:
        """``ACT_SIGN_UP`` — ``v3_routes`` already checked the FORMAT
        (name shape, password length/match) synchronously; this does the
        one check that needs the store (name actually free) plus the
        hashing, both off-thread."""
        self._start_account_job("sign_up", self._create_account_job,
                                self.state.player, self.state.password)

    def _create_account_job(self, display: str, password: str):
        if not account_store.is_available(self._account_data, display):
            return ("taken", None, None)
        key, code = account_store.create_account(self._account_data, display, password)
        account_store.save(self._account_root, self._account_data)
        return ("ok", key, code)

    def _apply_sign_up_result(self, result) -> None:
        outcome, key, code = result
        if outcome == "taken":
            self.state.touched.add("field_user")
            self.state.hint_bad = True
            self.state.hint_text = "Username already taken"
            return
        self.state.account_key = key
        self.state.player = self._account_data["accounts"][key]["display"]
        self.state.password = self.state.password2 = ""
        self.state.focus = None
        # TASK_accounts_1119 Phase 6 shows the one-time recovery code
        # dialog here; for now (Phase 2) go straight to the menu.
        self.state.go(F.MENU)
        _ = code

    def save_progress(self) -> None:
        """TASK_accounts_1119 Phase 5: ``history``/``recent``/the best
        score (derived from ``history``), so the leaderboard "you" row
        and the menu photo caption/damage survive a restart. Called
        after every match result, on sign-out and on exit (``app.py``) —
        a no-op for guest (never saved) or no account signed in at all.
        """
        key = self.state.account_key
        if not key or key == account_crypto.GUEST_KEY:
            return
        scores = [float(record.get("score", 0.0)) for record in self.state.history.values()
                 if isinstance(record, dict)]
        best_score = max(scores) if scores else None
        account_store.save_progress(self._account_data, key, self.state.history,
                                    self.state.recent, best_score)
        account_store.save(self._account_root, self._account_data)

    def note_match_finished(self) -> None:
        """TASK_accounts_1119 Phase 5: call once right after a match
        result is written into ``state.history`` (``app.py``'s
        ``_tick_album_match``). A signed-in account saves immediately;
        a guest instead gets a ONE-TIME toast (never saved — this IS the
        explanation why)."""
        if self.state.account_key == account_crypto.GUEST_KEY:
            if not self._guest_toast_shown:
                self._guest_toast_shown = True
                F.say(self.state, "Playing as guest — create an album to keep your scores.")
            return
        self.save_progress()

    def _load_progress(self, key: str) -> None:
        """The other half of ``save_progress`` — called right after a
        successful sign-in (``_apply_sign_in_result``) and on auto
        sign-in at launch, so ``flow.history``/``recent`` reflect what
        THIS account saved last time, not whatever a previous
        guest/account session left in memory."""
        progress = account_store.load_progress(self._account_data, key)
        if progress is None:
            return
        self.state.history = progress["history"]
        self.state.recent = progress["recent"]

    def sign_out(self) -> None:
        """TASK_accounts_1119 Phase 4: a deliberate sign-out must not
        leave a "remember me" session alive — same ``clear_session`` the
        switch's OFF path uses on a sign-in. Guest "sessions" never had
        one to begin with, but clearing ``session.json`` unconditionally
        is still correct (and cheap) either way. Phase 5: also the LAST
        chance to persist this session's progress before the account
        state about to be wiped by app.py's own "logout" handler."""
        self.save_progress()
        key = self.state.account_key
        if key and key != account_crypto.GUEST_KEY:
            account_store.clear_session(self._account_root, self._account_data, key)
            account_store.save(self._account_root, self._account_data)
        else:
            account_store.clear_session(self._account_root)

    def set_camera_frame(self, frame) -> None:
        """
        Nhận khung webcam mới nhất; chỉ đưa lên GPU khi màn đang mở cần nó.

        Màn chơi không có ô ``camera``, nên 99 % thời gian hàm này không làm
        gì — đúng luật "mỗi khung không upload texture" của F4.
        """
        if frame is None or not self.needs_camera():
            return
        token = id(frame)
        if token == self._camera_token:
            return
        self._camera_token = token
        self._camera_entry = _upload_bgr(self._camera_entry, frame)

    def needs_camera(self) -> bool:
        return any(p.get("kind") == "camera" for p in
                   self.manifest.screens[self.state.screen].get("photos", ()))

    def has_viewport(self) -> bool:
        return any(p.get("kind") == "viewport" for p in
                   self.manifest.screens[self.state.screen].get("photos", ()))

    # ---- vẽ --------------------------------------------------------------
    def draw(self, framebuffer, scene_draw=None) -> None:
        """Vẽ màn đang mở. ``scene_draw()`` vẽ cảnh 3D vào ô viewport."""
        name = self.state.screen
        self.painter.set_frame(framebuffer)
        if self.options.ui_shots:
            state = shot_state(self.manifest, name)
        else:
            state = v3_values.state_for(self.manifest, self.painter, self.state,
                                        self.game, self.mouse, accounts=self._account_data)
            state.hover = self.hover()
            state.pressed = self.mouse_down and state.hover is not None
            state.hover_fx = self._hover_fx
            state.camera_tex = self._camera_entry
            #: TASK_game_feel_fix_1118 Fix 10.4: every HUD screen (incl.
            #: PVP_HIDE — the hider's own HUD also gets the compact
            #: toggle), copied from the persistent flow each frame.
            state.hud_compact = self.state.hud_compact
            if scene_draw is not None:
                state.viewport_draw = lambda rect: self._scene(rect, scene_draw)
            if self._window is not None:
                self._window.set_cursor_shape(cursor_for(name, state.hover, state))
            self._step_fx_triggers(state, time.perf_counter())
            if name in (F.PLAY, F.PVP_HUNT):
                # TASK_game_feel_1118 P4: kick/hit marker/impact frame
                # stay fixed to the HUD; only the scene viewport rect
                # (below) and the fx layer (``offset=`` below) move with
                # shake — see ``shot_feel.py``'s module docstring.
                state.aim_kick_scale = self._shot_feel.kick_scale
                state.hit_markers = self._shot_feel.hit_markers
                state.crit_impact_alpha = self._shot_feel.crit_impact_alpha
                state.scene_shake = self._shot_feel.shake_offset
                # TASK_game_feel_1118 P6: ``visually_active`` (not
                # ``is_scoped``) — the vignette/reticle keep showing a
                # moment longer than the trigger-held state, matching the
                # zoom spring easing back to 1.0 after release.
                state.scope_zoom = (self._scope.z
                                    if self.scope_visually_active else 1.0)
                # TASK_game_feel_fix_1118 Fix 10.2/10.3: card alpha for
                # THIS screen's cards — ``_tick_hud_fade`` (in ``tick()``)
                # already stepped the Springs; this just reads them.
                state.card_alpha = self.hud_card_alpha
        if self._fx_enabled:
            if name != self._fx_screen:
                self._fx_library.enter_screen(name, self._fx_player)
                self._fx_screen = name
            self._fx_library.pump()
            state.fx_on = True
            shake = state.scene_shake
            # TASK_game_feel_fix_1118 Fix 5: a scene-anchored effect
            # (hit_kill/hit_crit/hit_block, spawned at the crosshair AT
            # THE TIME) must track its glass point while scoped, not stay
            # frozen at the old design-px spot — same treatment as shot
            # marks/hit markers (``_scope_adjust`` in ``v3.py``), applied
            # here via ``FxLibrary.draw``'s ``transform``/``hide_beyond``.
            if name in (F.PLAY, F.PVP_HUNT) and state.scope_zoom > 1.0 + 1e-3:
                from src.ui_album.scope import (VIGNETTE_RADIUS_PX,
                                               screen_scope_transform)

                crosshair, zoom = self.mouse, state.scope_zoom
                fx_transform = lambda p: screen_scope_transform(p, crosshair, zoom)
                fx_hide_beyond = (crosshair[0], crosshair[1], VIGNETTE_RADIUS_PX)
            else:
                fx_transform = None
                fx_hide_beyond = None
            state.fx_draw = lambda: self._fx_library.draw(
                self.painter, self._fx_player, offset=shake,
                transform=fx_transform, hide_beyond=fx_hide_beyond)
        self.painter.bind_screen(name, v3_values.extra_files(self.manifest,
                                                             name, state))
        self.painter.begin()
        draw_screen(self.painter, self.manifest, name, state)
        self.painter.end()

    # ---- Sakuga VFX (TASK_ui_sakuga_1117) ---------------------------------
    def spawn_fx(self, name: str, x: float, y: float, **kwargs) -> bool:
        """Spawn a named effect at design-space ``(x, y)``; no-op (returns
        ``False``) if fx is off or its texture isn't resident yet."""
        if not self._fx_enabled:
            return False
        return self._fx_library.try_spawn(self._fx_player, name, x, y, **kwargs)

    def spawn_fx_word(self, word: str, style: str, x: float, y: float, **kwargs) -> bool:
        """Spawn a bare word (no sprite) at design-space ``(x, y)``."""
        if not self._fx_enabled:
            return False
        return self._fx_library.try_spawn_word(self._fx_player, word, style, x, y, **kwargs)

    def clear_fx(self) -> None:
        """Drop every live effect right now, regardless of screen scope.

        TASK_game_feel_1118 P3: the sakuga screenshot stepper fires
        ``hit_kill``/``hit_crit``/``hit_block`` back to back on the SAME
        screen (``F.PLAY``) — ``_fx_library.enter_screen``'s scope-based
        clearing only runs on a SCREEN CHANGE, so the previous shot's
        effect (e.g. ``hit_crit``'s red DON) was still alive and visible
        in the NEXT shot's screenshot (``fx_miss.png``)."""
        if self._fx_enabled:
            self._fx_player.clear(lambda _name: True)

    @property
    def fx_enabled(self) -> bool:
        return self._fx_enabled

    def spawn_hit_fx(self, crit: bool) -> None:
        """A LOCAL shot just killed a bug (``app.py``'s ``_album_shoot``,
        only ever called for the player's own shots — never an
        opponent's) — hit_kill/hit_crit at the crosshair, plus a wordless
        star_pop over this screen's kill counter, if it has one."""
        if not self._fx_enabled:
            return
        mx, my = self.mouse
        self.spawn_fx("hit_crit" if crit else "hit_kill", mx, my)
        key = _KILL_STAR_TEXT.get(self.state.screen)
        if key is not None:
            spec = self.manifest.text(self.state.screen, key)
            if spec is not None:
                rect = spec["rect"]
                self.spawn_fx("star_pop", rect[0] + rect[2] * 0.5,
                             rect[1] + rect[3] * 0.5, words=False)

    def spawn_miss_fx(self) -> None:
        """A local shot missed — hit_block at the crosshair."""
        if not self._fx_enabled:
            return
        mx, my = self.mouse
        self.spawn_fx("hit_block", mx, my)

    # ---- shooting feel (TASK_game_feel_1118 P4) ---------------------------
    def register_shot(self, hit: bool, crit: bool) -> None:
        """Crosshair kick/shake/hit marker/crit impact for a shot
        that just resolved (``app.py``'s ``_album_shoot``, AFTER
        ``game_bridge.fire()`` — this never influences that result, only
        reacts to it). Independent of Sakuga fx — runs even with
        ``HCPD_NO_FX=1``."""
        self._shot_feel.on_shot(self.state.weapon_name(), self.mouse,
                                hit=hit, crit=crit)

    def bump_auto_bloom(self) -> None:
        """TASK_game_feel_1118 P5: one more consecutive Heavy-mag auto
        shot — called right after ``register_shot`` for that shot."""
        self._shot_feel.add_bloom()

    def release_auto_bloom(self) -> None:
        """The full-auto trigger was just released."""
        self._shot_feel.release_bloom()

    # ---- Precision-gun scope (TASK_game_feel_1118 P6) ---------------------
    @property
    def is_scoped(self) -> bool:
        return self._scope.active

    @property
    def scope_visually_active(self) -> bool:
        """``True`` a moment longer than ``is_scoped`` — the zoom is still
        meaningfully above 1.0 while it eases back down, so the camera
        keeps rendering the (shrinking) magnified view smoothly instead
        of cutting it off the instant the trigger is released."""
        return self._scope.visually_active

    @property
    def scope_zoom(self) -> float:
        return self._scope.z

    def enter_scope(self) -> None:
        self._scope.enter(self.mouse)

    def exit_scope(self) -> None:
        self._scope.exit()

    def step_scope(self, dt: float) -> None:
        self._scope.step(dt)

    def apply_scope_aim(self, raw_mouse: tuple) -> tuple:
        """Finer aim while scoped — see ``ScopeState.update_crosshair``.
        Caller (``app.py``) is responsible for only calling this while
        ``is_scoped``, and for assigning the result to ``self.mouse``."""
        return self._scope.update_crosshair(self.mouse, raw_mouse)

    # ---- per-tick/per-screen fx triggers (TASK_ui_sakuga_1117 Phase 6) ----
    def _step_fx_triggers(self, state, now: float) -> None:
        """Effects that fire on a STATE TRANSITION rather than a discrete
        click/shot — loading bar, tilt checklist, a new toast, entering
        WIN/PVP_RESULT. Reads ``state`` (this frame's already-computed
        ``v3_values.state_for`` output) so it never re-derives business
        logic (win rank, pipeline step, ...) that already lives there."""
        if not self._fx_enabled:
            return
        dt = (0.0 if self._fx_trigger_last_t is None
             else max(0.0, now - self._fx_trigger_last_t))
        self._fx_trigger_last_t = now
        screen = self.state.screen
        if screen != self._fx_trigger_screen:
            self._on_fx_trigger_screen_enter(screen, now)
            self._fx_trigger_screen = screen
        if screen in (F.LOADING, F.LOADING_READY):
            self._step_loading_fx(state, dt)
        elif screen == F.TILT:
            self._step_tilt_fx()
        elif screen == F.WIN:
            self._step_win_fx(state, now)
        elif screen == F.PVP_RESULT:
            self._step_pvp_result_fx(state)
        if screen not in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE):
            self._step_toast_fx(state, screen)

    def _on_fx_trigger_screen_enter(self, screen: str, now: float) -> None:
        if screen == F.LOADING:
            self._fx_load_edge_elapsed = 0.0
            self._fx_gogo_spawned = False
            self._fx_load_done_spawned = False
            self._fx_last_step_done = 0
        if screen == F.TILT:
            self._fx_last_tilt_checks = None
        if screen == F.WIN:
            self._fx_win_entered_t = now
            self._fx_win_star_spawned = False
            self._fx_rank_up_spawned = False
        if screen == F.PVP_RESULT:
            self._fx_pvp_result_spawned = False

    def _step_loading_fx(self, state, dt: float) -> None:
        # Mỗi "đã spawn" dưới đây chỉ CHỐT khi ``spawn_fx`` trả True — sheet
        # chưa kịp decode (lần đầu vào màn, cache nguội) thì thử lại đúng
        # khung sau, không bỏ cuộc sau một lần trượt (từng là lỗi thật: cờ
        # "đã spawn" chốt trước khi biết kết quả, nên lần đầu lạnh cache là
        # mất hẳn hiệu ứng đó, không bao giờ thử lại).
        pct = max(0.0, min(1.0, float(state.bars.get("progress", 0.0))))
        bar_x = _LOAD_BAR_X0 + _LOAD_BAR_W * pct
        if self.state.screen == F.LOADING and 0.0 < pct < 1.0:
            self._fx_load_edge_elapsed += dt
            if self._fx_load_edge_elapsed >= _LOAD_EDGE_INTERVAL_S:
                self._fx_load_edge_elapsed = 0.0
                self.spawn_fx("load_edge", bar_x, _LOAD_BAR_Y)
            if not self._fx_gogo_spawned and 0.30 < pct < 0.90:
                word_x = max(_GOGO_X_RANGE[0], min(_GOGO_X_RANGE[1], bar_x))
                if self.spawn_fx_word("gogo", "ink", word_x, _GOGO_Y, size=150.0,
                                      rot=-0.08, life=2.9):
                    self._fx_gogo_spawned = True
        if not self._fx_load_done_spawned and pct >= 1.0:
            if self.spawn_fx("load_done", *_LOAD_DONE_XY):
                self._fx_load_done_spawned = True
        step_dots = state.pips.get("step_dots")
        if isinstance(step_dots, (list, tuple)):
            done = sum(1 for flag in step_dots if flag)
            index = self._fx_last_step_done
            while index < min(done, len(_STEP_DOT_X)):
                if not self.spawn_fx("star_pop", _STEP_DOT_X[index], _STEP_DOT_Y,
                                     words=False):
                    break               # chua decode xong: thu lai khung sau
                index += 1
            self._fx_last_step_done = index

    def _step_tilt_fx(self) -> None:
        checks = tuple(self.state.tilt_checks())
        last = self._fx_last_tilt_checks
        if last is None:
            # Lần đầu sau khi vào TILT: chỉ lấy làm MỐC, không nổ sao cho
            # dấu nào đã xong từ trước (quay lại màn này sau một lần mất
            # mạng chẳng hạn) — đúng nghĩa "chỉ khi CHUYỂN sang xong".
            self._fx_last_tilt_checks = checks
            return
        new_last = list(last)
        for index, done in enumerate(checks):
            if not done:
                continue
            if last[index] or self.spawn_fx(
                    "star_pop", _TILT_STAR_X,
                    _TILT_STAR_Y0 + _TILT_STAR_DY * index, words=False):
                new_last[index] = True
            # else: chua decode xong, giu False de thu lai khung sau.
        self._fx_last_tilt_checks = tuple(new_last)

    def _step_win_fx(self, state, now: float) -> None:
        if not self._fx_win_star_spawned:
            if self.spawn_fx("star_five", *_WIN_SCORE_XY):
                self._fx_win_star_spawned = True
        if (not self._fx_rank_up_spawned and "rank_up" in state.sprites
                and now - self._fx_win_entered_t >= _RANK_UP_DELAY_S):
            record = self.manifest.sprite(F.WIN, "rank_up")
            if record is not None:
                rect = record["rect"]
                if self.spawn_fx("star_pop", rect[0] + rect[2] * 0.5,
                                 rect[1] + rect[3] * 0.5, words=False):
                    self._fx_rank_up_spawned = True

    def _step_pvp_result_fx(self, state) -> None:
        if self._fx_pvp_result_spawned:
            return
        if "crown" not in state.sprites:
            # Thua/hoà: không có hiệu ứng, và KHÔNG có gì để spawn nên
            # không cần thử lại — chốt luôn, khỏi gọi lại mỗi khung.
            self._fx_pvp_result_spawned = True
            return
        if self.spawn_fx("star_five", *_PVP_CROWN_XY):
            self._fx_pvp_result_spawned = True

    def _step_toast_fx(self, state, screen: str) -> None:
        until = self.state.message_until_s
        if not state.message or until == self._fx_last_message_until:
            return
        self._fx_last_message_until = until
        x, y = _TOAST_PHOTOS_XY if screen == F.PHOTOS else _TOAST_XY
        self.spawn_fx("toast_in", x, y)

    def _scene(self, rect, scene_draw) -> None:
        """
        Cảnh 3D chỉ hiện **trong ô viewport**, bằng ``glScissor``.

        Phép chiếu Kooima giữ nguyên cho cả cửa sổ (D2 — đúng vật lý), ô
        viewport là một **khung cắt** của khung nhìn thật, không co giãn gì.
        """
        self.painter.end()
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(*self.painter.viewport_px(rect))
        scene_draw()
        GL.glDisable(GL.GL_SCISSOR_TEST)
        self.painter.begin()

    def draw_hidden_scene(self, scene_draw) -> None:
        """Chạy trọn ``renderer.render`` nhưng cắt còn **1 pixel** (xem ui.py)."""
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(0, 0, 1, 1)
        try:
            scene_draw()
        finally:
            GL.glDisable(GL.GL_SCISSOR_TEST)

    # ---- chụp cả bộ màn (U1.3) ------------------------------------------
    def shot_plan(self) -> list[str]:
        """23 màn, theo thứ tự manifest. Một theme duy nhất nên không nhân 2."""
        return self.manifest.screen_names()

    def apply_shot(self, index: int) -> str | None:
        plan = self.shot_plan()
        if index >= len(plan):
            return None
        self.state.screen = plan[index]
        return plan[index]

    def release(self) -> None:
        if self._camera_entry is not None:
            GL.glDeleteTextures(1, [self._camera_entry[0]])
            self._camera_entry = None
        if self._fx_library is not None:
            self._fx_library.release()
        self.painter.release()


def _upload_bgr(entry, frame):
    """Khung BGR của OpenCV → texture RGBA, **dùng lại** tên texture cũ."""
    import numpy as np

    from src.ui_album.textures import upload_rgba

    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[2] < 3:
        return entry
    rgba = np.empty((array.shape[0], array.shape[1], 4), dtype=np.uint8)
    rgba[..., 0] = array[..., 2]
    rgba[..., 1] = array[..., 1]
    rgba[..., 2] = array[..., 0]
    rgba[..., 3] = 255
    if entry is not None and (entry[1], entry[2]) == (array.shape[1], array.shape[0]):
        GL.glBindTexture(GL.GL_TEXTURE_2D, entry[0])
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, array.shape[1],
                           array.shape[0], GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                           np.ascontiguousarray(rgba))
        return entry
    if entry is not None:
        GL.glDeleteTextures(1, [entry[0]])
    return (upload_rgba(rgba), array.shape[1], array.shape[0])
