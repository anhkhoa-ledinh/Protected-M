"""
Cầu nối giữa app và giao diện v3: một đối tượng, ba việc.

``handle_input`` (chuột/phím) → ``tick`` (nhịp mô phỏng) → ``draw`` (vẽ màn
đang mở) — y lối của ``ui.py`` cho bộ asset cũ, để ``app.py`` chỉ phải chọn
một trong hai đối tượng chứ không phải hai đường mã.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from OpenGL import GL

from src.shared import account_rules
from src.common.stage_profile import stage
from src.fx.anim import Spring
from src.fx.library import FxLibrary
from src.fx.player import FxPlayer
from src.fx.spec import load_spec
from src.ui_album import v3_routes, v3_values
from src.ui_album import v3_flow as F
from src.ui_album.hud_cards import card_index_for_point, cards_for_screen
from src.ui_album.texture_loader import PRIORITY_PREFETCH
from src.ui_album.scope import ScopeState
from src.ui_album.shot_feel import ShotFeel
from src.ui_album.v3 import V3Manifest, draw_screen, shot_state
from src.ui_album.v3_hit import hit_test
from src.ui_album.v3_painter import V3Painter
from src.ui_album.v3_hover import HoverFx, cursor_for
from src.ui_album import ids as ID
from src.ui_album.v3_account import AccountMixin
from src.ui_album.v3_click_geometry import ClickGeometryMixin
from src.ui_album.v3_fx_steps import FxStepsMixin
from src.ui_album.v3_consts import (_HUD_FADE_GROW_PX, _HUD_FADE_PROXIMITY_ALPHA, _HUD_FADE_SPRING_D, _HUD_FADE_SPRING_K, _HUD_HIDDEN_TOAST, _HUD_HIDE_SPRING_D, _HUD_HIDE_SPRING_K)

logger = logging.getLogger(__name__)
class V3UI(AccountMixin, FxStepsMixin, ClickGeometryMixin):
    """Giao diện v3 gắn vào một cửa sổ GLFW đã có context."""

    def __init__(self, root: Path, shader_dir: Path, options, window=None, *,
                 accounts, shared_album, prefs) -> None:
        self.manifest = V3Manifest(root)
        self.painter = V3Painter(self.manifest, shader_dir)
        self.state = F.V3Flow()
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
        self._shot_feel.shake_enabled = prefs.screen_shake_enabled()
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
        #: TASK_hud_fix_1119 Fix 2: H hide/show — session-only (never
        #: persisted), and reset to visible whenever a NEW match begins
        #: (tracked via ``_hud_hidden_match_screen``: the gun-screen name
        #: changing TO one of PLAY/PVP_HUNT/PVP_HIDE from something else
        #: means a fresh match just started, not just a frame redrawing
        #: the same ongoing one).
        self._hud_hidden: bool = False
        self._hud_hide_spring = Spring(1.0, _HUD_HIDE_SPRING_K, _HUD_HIDE_SPRING_D)
        self._hud_hidden_match_screen: str | None = None
        #: Khung webcam đã lên GPU: ô ``camera`` của màn cài đặt / thử nghiêng.
        self._camera_entry = None
        self._camera_token = None
        # TASK_accounts_1119: kho tài khoản cục bộ — nạp MỘT LẦN ở đây,
        # giống `_fx_enabled`/`_shot_feel`: thuộc phiên giao diện, không
        # phải trạng thái màn (``V3Flow``). ``app.py`` không tự đọc/ghi
        # file tài khoản — chỉ gọi các phương thức công khai dưới đây.
        self._root = root
        #: The injected ``AccountService`` (``app`` builds it): the UI never
        #: touches the accounts files or the hashing itself.
        self._accounts = accounts
        self._handlers: dict = {}
        self._prefix_handlers: list = []
        self._register_account_handlers()
        self._shared_album = shared_album
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
            self.state.account_key = account_rules.GUEST_KEY
            self.state.go(F.MENU)
        else:
            # TASK_accounts_1119 Phase 4: a valid "remember me" session
            # signs in STRAIGHT to the menu, no sign-in screen — and
            # refreshes the expiry (a session that's actively used should
            # not quietly expire mid-way through a long play session).
            resumed_key = self._accounts.resume_session()
            if resumed_key is not None:
                self.state.account_key = resumed_key
                self.state.player = self._accounts.display_of(resumed_key)
                self._load_progress(resumed_key)
                self.state.go(F.MENU)
            else:
                # No session: pre-fill the username field with the last
                # account that signed in on this machine (nothing to
                # pre-fill if none ever has).
                last_display = self._accounts.last_display()
                if last_display is not None:
                    self.state.player = last_display
        # Móc đo: mở thẳng một màn để đếm FPS trên đúng màn đó (vd. màn chơi).
        forced = os.environ.get("HCPD_V3_SCREEN", "").strip()
        if forced and forced in self.manifest.screens:
            self.state.go(forced)

        # TASK_ui_sakuga_1117: tắt qua HCPD_NO_FX hoặc "fx": false trong
        # user_prefs.json (G6) — tắt thì KHÔNG đọc fx.json luôn, không chỉ
        # không vẽ, nên không texture nào của bộ này lên GPU.
        self._fx_enabled = prefs.fx_enabled()
        self._fx_player = None
        self._fx_library = None
        self._fx_screen: str | None = None
        self._fx_last_t: float | None = None
        #: TASK_perf_textures_1119 Fix 4: which pack is currently active
        #: ("fx" or "fx@2x"), and the ``(dir, player, library)`` of a pack
        #: switch in progress (``None`` once there isn't one) — see
        #: ``_ensure_fx_pack``/``_poll_fx_pack_switch``.
        self._fx_dir: str | None = None
        self._fx_pending: tuple[str, object, object] | None = None
        if self._fx_enabled:
            try:
                spec = load_spec(root, fx_dir="fx")
            except (OSError, ValueError) as err:
                logger.warning("V3UI: khong nap duoc fx.json (%s) — tat hieu ung", err)
                self._fx_enabled = False
            else:
                self._fx_player = FxPlayer(spec)
                self._fx_library = FxLibrary(spec)
                self._fx_dir = "fx"

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
        # TASK_perf_textures_1119 Fix 3: queue the screens most launches
        # land on FIRST, at prefetch priority, so by the time the player
        # reaches MENU its textures are already decoding/decoded instead
        # of starting cold.
        for name in (F.SIGN_IN, F.SIGN_UP, F.SIGN_IN_ERR, F.MENU):
            self._prefetch_screen(name)
        self._prefetch_last_screen: str | None = None

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
        if self.state.recovery_code_shown:
            return self._recovery_hover(*self.mouse)
        if self.state.forgot_open:
            return self._forgot_hover(*self.mouse)
        return hit_test(self.manifest, self.state.screen, *self.mouse)


    def click(self) -> str:
        """Bấm chuột trái tại vị trí hiện tại; trả hành động cho app."""
        return self._click(self.hover(), from_pointer=True)

    def activate(self, name: str) -> str:
        """Kích hoạt một hotspot CỤ THỂ (phím tắt U/R…), bỏ qua vị trí chuột."""
        return self._click(name, from_pointer=False)

    def _click(self, name: str | None, *, from_pointer: bool = True) -> str:
        """``on_click`` + (TASK_ui_sakuga_1117 Phase 6) a ui_tap/ui_confirm
        burst at the hotspot's PRE-click position — captured before
        ``on_click`` runs, since it may change ``self.state.screen`` out
        from under a post-click lookup.

        ``from_pointer`` (TASK_small_fixes_1120): ``True`` only for a real
        mouse click (``click()``) — ``activate()`` (keyboard shortcuts,
        test harnesses) ignores ``self.mouse`` for HOTSPOT resolution
        already (its own docstring), so the caret it leaves behind must
        land at the field's END, not at some stale mouse position."""
        if not name:
            self.pending = ""
            return ""
        screen = self.state.screen
        in_dialog = bool(self.state.shared_album_pending
                         or self.state.recovery_code_shown
                         or self.state.forgot_open)
        rect = self._click_rect(screen, name, in_dialog)
        is_primary = (not in_dialog and
                     name == v3_routes.primary_button(self.manifest, self.state))
        pre_items = list(self.state.items) if hasattr(self.state, "items") else None
        pre_focus = self.state.forgot_focus if self.state.forgot_open else self.state.focus
        self._sync_shared_album_flags()
        action = v3_routes.on_click(self.state, name, time.perf_counter(),
                                    self.mouse)
        self.pending = action
        self._sync_caret_after_click(name, pre_focus, from_pointer)
        self._spawn_click_fx(name, screen, rect, in_dialog, is_primary, pre_items)
        return action


    def key(self, char: str) -> None:
        # TASK_accounts_1119 Phase 6: the forgot dialog has its OWN four
        # fields/focus, separate from whatever screen it overlays.
        if self.state.forgot_open:
            F.forgot_type_char(self.state, char)
            return
        F.type_char(self.state, char)

    def backspace(self) -> None:
        if self.state.forgot_open:
            F.forgot_backspace(self.state)
            return
        F.backspace(self.state)

    def delete_char(self) -> None:
        """Delete key: removes the character RIGHT of the caret."""
        if self.state.forgot_open:
            F.forgot_delete_char(self.state)
            return
        F.delete_char(self.state)

    def move_caret(self, delta: int) -> None:
        """Left/Right arrow keys."""
        F.move_caret(self.state, delta)

    def caret_home(self) -> None:
        F.caret_home(self.state)

    def caret_end(self) -> None:
        F.caret_end(self.state)

    def focus_next(self) -> None:
        if self.state.forgot_open:
            F.forgot_focus_next(self.state)
            return
        fields = {h["n"] for h in
                  self.manifest.screens[self.state.screen].get("hotspots", ())}
        F.focus_next(self.state, fields)

    def submit(self) -> str:
        """Enter = nút chính của màn đang mở (U2: "Enter = the primary
        button") — or of whichever dialog is open, if one is."""
        if self.state.recovery_code_shown:
            return self._click(ID.BTN_RECOVERY_SAVED)
        if self.state.forgot_open:
            return self._click(ID.BTN_FORGOT_SUBMIT)
        name = v3_routes.primary_button(self.manifest, self.state)
        if not name:
            return ""
        return self._click(name)

    def back(self) -> str:
        """Esc = lùi một bước — hoặc đóng hộp thoại đang mở, nếu có
        (TASK_accounts_1119 Phase 6: không khoá người dùng lại trong một
        hộp thoại không có đường thoát bằng bàn phím)."""
        if self.state.recovery_code_shown:
            self.state.recovery_code_shown = ""
            return ""
        if self.state.forgot_open:
            self.state.forgot_open = False
            self.state.forgot_focus = None
            self.state.forgot_error = ""
            return ""
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
        # TASK_perf_textures_1119 Fix 3: scrypt hashing (account_busy)
        # measurably competes with the texture loader's own background
        # thread for the GIL on this machine (G3's own frame budget is
        # tighter than either alone) — idle the loader for the ~100-300ms
        # a hash job runs instead of racing it.
        if self.account_busy:
            self.painter.pause_loading()
        else:
            self.painter.resume_loading()
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
        self._tick_hud_hidden(dt)
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

    def _tick_hud_hidden(self, dt: float) -> None:
        """TASK_hud_fix_1119 Fix 2: steps the H-hide spring every tick
        (same reasoning as ``_tick_hud_fade`` — keeps converging even
        across a one-frame screen flicker). A match screen entered FRESH
        (it wasn't the match screen the tick before) snaps hidden back
        to visible instantly — "every match starts with the full HUD" —
        without that re-triggering every single frame of one ongoing
        match."""
        name = self.state.screen
        is_match_screen = name in (F.PLAY, F.PVP_HUNT, F.PVP_HIDE)
        if is_match_screen and name != self._hud_hidden_match_screen:
            self._hud_hidden = False
            self._hud_hide_spring.x = 1.0
            self._hud_hide_spring.v = 0.0
            self._hud_hide_spring.to = 1.0
        self._hud_hidden_match_screen = name if is_match_screen else None
        self._hud_hide_spring.step(dt)

    @property
    def hud_hidden_alpha(self) -> float:
        return self._hud_hide_spring.x

    @property
    def hud_hidden(self) -> bool:
        return self._hud_hidden

    def toggle_hud_hidden(self) -> bool:
        """TASK_hud_fix_1119 Fix 2: H key — session-only, never
        persisted (no prefs write anywhere in this method). Returns the
        new hidden state. A toast fires only when HIDING; nothing on
        show, same as the task's own wording."""
        self._hud_hidden = not self._hud_hidden
        self._hud_hide_spring.to = 0.0 if self._hud_hidden else 1.0
        if self._hud_hidden:
            F.say(self.state, _HUD_HIDDEN_TOAST)
        return self._hud_hidden


    # ---- B5: actions go through one door -------------------------------
    def register_handler(self, action: str, handler) -> None:
        """``handler(action, now)`` runs when ``dispatch(action, now)`` is called."""
        self._handlers[action] = handler

    def register_prefix_handler(self, prefix: str, handler) -> None:
        self._prefix_handlers.append((prefix, handler))

    def dispatch(self, action: str, now: float) -> bool:
        """Forward an action name (``routes`` returns these) to the callback
        ``app`` registered for it, or to the account controller. ``False`` =
        nobody handles it (the empty action is not an error)."""
        handler = self._handlers.get(action)
        if handler is None:
            handler = next((h for prefix, h in self._prefix_handlers
                            if action.startswith(prefix)), None)
        if handler is None:
            return False
        handler(action, now)
        return True

    def _register_account_handlers(self) -> None:
        self.register_handler("account_sign_in", lambda action, now: self.sign_in_submit())
        self.register_handler("account_sign_up", lambda action, now: self.sign_up_submit())
        self.register_handler("account_forgot_submit",
                              lambda action, now: self.forgot_submit())
        self.register_handler("account_copy_recovery_code", self._copy_recovery_code_handler)

    def _copy_recovery_code_handler(self, action: str, now: float) -> None:
        self.copy_recovery_code()
        F.say(self.state, "Code copied")

    def _sync_shared_album_flags(self) -> None:
        flow = self.state
        flow.shared_can_upload = self._shared_album.can_upload()
        flow.shared_can_list = self._shared_album.can_list()
        flow.shared_not_configured = self._shared_album.NOT_CONFIGURED


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
        self._ensure_fx_pack(framebuffer[0])
        self._poll_fx_pack_switch()
        if self.options.ui_shots:
            state = shot_state(self.manifest, name)
        else:
            self._sync_shared_album_flags()
            state = v3_values.state_for(self.manifest, self.painter, self.state,
                                        None if self.game is None
                                        else self.game.view(time.perf_counter()), self.mouse, accounts=self._accounts.accounts_view())
            state.hover = self.hover()
            state.pressed = self.mouse_down and state.hover is not None
            state.hover_fx = self._hover_fx
            state.camera_tex = self._camera_entry
            #: TASK_hud_fix_1119 Fix 2: every HUD screen (incl. PVP_HIDE
            #: — the hider's own cards fade too), read from the spring
            #: ``_tick_hud_hidden`` (in ``tick()``) already stepped.
            state.hud_hidden_alpha = self.hud_hidden_alpha
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
            def _fx_draw() -> None:
                with stage("fx"):
                    self._fx_library.draw(
                        self.painter, self._fx_player, offset=shake,
                        transform=fx_transform, hide_beyond=fx_hide_beyond)
            state.fx_draw = _fx_draw
        if name != self._prefetch_last_screen:
            self._prefetch_last_screen = name
            for neighbour in v3_routes.NEXT_SCREENS.get(name, ()):
                self._prefetch_screen(neighbour)
        self.painter.bind_screen(name, v3_values.extra_files(self.manifest,
                                                             name, state))
        self.painter.pump_uploads()
        self.painter.begin()
        draw_screen(self.painter, self.manifest, name, state)
        self.painter.end()

    def _prefetch_screen(self, screen: str) -> None:
        """TASK_perf_textures_1119 Fix 3: queue ``screen``'s textures at
        background priority — never blocks, never pre-empts the CURRENT
        screen's own (higher-priority) requests in the loader queue."""
        self.painter.bind_screen(screen, (), priority=PRIORITY_PREFETCH)

    def prepare_screen(self) -> bool:
        """TASK_perf_textures_1119 Fix 3: queues the current screen's
        textures and drains finished GL uploads — called every frame by
        ``app.py`` BEFORE it decides whether to draw this screen or keep
        showing the last one (``current_screen_ready``). Doing this
        outside ``draw()`` means loading keeps progressing even while
        ``app.py`` is choosing to freeze on the old frame.

        ``extra_files`` (borrowed-sprite files, e.g. a HUD slot icon
        borrowed from LOADOUT's sprite set) needs the full draw ``state``
        ``v3_values.state_for`` builds, which this pre-draw check doesn't
        have — those few sprites get requested from ``draw()`` itself
        instead, same as before Fix 3; this only tracks base/top/screen's
        OWN sprites, the ones actually responsible for the freeze."""
        name = self.state.screen
        self.painter.bind_screen(name, ())
        self.painter.pump_uploads()
        return self.painter.screen_ready(name, ())

    def current_screen_ready(self) -> bool:
        return self.painter.screen_ready(self.state.screen, ())


    @property
    def fx_enabled(self) -> bool:
        return self._fx_enabled


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
        if self._fx_pending is not None:
            self._fx_pending[2].release()
            self._fx_pending = None
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
