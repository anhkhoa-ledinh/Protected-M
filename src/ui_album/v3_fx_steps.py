"""Part of ``V3UI``, moved out of ``v3_ui.py`` unchanged (B5): see ``FxStepsMixin``."""

from __future__ import annotations

import logging
from src.fx.library import FxLibrary
from src.fx.player import FxPlayer
from src.fx.spec import load_spec
from src.ui_album import v3_flow as F
from src.ui_album import ids as ID
from src.ui_album.v3_consts import (FX_WIDE_THRESHOLD_PX, _GOGO_X_RANGE, _GOGO_Y, _KILL_STAR_TEXT, _LOAD_BAR_W, _LOAD_BAR_X0, _LOAD_BAR_Y, _LOAD_DONE_XY, _LOAD_EDGE_INTERVAL_S, _PVP_CROWN_XY, _RANK_UP_DELAY_S, _STEP_DOT_X, _STEP_DOT_Y, _TILT_STAR_DY, _TILT_STAR_X, _TILT_STAR_Y0, _TOAST_PHOTOS_XY, _TOAST_XY, _UI_TAP_BIG_HEIGHT_PX, _UI_TAP_BIG_WIDTH_PX, _UI_TAP_EXACT, _UI_TAP_PREFIXES, _WIN_SCORE_XY)

logger = logging.getLogger("src.ui_album.v3_ui")


class FxStepsMixin:
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
        if (self.state.last_result or {}).get("rejected"):
            return          # the server did not count this win: no celebration
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

    def _spawn_click_fx(self, name: str, screen: str, rect, in_dialog: bool,
                        is_primary: bool, pre_items) -> None:
        if not self._fx_enabled:
            return
        if in_dialog:
            if name in (ID.BTN_CONFIRM_YES, ID.BTN_CONFIRM_NO) and rect is not None:
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

    # ---- fx: 1.5x vs 2x pack (TASK_perf_textures_1119 Fix 4) --------------
    def _ensure_fx_pack(self, width: int) -> None:
        """Starts loading the pack ``width`` wants, in the BACKGROUND, if
        it isn't already the active or in-progress one. Never touches
        the currently-drawing pack — ``_poll_fx_pack_switch`` swaps over
        only once the new one has finished preloading."""
        if not self._fx_enabled:
            return
        wanted = "fx@2x" if width > FX_WIDE_THRESHOLD_PX else "fx"
        if wanted == self._fx_dir:
            return
        if self._fx_pending is not None and self._fx_pending[0] == wanted:
            return
        try:
            spec = load_spec(self._root, fx_dir=wanted)
        except (OSError, ValueError) as err:
            logger.warning("V3UI: khong doi duoc sang goi fx %s (%s)", wanted, err)
            return
        library = FxLibrary(spec)
        if not library.check_capability():
            return
        player = FxPlayer(spec)
        library.start_preload()
        self._fx_pending = (wanted, player, library)

    def _poll_fx_pack_switch(self) -> None:
        if self._fx_pending is None:
            return
        wanted, player, library = self._fx_pending
        # The pending library's own decode results only turn into resident
        # textures (and ``preload_remaining`` drop) when something calls
        # ITS ``pump()`` -- the active pack's ``pump()`` call in ``draw()``
        # only ever drains ITSELF.
        library.pump()
        if library.preload_remaining > 0:
            return
        old_library = self._fx_library
        self._fx_player, self._fx_library, self._fx_dir = player, library, wanted
        self._fx_pending = None
        if old_library is not None:
            old_library.release()
        logger.info("V3UI: doi sang goi fx %s (framebuffer rong hon %dpx)",
                   wanted, FX_WIDE_THRESHOLD_PX)
