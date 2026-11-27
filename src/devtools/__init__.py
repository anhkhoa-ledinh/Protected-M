"""TASK_arch_refactor_1120 R3: bench/evidence/shots/automatch hooks, moved
out of ``src.engine.app.HcpdApplication`` (about 27% of that file). Every
``HCPD_*`` env flag name and behaviour is unchanged — only WHERE the code
lives moved.

``HcpdApplication`` never imports this module unless at least one
``HCPD_*`` flag (see ``HOOK_ENV_VARS``) is set at startup — a normal run
never touches it (``'src.devtools' not in sys.modules``). When no flag is
set, ``HcpdApplication._devtools`` is a tiny local no-op stub instead
(``src.engine.app._NullDevTools``) with the same method names, so every
per-frame call site is unconditional.

``DevTools`` holds one reference, ``self._app`` (the ``HcpdApplication``
instance) — every place a moved method used to read ``self.xxx`` for
something that lives on the app (``_album``, ``_renderer``, ``_window``,
``_game``, ``_bugs``, ``_root``, ``_album_options``, ``_content``, core
methods like ``_tick_scope``/``_save_album_shot``/``_album_shoot``/
``_start_album_match``/``_pvp_action``) now reads ``self._app.xxx``
instead. Transient bookkeeping that belongs to a hook itself (bench
timers, shot-sequence indices) stays on ``self`` — it now lives on this
object instead of the app, which is the point of the move.
"""

from __future__ import annotations

import logging
import math
import os
import time
from pathlib import Path
from src.ui_album import ids as ID

logger = logging.getLogger(__name__)


class DevTools:
    #: fx_kill/crit/miss: once ``_fx_shot_pending`` finally spawns (texture
    #: decode is unbounded — depends on disk/CPU), hold this many MORE
    #: frames so the capture lands mid-animation, not frame 0.
    _FX_PENDING_POST_FRAMES = 8

    def __init__(self, app) -> None:
        from src.engine.dev_hooks import DevHooks

        self._app = app if isinstance(app, DevHooks) else DevHooks(app)
        from src.devtools import room_1127

        self._room_script = room_1127.from_env(self._app)

    def pinned_eye(self):
        """HCPD_ROOM_1127: the eye pinned by the script, else ``None``."""
        return None if self._room_script is None else self._room_script.pinned_eye()

    def tick_room(self, now: float) -> None:
        if self._room_script is not None:
            self._room_script.tick(now)

    def env(self, name: str, default: str = "") -> str:
        """The one place that actually reads an ``HCPD_*`` env var —
        every other call site (in this module or in app.py) goes through
        this instead of ``os.environ.get`` directly."""
        return os.environ.get(name, default)

    # ---- per-frame entry points app.py calls -------------------------------

    def tick_pre_match(self, now: float) -> None:
        """Replaces, in the SAME order, the six devtools ticks that used
        to run right before ``_tick_album_match`` in ``_tick_album``."""
        self._tick_album_automatch(now)
        self._tick_album_autowin(now)
        self._tick_fx_bench(now)
        self._tick_scope_bench(now)
        self._tick_evidence_1118(now)
        self._tick_hud_fix_1119_shots(now)
        self._tick_hardening_1124_shots(now)
        self._tick_reload_1124(now)

    def tick_pvp(self, now: float) -> None:
        """Replaces, in the SAME order, the two devtools ticks that used
        to run right after ``_tick_pvp`` in ``_tick_album``."""
        self._tick_pvp_auto(now)
        self._tick_pvp_shots(now)

    def tick_post_render(self, now: float) -> None:
        """Replaces the post-render shots if/elif chain in ``run()``."""
        app = self._app
        if app.album_options.shots or app.album_options.ui_shots:
            self._album_shots_step()
        elif self.env("HCPD_ECONOMY_1124"):
            self._tick_economy_1124(now)
        elif self.env("HCPD_V3_LIVE") == "1":
            self._tick_v3_live_shots()
        elif self.env("HCPD_UI_SHOTS_1117") == "1":
            self._tick_sakuga_shots()
        elif self.env("HCPD_AA_BASELINE_1118") == "1":
            self._tick_aa_baseline_shots()

    # ---- TASK_ui_v3_1107/U3: live-value screenshot sequence ----------------

    def _tick_v3_live_shots(self) -> None:
        """
        ``HCPD_V3_LIVE=1``: chụp bộ ảnh U3 bằng **đường giá trị sống**.

        Mỗi preset vẽ đủ 3 khung rồi mới đọc pixel — cùng lý do với
        ``_album_shots_step``: đọc ngay khung đầu sau khi đổi màn cho ra ảnh
        của màn trước.
        """
        from OpenGL import GL

        from src.devtools import v3_live

        app = self._app
        if app.window.is_iconified():
            return
        self._live_hold = getattr(self, "_live_hold", 0) + 1
        if self._live_hold < 3:
            return
        self._live_hold = 0
        index = getattr(self, "_live_index", -1)
        if index >= 0:
            GL.glFinish()
            name = v3_live.ORDER[index]
            if name == v3_live.DESKTOP:
                self._grab_desktop(name)
            else:
                app.save_album_shot(name)
        index += 1
        if index >= len(v3_live.ORDER):
            logger.info("UI v3 live: xong %d anh", len(v3_live.ORDER))
            app.window.request_close()
            return
        self._live_index = index
        app.album.game = v3_live.apply(v3_live.ORDER[index],
                                        app.album.state, app.window)

    # ---- TASK_economy_1124: scripted walk over the real screens ------------

    def _tick_economy_1124(self, now: float) -> None:
        """``HCPD_ECONOMY_1124="step;step;..."`` on a scripted run (temp
        accounts dir, see ``HCPD_ECONOMY_SEED``/``_DIR``/``_SIGNIN``). Steps:
        ``go:<screen attr>`` (``F.MENU``...), ``level:N``, ``equip:<Item>``,
        ``click:<hotspot>`` (same path as a mouse click), ``seg:<name>``,
        ``shot:<stem>`` (framebuffer -> ``HCPD_ECONOMY_OUT``), ``log:<what>``
        (account numbers into the log), ``quit``. Each step waits 6 drawn
        frames AND 4 s so a shot shows the screen the step produced."""
        from src.ui_album.shots import save_framebuffer

        app = self._app
        if app.window.is_iconified():
            return
        steps = [s.strip() for s in self.env("HCPD_ECONOMY_1124").split(";") if s.strip()]
        # Art textures decode off-thread; give each step time, not a frame count.
        self._eco_hold = getattr(self, "_eco_hold", 0) + 1
        if self._eco_hold < 6 or now - getattr(self, "_eco_last_s", 0.0) < getattr(self, "_eco_wait", 4.0):
            return
        self._eco_hold = 0
        self._eco_last_s = now
        index = getattr(self, "_eco_index", 0)
        if index >= len(steps):
            return
        self._eco_index = index + 1
        step = steps[index]
        kind, _, arg = step.partition(":")
        # a toast lives 4 s: the shot right after a click/equip must come sooner
        self._eco_wait = 1.5 if kind in ("equip", "click", "item") else 4.0
        F = app.album.flow
        state = app.album.state
        if kind == "go":
            state.go(getattr(F, arg))
        elif kind == "level":
            state.level = int(arg)
        elif kind == "equip":
            F.toggle_item(state, arg)
        elif kind == "click":
            app.album_action(app.album.activate(arg), now)
        elif kind == "item":                       # same call as number key N
            app.use_item_slot(int(arg), now)
        elif kind == "type":
            for char in arg:                       # the same path as a key press
                app.album.key(char)
        elif kind == "seg":
            state.board_seg = arg
        elif kind == "shot":
            from OpenGL import GL

            GL.glFinish()
            out = Path(self.env("HCPD_ECONOMY_OUT") or app.root / "results"
                       / "economy_1124" / "shots")
            width, height = app.window.framebuffer_size()
            save_framebuffer(out / f"{arg}.png", width, height)
            logger.info("ECONOMY: shot %s (%dx%d) screen=%s msg=%r", arg, width,
                        height, state.screen, state.message)
        elif kind == "log":
            logger.info("ECONOMY: %s wallet=%s earned=%s cleared=%s items=%s",
                        arg, state.wallet, state.earned_total,
                        state.levels_cleared, state.items)
        elif kind == "quit":
            app.window.request_close()
        logger.info("ECONOMY: step %d/%d %s", index + 1, len(steps), step)

    def _grab_desktop(self, stem: str) -> None:
        """
        Ảnh này chụp **bằng tay**; ở đây chỉ dọn sân rồi nhắc.

        Đọc framebuffer không dùng được: thứ cần chứng minh là cái **nằm
        ngoài** cửa sổ (viền thật + desktop xung quanh). Còn ``PIL.ImageGrab``
        thì chụp cả desktop, mà Windows không cho tiến trình nền nhảy lên
        trên cửa sổ đang có focus — hai lượt thử ra đúng hai ảnh Chrome, lượt
        sau còn lọt cả Messenger riêng tư của người dùng.

        Nên hàm này **không ghi file**. Ghi thì lượt ``HCPD_V3_LIVE=1`` kế
        tiếp sẽ lặng lẽ đè mất tấm ảnh người dùng vừa chụp tay bằng một ảnh
        trình duyệt — hỏng còn tệ hơn là thiếu.
        """
        app = self._app
        path = app.root / "results" / "ui_v3_1107" / "live" / f"{stem}.png"
        logger.info(
            "UI v3 live: cua so dang o che do Window %s — chup tay bang "
            "Win+Shift+S roi luu de len %s", app.window.window_rect(), path)

    # ---- TASK_game_feel_1118 P6/P7: scope bench -----------------------------

    def _tick_scope_bench(self, now: float) -> None:
        """
        TASK_game_feel_1118 P7: ``HCPD_SCOPE_BENCH=1`` (dùng cùng
        ``HCPD_ALBUM_AUTOMATCH=1``, lấy sẵn khẩu Precision gun mặc định) —
        10 giây scoped, tâm ngắm chạy vòng tròn chậm, một phát mỗi 0.6 s.
        ``_tick_scope`` tự coi ``HCPD_SCOPE_BENCH=1`` là "đang giữ chuột
        phải" (xem ở đó), nên hàm này chỉ còn việc đưa tâm ngắm vòng tròn
        và bắn theo nhịp.
        """
        app = self._app
        if (self.env("HCPD_SCOPE_BENCH") != "1" or not app.album_options.is_v3
                or self.env("HCPD_EVIDENCE_1118") == "1"):
            # TASK_game_feel_fix_1118: the evidence hook flips
            # HCPD_SCOPE_BENCH itself (so _tick_scope treats it as "held"
            # for ITS OWN fixed crosshair) — this bench must not also
            # react to that and take mouse control away with its own
            # circular motion.
            return
        F = app.album.flow
        if app.album.state.screen != F.PLAY:
            return
        if not hasattr(self, "_scope_bench_start"):
            if not self._measurement_ready():
                return
            self._scope_bench_start = now
            self._scope_bench_last_shot = now
            self._scope_bench_last_frame_t = now
            self._scope_bench_frame_times: list = []
            return
        elapsed = now - self._scope_bench_start
        angle = elapsed * (2.0 * math.pi / 10.0)      # one slow turn over 10s
        circle_mouse = (640.0 + 150.0 * math.cos(angle),
                        360.0 + 150.0 * math.sin(angle))
        app.album.mouse = app.tick_scope(now, circle_mouse)
        self._scope_bench_frame_times.append(now - self._scope_bench_last_frame_t)
        self._scope_bench_last_frame_t = now
        if now - self._scope_bench_last_shot >= 0.6:
            self._scope_bench_last_shot += 0.6
            app.album_shoot(now)
        if elapsed >= 10.0:
            times = sorted(self._scope_bench_frame_times)
            n = len(times)
            median_dt = times[n // 2] if n else 0.0
            fps = (1.0 / median_dt) if median_dt > 0.0 else 0.0
            logger.info("SCOPE_BENCH: %d khung trong 10s, FPS median %.1f, "
                        "zoom cuoi %.2f", n, fps, app.album.scope_zoom)
            app.window.request_close()

    # ---- TASK_game_feel_fix_1118: evidence screenshot sequence --------------

    def _tick_evidence_1118(self, now: float) -> None:
        """``HCPD_EVIDENCE_1118=1`` (with ``HCPD_ALBUM_AUTOMATCH=1`` +
        ``HCPD_V3_SCREEN=v3_08_gameplay_hud``): one-off screenshot
        sequence into ``results/game_feel_fix_1118/`` for the fix
        report's evidence section, then self-closes — same spirit as
        ``HCPD_FX_BENCH``/``HCPD_SCOPE_BENCH``, not meant to ship."""
        app = self._app
        if self.env("HCPD_EVIDENCE_1118") != "1" or not app.album_options.is_v3:
            return
        F = app.album.flow
        if app.album.state.screen not in (F.PLAY, F.WIN):
            return
        if not hasattr(self, "_evidence_step"):
            if not self._measurement_ready():
                return
            self._evidence_step = 0
            self._evidence_step_t = now
            self._evidence_out = app.root / "results" / "game_feel_fix_1118"
            self._evidence_steps = self._build_evidence_steps()
            return
        if self._evidence_step >= len(self._evidence_steps):
            app.window.request_close()
            return
        name, setup, hold_s = self._evidence_steps[self._evidence_step]
        setup(app)
        if now - self._evidence_step_t >= hold_s:
            if name is not None:
                from OpenGL import GL

                from src.ui_album.shots import save_framebuffer

                app.window.restore_and_focus()
                app.window.pump_events()
                # Force an explicit, synchronous redraw against the
                # state THIS tick just set, instead of trusting whatever
                # the main loop's own next render happens to land on —
                # this hook runs during the TICK phase, one step removed
                # from the render/swap phase, and a one-shot state change
                # (go_win()) was observed to still show the PREVIOUS
                # screen's content without this. Reuse the last real
                # scene closure (Fix 9's own ``_last_scene``, same thing
                # ``_on_refresh`` redraws with) for steps that DO have a
                # 3D viewport (scope/HUD proximity) — a no-op would
                # render those as an empty void instead of the real scene.
                scene = app.last_scene or (lambda: None)
                if not app.album.has_viewport():
                    app.album.draw_hidden_scene(scene)
                app.album_draw_guarded(scene)
                width, height = app.window.framebuffer_size()
                if width >= 2 and height >= 2:
                    GL.glFinish()
                    save_framebuffer(self._evidence_out / f"{name}.png", width, height)
                    logger.info("EVIDENCE_1118: %s.png (screen=%s game=%s)",
                               name, app.album.state.screen,
                               app.album.game is not None)
                app.window.swap_buffers()
            self._evidence_step += 1
            self._evidence_step_t = now

    def _build_evidence_steps(self) -> list:
        def crosshair(xy):
            def setup(app):
                app.album.mouse = xy
            return setup

        def scoped(xy):
            # Drives the REAL _tick_scope pipeline (entering, camera
            # sync, zoom spring, finer-aim) with a FIXED raw mouse — not
            # just setting ``album.mouse``/``scope_zoom`` by hand, which
            # would move the 2D overlay number without ever warping the
            # 3D camera (exactly the 1118 report's own scope_in.png bug:
            # evidence that didn't come from an actually-magnified render).
            def setup(app):
                import os as _os

                _os.environ["HCPD_SCOPE_BENCH"] = "1"
                app.album.mouse = app.tick_scope(time.perf_counter(), xy)
            return setup

        def unscoped(xy):
            def setup(app):
                import os as _os

                _os.environ.pop("HCPD_SCOPE_BENCH", None)
                app.album.mouse = app.tick_scope(time.perf_counter(), xy)
            return setup

        def crit_impact(active):
            # ``crit_impact_alpha`` is a read-only property, computed from
            # ``_crit_impact_age`` — go through the real ``on_shot()``
            # entry point (same as an actual crit would) instead of
            # poking a private field, except for turning it back OFF
            # (nothing public does that early; the real path is just
            # waiting out CRIT_IMPACT_S).
            def setup(app):
                feel = app.album._shot_feel
                if active:
                    if feel._crit_impact_age is None:
                        feel.on_shot("Precision gun", (640.0, 360.0),
                                     hit=True, crit=True)
                else:
                    feel._crit_impact_age = None
            return setup

        def go_win():
            def setup(app):
                if app.album.state.screen != app.album.flow.WIN:
                    app.album.state.go(app.album.flow.WIN)
            return setup

        centre, corner, edge = (640.0, 360.0), (120.0, 120.0), (1180.0, 620.0)
        return [
            # G2: scope magnification, 3 crosshair positions, eye frozen.
            ("scope_proof_centre_unscoped", unscoped(centre), 1.0),
            ("scope_proof_centre_scoped", scoped(centre), 1.5),
            ("scope_proof_corner_unscoped", unscoped(corner), 1.0),
            ("scope_proof_corner_scoped", scoped(corner), 1.5),
            ("scope_proof_edge_unscoped", unscoped(edge), 1.0),
            ("scope_proof_edge_scoped", scoped(edge), 1.5),
            # G3: crit impact frame vs normal, crosshair away from cards.
            # Reset the scope zoom spring back to 1.0 first — the edge
            # step above left it scoped, and nothing after this touches
            # _tick_scope again, so a stale zoom would leak into every
            # later step's vignette otherwise.
            (None, unscoped(centre), 0.6),
            ("hud_layer_check_normal", crit_impact(False), 0.5),
            ("hud_layer_check_crit",
             lambda app: (crit_impact(True)(app), crosshair((640.0, 360.0))(app)),
             0.5),
            (None, crit_impact(False), 0.1),   # reset before continuing
            # G12: proximity fade (crosshair over a weapon card).
            ("hud_proximity", crosshair((1130.0, 650.0)), 1.0),
            # G3/10.3: scoped near the user's own screenshot spot.
            ("hud_scope", scoped((980.0, 560.0)), 1.5),
            (None, unscoped(centre), 0.6),
            # Fix 7: star_five on WIN.
            # star_five's own duration is 0.95s — capture mid-burst, not
            # after it's already faded to its final tiny-particle frames.
            ("fx_win", go_win(), 0.5),
        ]

    # ---- TASK_hud_fix_1119: HUD fix evidence screenshot sequence ------------

    def _tick_hud_fix_1119_shots(self, now: float) -> None:
        """``HCPD_HUD_FIX_1119_SHOTS=1`` (with ``HCPD_ALBUM_AUTOMATCH=1``):
        one-off screenshot sequence into ``results/hud_fix_1119/`` for Fix
        1/2's own evidence (the designed HUD with every card, proximity
        fade, H-hidden, scoped) — same spirit as ``HCPD_EVIDENCE_1118``,
        not meant to ship."""
        app = self._app
        if (self.env("HCPD_HUD_FIX_1119_SHOTS") != "1"
                or not app.album_options.is_v3):
            return
        F = app.album.flow
        if app.album.state.screen != F.PLAY:
            return
        if not hasattr(self, "_hud1119_step"):
            if not self._measurement_ready():
                return
            self._hud1119_step = 0
            self._hud1119_step_t = now
            self._hud1119_out = app.root / "results" / "hud_fix_1119"
            self._hud1119_steps = self._build_hud_fix_1119_steps()
            return
        if self._hud1119_step >= len(self._hud1119_steps):
            app.window.request_close()
            return
        name, setup, hold_s = self._hud1119_steps[self._hud1119_step]
        setup(app)
        if now - self._hud1119_step_t >= hold_s:
            if name is not None:
                from OpenGL import GL

                from src.ui_album.shots import save_framebuffer

                app.window.restore_and_focus()
                app.window.pump_events()
                scene = app.last_scene or (lambda: None)
                if not app.album.has_viewport():
                    app.album.draw_hidden_scene(scene)
                app.album_draw_guarded(scene)
                width, height = app.window.framebuffer_size()
                if width >= 2 and height >= 2:
                    GL.glFinish()
                    save_framebuffer(self._hud1119_out / f"{name}.png", width, height)
                    logger.info("HUD_FIX_1119: %s.png", name)
                app.window.swap_buffers()
            self._hud1119_step += 1
            self._hud1119_step_t = now

    def _build_hud_fix_1119_steps(self) -> list:
        def crosshair(xy):
            def setup(app):
                app.album.mouse = xy
            return setup

        def scoped(xy):
            def setup(app):
                import os as _os

                _os.environ["HCPD_SCOPE_BENCH"] = "1"
                app.album.mouse = app.tick_scope(time.perf_counter(), xy)
            return setup

        def hidden(want: bool):
            def setup(app):
                if app.album.hud_hidden != want:
                    app.album.toggle_hud_hidden()
            return setup

        def weapon(name):
            def setup(app):
                self._apply_bench_weapon(name)
            return setup

        def clear_toast():
            def setup(app):
                app.album.state.message = ""
            return setup

        centre, weapon_card = (640.0, 360.0), (1130.0, 650.0)
        return [
            ("hud_full", crosshair(centre), 1.0),
            ("hud_proximity", crosshair(weapon_card), 1.0),
            (None, hidden(True), 0.05),
            ("hud_hidden", crosshair(centre), 0.3),
            (None, hidden(False), 0.05),
            (None, weapon("Precision gun"), 0.1),
            # the H-hidden toast above is a timed (6s) message, not
            # cleared by un-hiding (Fix 2: "nothing on show") -- clear it
            # by hand so it doesn't linger into the scope screenshot.
            (None, clear_toast(), 0.1),
            ("hud_scope", scoped(centre), 1.5),
        ]

    # ---- TASK_hardening_1124: item-slot fade shots + bad-photo check ---------

    def _shoot_framebuffer(self, path: Path) -> None:
        """One F12-style shot of the real window (same path as the other hooks)."""
        from OpenGL import GL

        from src.ui_album.shots import save_framebuffer

        app = self._app
        app.window.restore_and_focus()
        app.window.pump_events()
        scene = app.last_scene or (lambda: None)
        if not app.album.has_viewport():
            app.album.draw_hidden_scene(scene)
        app.album_draw_guarded(scene)
        width, height = app.window.framebuffer_size()
        if width >= 2 and height >= 2:
            GL.glFinish()
            save_framebuffer(path, width, height)
        app.window.swap_buffers()

    def _bad_photo_1124(self, now: float) -> bool:
        """``HCPD_BAD_PHOTO_1124=<file>``: before the automatch loads its own
        photo, push this (bad) file through the real loader ``_try_load_media``
        and shoot the window with the error toast on it. True while busy."""
        app = self._app
        bad = self.env("HCPD_BAD_PHOTO_1124").strip()
        if not bad or getattr(self, "_bad_photo_done", False):
            return False
        out = Path(self.env("HCPD_HARDENING_1124_OUT")
                   or app.root / "results" / "hardening_1124" / "g5")
        if not hasattr(self, "_bad_photo_t"):
            if not self._measurement_ready():
                return True
            app.try_load_media(Path(bad), now)
            self._bad_photo_t = now
            logger.info("BAD_PHOTO_1124: loaded %s through _try_load_media", bad)
            return True
        if now - self._bad_photo_t < 0.7:
            return True
        self._shoot_framebuffer(out / "g5_bad_file_error.png")
        logger.info("BAD_PHOTO_1124: shot g5_bad_file_error.png")
        self._bad_photo_done = True
        return False

    def _tick_hardening_1124_shots(self, now: float) -> None:
        """``HCPD_HARDENING_1124_SHOTS=1`` (with ``HCPD_ALBUM_AUTOMATCH=1`` and
        ``HCPD_AUTOMATCH_ITEMS`` of two items): the item-slot fade evidence,
        shots of the real window into ``HCPD_HARDENING_1124_OUT``. Each step
        also logs the live card alphas, so the pixel check has the numbers
        the UI itself used."""
        app = self._app
        if self.env("HCPD_HARDENING_1124_SHOTS") != "1" or not app.album_options.is_v3:
            return
        F = app.album.flow
        if app.album.state.screen != F.PLAY:
            return
        if not hasattr(self, "_h24_step"):
            if not self._measurement_ready():
                return
            self._h24_step, self._h24_t = 0, now
            self._h24_out = Path(self.env("HCPD_HARDENING_1124_OUT")
                                 or app.root / "results" / "hardening_1124" / "g3")
            self._h24_steps = self._build_hardening_1124_steps()
            return
        if self._h24_step >= len(self._h24_steps):
            app.window.request_close()
            return
        name, setup, hold_s = self._h24_steps[self._h24_step]
        setup(app)
        app.album.state.message = ""      # the H-hide toast must not sit on a slot
        if now - self._h24_t >= hold_s:
            if name is not None:
                self._shoot_framebuffer(self._h24_out / f"{name}.png")
                logger.info("HARDENING_1124: %s.png alphas=%s", name,
                            {k: round(v, 3) for k, v in app.album.hud_card_alpha.items()})
            self._h24_step += 1
            self._h24_t = now

    def _tick_reload_1124(self, now: float) -> None:
        """``HCPD_RELOAD_1124=1`` (with automatch): fire 10 rounds, then press
        R through the window's real key flag, then close once the reload is
        over. The numbers come from the ``Reload start/end`` log lines."""
        app = self._app
        if self.env("HCPD_RELOAD_1124") != "1":
            return
        match = app.album.game
        if match is None or app.album.state.screen != app.album.flow.PLAY:
            return
        if not hasattr(self, "_rl_phase"):
            if not self._measurement_ready():
                return
            self._rl_phase, self._rl_fired = "fire", 0
            return
        if self._rl_phase == "fire":
            if match.ready_to_fire(now) and match.take_round(now):
                self._rl_fired += 1
            if self._rl_fired >= 10:
                logger.info("RELOAD_1124: fired 10, rounds %d/%d, pressing R",
                            match.rounds_left, match.mag_size)
                import glfw
                app.window._on_key(None, glfw.KEY_R, 0, glfw.PRESS, 0)  # the real key callback
                logger.info("RELOAD_1124: focus=%r forgot_open=%r field_focused=%r",
                            app.album.state.focus, app.album.state.forgot_open,
                            app.v3_field_focused())
                self._rl_phase, self._rl_t = "wait", now
        elif self._rl_phase == "wait":
            if now - self._rl_t > match.reload_s + 1.5:
                logger.info("RELOAD_1124: after, rounds %d/%d",
                            match.rounds_left, match.mag_size)
                app.window.request_close()
                self._rl_phase = "done"

    def _build_hardening_1124_steps(self) -> list:
        def crosshair(xy):
            def setup(app):
                app.album.mouse = xy
            return setup

        def scoped(xy):
            def setup(app):
                import os as _os

                _os.environ["HCPD_SCOPE_BENCH"] = "1"
                app.album.mouse = app.tick_scope(time.perf_counter(), xy)
            return setup

        def hidden(want: bool):
            def setup(app):
                if app.album.hud_hidden != want:
                    app.album.toggle_hud_hidden()
            return setup

        def weapon(name):
            def setup(app):
                self._apply_bench_weapon(name)
            return setup

        def clear_toast():
            def setup(app):
                app.album.state.message = ""
            return setup

        far, slot, health = (640.0, 330.0), (640.0, 628.0), (204.0, 626.0)
        return [
            (None, clear_toast(), 0.1),
            ("a_far", crosshair(far), 1.2),
            (None, hidden(True), 0.6),
            ("bg_unscoped", crosshair(far), 0.4),
            (None, hidden(False), 0.6),
            ("b_slot", crosshair(slot), 1.2),
            ("c_health", crosshair(health), 1.2),
            (None, weapon("Precision gun"), 0.1),
            (None, clear_toast(), 0.1),
            ("d_scoped", scoped(far), 1.8),
            (None, hidden(True), 0.6),
            ("bg_scoped", scoped(far), 0.4),
        ]

    # ---- TASK_hud_fix_1119 Fix 4 / Part A: automatch + autowin --------------

    def _tick_album_automatch(self, now: float) -> None:
        """
        Móc đo ``HCPD_ALBUM_AUTOMATCH=1``: tự mở một ván khi lớp đã xong.

        Để **tự kiểm được bằng mắt** mà không phải bấm qua cả luồng — người
        dùng 07/11: "không thấy bọ ở đâu cả, không tin bạn có thể tự test".
        """
        app = self._app
        F = app.album.flow

        if self.env("HCPD_ALBUM_AUTOMATCH") not in ("1", "2"):
            return
        if getattr(self, "_automatch_fired", False):
            # TASK_automatch_regress_1121: a won/lost match clears
            # ``app.album.game`` and leaves the result screen (see
            # ``_tick_album_match``'s ``state.go(F.WIN/F.LOSE)``) -- to
            # this hook that looks identical to the cold-launch MENU
            # state below, so without this guard it jumps back to PLAY
            # and starts a new match every tick, forever.
            return
        if app.album.game is not None or app.renderer.waiting_for_depth:
            return
        if app.album.state.screen != F.PLAY:
            # TASK_hud_fix_1119 Fix 4: accounts added a sign-in screen
            # ahead of MENU since this hook was written -- a cold
            # ``--album-guest`` launch now lands on MENU, not PLAY, so
            # "no clicking through the whole flow" needs to jump there
            # itself instead of just waiting for it.
            app.album.state.go(F.PLAY)
            return
        if self._bad_photo_1124(now):
            return
        photo = self.env("HCPD_AUTOMATCH_PHOTO").strip()
        if photo and not getattr(self, "_automatch_photo_loaded", False):
            # TASK_automatch_regress_1121: lets a baseline/evidence run
            # pick ITS OWN test photo (one that hides every bug the
            # level asks for) instead of whatever ``content.game.json``
            # happens to point at -- ``_try_load_media`` never writes
            # that file for ``--album`` (see its own PB-055/1121 guard),
            # so this is as config-safe as the hook it modifies.
            self._automatch_photo_loaded = True
            app.try_load_media(Path(photo), now)
            return
        self._automatch_fired = True
        app.album.state.weapon = 0
        if self.env("HCPD_ALBUM_AUTOMATCH") == "2":
            # Bọ đặt TAY ở chỗ biết trước: để kiểm riêng phần VẼ sprite, tách
            # khỏi phần tìm chỗ giấu (place_props có ảnh trả về 0 chỗ).
            from src.content.hidden_props import layer_planes
            from src.content.prop_game import GameState, bug_radius_mm
            from src.game.match import start_match

            radius = bug_radius_mm(app.content)
            props = [{"pos_mm": (x, 0.0, -180.0), "radius_mm": radius,
                      "reveal_axis": "x", "reveal_450_mm": 0.0,
                      "reveal_330_mm": 0.0} for x in (-70.0, 0.0, 70.0)]
            app.game = GameState(image="debug", props=props,
                                  tracking_on=True, started_at_s=now)
            app.game_planes = layer_planes(app.renderer.layer_records())
            # Không gán số hiệu bộ lớp thì _sync_prop_game bỏ ván ngay khung sau.
            app.game_layers_revision = app.renderer.layers_revision
            app.renderer.set_props([])
            if app.bugs is not None:
                app.bugs.reset()
            app.album.game = start_match(1, app.game, "Sung", [])
            logger.info("DEBUG MATCH: dat tay %d bo", len(props))
            return
        items = [n.strip() for n in self.env("HCPD_AUTOMATCH_ITEMS").split(",")
                 if n.strip()]
        for name in items:
            F.toggle_item(app.album.state, name)
        if items:
            logger.info("AUTOMATCH: items %s", app.album.state.items)
        app.start_album_match(now)
        # TASK_hud_fix_1119 Fix 4: the real-click SendInput check needs a
        # SPECIFIC gun (Spread, mag 8) rather than automatch's always-
        # Precision default -- reuses ``_apply_bench_weapon`` (until now
        # only reachable via ``HCPD_FX_BENCH``) so ``HCPD_BENCH_WEAPON``
        # alone, without the 20 s synthetic-fire loop, picks the weapon.
        weapon = self.env("HCPD_BENCH_WEAPON").strip()
        if weapon:
            from src.game.match import WEAPONS_SPEC

            if weapon in WEAPONS_SPEC:
                self._apply_bench_weapon(weapon)

    def _tick_album_autowin(self, now: float) -> None:
        """TASK_arch_refactor_1120 Part A: ``HCPD_ALBUM_AUTOWIN=1`` — once
        ``_tick_album_automatch`` (or any other hook) has a real match
        running on ``PLAY``, hạ từng con bọ một (mỗi 0.1 s) cho tới khi
        ván thắng, để kiểm được đường thật từ đầu tới màn kết quả (lưu
        tiến trình) mà không phải rà chuột khắp khung hình đoán chỗ bọ
        ẩn. Dùng đúng ``AlbumMatch.apply_damage`` sản phẩm dùng mỗi phát
        bắn — chỉ khác chỗ chọn mục tiêu (ở đây là tuần tự theo chỉ số,
        không phải dò theo tâm ngắm)."""
        app = self._app
        if self.env("HCPD_ALBUM_AUTOWIN") != "1":
            return
        match = app.album.game
        F = app.album.flow
        if match is None or app.album.state.screen != F.PLAY:
            return
        if app.game is None or not app.game.props:
            return
        if now < getattr(self, "_autowin_next_s", 0.0):
            return
        # ``HCPD_ALBUM_AUTOWIN_GAP_S``: a slower pace, so the central server's
        # own "too fast to be human" check (MIN_WIN_S_PER_BUG) can be passed.
        try:
            gap = float(self.env("HCPD_ALBUM_AUTOWIN_GAP_S") or 0.1)
        except ValueError:
            gap = 0.1
        self._autowin_next_s = now + max(0.1, gap)
        for index, found in enumerate(match.state.found):
            if found:
                continue
            killed = match.apply_damage(index, 1_000_000.0, (0.0, 0.0), now)
            if killed:
                match.state.found[index] = True
                match.last_shot_hits = [(index, False, True)]
                app.renderer.set_prop_found(index)
                if app.bugs is not None:
                    app.bugs.mark_found(
                        index, [v / 1000.0 for v in app.game.props[index]["pos_mm"]])
                logger.info("AUTOWIN: ha bo #%d (%d/%d)", index,
                            match.bugs_found, match.bugs_total)
            break

    # ---- TASK_game_feel_fix_1118: measurement hygiene -----------------------

    def _measurement_ready(self) -> bool:
        """
        TASK_game_feel_fix_1118 "Measurement hygiene": a bench/screenshot
        hook's measured window starts only once ALL of — fx preload done,
        depth layers built, 2 s have passed since — hold. Logs the wait
        once. Benches that measured through the preload/layer-build
        window (the 1118 report's own ``bench_precision.log``: preload
        still finishing 11 s in, 16 skipped spawns, first 5 s at 16.6
        FPS) were measuring one-time launch cost, not steady-state play.
        """
        app = self._app
        fx = getattr(app.album, "_fx_library", None)
        fx_ready = fx is None or not app.album.fx_enabled or fx.preload_remaining == 0
        depth_ready = not app.renderer.waiting_for_depth
        now = time.perf_counter()
        if not (fx_ready and depth_ready):
            self._measurement_ready_since = None
            return False
        since = getattr(self, "_measurement_ready_since", None)
        if since is None:
            self._measurement_ready_since = now
            logger.info("Do luong: fx+lop da xong, cho on dinh 2s truoc khi bat dau do")
            return False
        return (now - since) >= 2.0

    # ---- TASK_game_feel_1118 P7/TASK_game_feel_fix_1118: fx bench -----------

    def _tick_fx_bench(self, now: float) -> None:
        """
        Móc đo ``HCPD_FX_BENCH=1`` (dùng cùng ``HCPD_ALBUM_AUTOMATCH=1``):
        4 phát/giây trong 20 giây trên ``PLAY``, xoay vòng giết/chí mạng/hụt
        — qua ĐÚNG API Sakuga thật (``spawn_hit_fx``/``spawn_miss_fx``),
        không giả bằng cách bắn thật (ngắm trúng/hụt không kiểm soát được) —
        rồi ghi khung hình để tính FPS median (TASK_ui_sakuga_1117 G2).

        TASK_game_feel_1118 P7: ``HCPD_BENCH_WEAPON`` (one of
        ``WEAPONS_SPEC``'s keys) also exercises that weapon's
        ``shot_feel`` (kick/shake/hit marker) via ``register_shot`` on
        each cycled shot — and for the Heavy-mag gun specifically, holds
        the (fake) trigger down the whole 20 s so the REAL
        ``_tick_auto_fire`` path fires, instead of the synthetic cycle.

        TASK_game_feel_fix_1118 "Measurement hygiene": does not start its
        20 s window until ``_measurement_ready()`` holds — see there.
        Also logs, for the single slowest frame in the window, how much
        of it was fx GL-thread work (mipmap+upload) versus everything
        else, and the single slowest decode/bleed/mipmap/upload file
        recorded across the whole window (G6's own check).
        """
        app = self._app
        if self.env("HCPD_FX_BENCH") != "1" or not app.album_options.is_v3:
            return
        F = app.album.flow
        if app.album.state.screen != F.PLAY:
            return
        if not hasattr(self, "_fx_bench_start"):
            if not self._measurement_ready():
                return
            self._fx_bench_start = now
            self._fx_bench_last_shot = now
            self._fx_bench_last_frame_t = now
            self._fx_bench_frame_times: list = []
            self._fx_bench_fx_times: list = []
            self._fx_bench_cycle = 0
            requested = self.env("HCPD_BENCH_WEAPON").strip()
            from src.game.match import WEAPONS_SPEC

            self._fx_bench_weapon = requested if requested in WEAPONS_SPEC else None
            self._fx_bench_weapon_applied = False
            return
        weapon = getattr(self, "_fx_bench_weapon", None)
        if weapon and not self._fx_bench_weapon_applied and app.album.game is not None:
            self._apply_bench_weapon(weapon)
            self._fx_bench_weapon_applied = True
        self._fx_bench_frame_times.append(now - self._fx_bench_last_frame_t)
        self._fx_bench_last_frame_t = now
        fx = getattr(app.album, "_fx_library", None)
        self._fx_bench_fx_times.append(
            (fx.frame_mipmap_s + fx.frame_upload_s) if fx is not None else 0.0)
        if weapon == "Heavy-mag gun":
            # P7: "holds the button so auto-fire is exercised" — the real
            # click-and-hold path (``_tick_auto_fire``), not the
            # synthetic cycle below.
            app.album.mouse_down = True
        elif now - self._fx_bench_last_shot >= 0.25:       # 4 phat/s
            self._fx_bench_last_shot += 0.25
            kind = self._fx_bench_cycle % 3
            self._fx_bench_cycle += 1
            if weapon:
                app.album.register_shot(kind != 2, kind == 1)
            if kind == 0:
                app.album.spawn_hit_fx(False)
            elif kind == 1:
                app.album.spawn_hit_fx(True)
            else:
                app.album.spawn_miss_fx()
        if now - self._fx_bench_start >= 20.0:
            times = sorted(self._fx_bench_frame_times)
            n = len(times)
            median_dt = times[n // 2] if n else 0.0
            fps = (1.0 / median_dt) if median_dt > 0.0 else 0.0
            p99_dt = times[min(n - 1, int(round(0.99 * (n - 1))))] if n else 0.0
            hitches = sum(1 for t in times if t > 0.050)
            logger.info("FX_BENCH%s: %d khung trong 20s, FPS median %.1f, "
                        "p99 %.1f ms, khung >50ms: %d, khung cham nhat %.1f ms",
                        f" ({weapon})" if weapon else "", n, fps,
                        p99_dt * 1000.0, hitches,
                        max(times) * 1000.0 if times else 0.0)
            self._log_slowest_frame_stage(self._fx_bench_frame_times,
                                          self._fx_bench_fx_times, fx)
            app.window.request_close()

    def _log_slowest_frame_stage(self, frame_times: list, fx_times: list, fx) -> None:
        """TASK_game_feel_fix_1118 Fix 2: breaks down the single slowest
        frame in a bench window into fx GL-thread cost vs everything
        else, and names the single slowest file per stage across the
        whole window — "no frame > 50ms whose slowest stage is fx
        decode, bleed or upload" (G6) needs this to actually check."""
        if not frame_times:
            return
        worst_i = max(range(len(frame_times)), key=lambda i: frame_times[i])
        worst_ms = frame_times[worst_i] * 1000.0
        fx_ms = (fx_times[worst_i] * 1000.0) if worst_i < len(fx_times) else 0.0
        logger.info(
            "Khung cham nhat: %.1f ms tong, fx (mipmap+upload) %.1f ms, "
            "con lai %.1f ms", worst_ms, fx_ms, worst_ms - fx_ms)
        if fx is not None:
            for stage in ("decode_s", "bleed_s", "mipmap_s", "upload_s"):
                found = fx.slowest_file(stage)
                if found is not None:
                    key, seconds = found
                    logger.info("  %s cham nhat: %s (%.1f ms)",
                               stage, key, seconds * 1000.0)

    def _apply_bench_weapon(self, weapon_name: str) -> None:
        """TASK_game_feel_1118 P7: force the live match onto
        ``weapon_name`` (automatch always picks the Precision gun) with a
        fresh magazine, for ``HCPD_BENCH_WEAPON``."""
        from src.game.match import WEAPONS_SPEC

        match = self._app.album.game
        if match is None:
            return
        spec = WEAPONS_SPEC[weapon_name]
        match.weapon_name = weapon_name
        match.mag_size = spec.mag
        match.rounds_left = spec.mag
        match.fire_interval_s = spec.fire_s
        match.reload_s = spec.reload_s
        match.reload_until_s = 0.0
        match.next_shot_s = 0.0
        logger.info("FX_BENCH: ep vu khi %s (bang %d, nhip %.2fs, nap %.1fs)",
                    weapon_name, spec.mag, spec.fire_s, spec.reload_s)

    # ---- TASK_pvp_lan_1116: PvP autoplay + screenshots ----------------------

    def _tick_pvp_auto(self, now: float) -> None:
        """
        Móc đo ``HCPD_PVP_AUTO=1``: tự đi qua sảnh PvP — y hệt
        ``HCPD_ALBUM_AUTOMATCH`` nhưng cho PvP (Find match với mode/level
        hiện tại, không cần bấm gì). Sau khi một trận xong (``PVP_RESULT``),
        quay lại tìm trận tiếp — để đo G2 "3/3 trận" bằng một lần chạy liên
        tục, khớp ``pvp_bot.py --matches 3``.
        """
        app = self._app
        if self.env("HCPD_PVP_AUTO") != "1":
            return
        from src.ui_album import v3_routes

        F = app.album.flow
        state = app.album.state
        if state.screen in F.LOCKED:
            state.player = state.player or "guest"
            state.go(F.MENU)
        elif state.screen == F.MENU:
            state.go(F.PVP_MODE)
        elif state.screen == F.PVP_MODE:
            # HCPD_PVP_MODE (tuỳ chọn, P7 report only): đổi mode trước khi
            # tìm trận — mặc định là mode hiện tại của flow (Competitive).
            wanted = self.env("HCPD_PVP_MODE") or None
            if wanted and wanted in F.PVP_MODES and state.mode != wanted:
                v3_routes.on_click(state, f"mode_{wanted}", now)
                return
            # Qua đúng cổng một cú bấm thật đi qua (on_click), không gọi
            # thẳng _pvp_action: _pvp()'s F.start_search() (đổi màn sang
            # PVP_SEARCH) phải chạy, không chỉ phần mạng.
            action = v3_routes.on_click(state, ID.BTN_FIND_MATCH, now)
            app.pvp_action(action, now)
        elif state.screen == F.PVP_RESULT:
            # Đứng lại 2 s trước khi quay lại tìm trận tiếp — không chỉ cho
            # _tick_pvp_shots kịp chụp (PVP_RESULT biến mất ngay khung đầu
            # nếu không thì, trước khi hẹn giờ của _tick_pvp_shots tới
            # lượt), còn để kết quả hiện đủ lâu để đọc được.
            entered = getattr(self, "_pvp_auto_result_entered_s", None)
            if entered is None:
                self._pvp_auto_result_entered_s = now
            elif now - entered >= 2.0:
                self._pvp_auto_result_entered_s = None
                state.go(F.PVP_MODE)
        else:
            # Rời PVP_RESULT bằng đường khác (vd mất kết nối) — dọn mốc giờ
            # để lần SAU vào lại PVP_RESULT tính giờ lại từ đầu, không bị
            # mốc cũ làm rời màn ngay khung đầu.
            self._pvp_auto_result_entered_s = None

    def _tick_pvp_shots(self, now: float) -> None:
        """
        ``HCPD_PVP_SHOTS=1``: chụp bốn màn PvP cho báo cáo P7, ra
        ``results/pvp_1116/<man>.png`` — F12 là một phương thức bình
        thường (``_save_album_shot``), không buộc phải có phím bấm thật,
        xem ``_album_shots_step`` (đã làm y vậy cho ``--ui-shots``).

        Mỗi màn chụp MỘT lần, sau khi đã đứng yên ở đó một lúc (để HUD
        kịp cập nhật — "giữa trận" chứ không phải khung đầu tiên trống
        trơn), và PVP_FOUND chỉ chụp khi đã có tên đối thủ thật.
        """
        app = self._app
        if self.env("HCPD_PVP_SHOTS") != "1":
            return
        F = app.album.flow
        state = app.album.state
        screen = state.screen
        delay_s = {F.PVP_FOUND: 0.5, F.PVP_HUNT: 2.0, F.PVP_HIDE: 2.0,
                  F.PVP_RESULT: 0.5}.get(screen)
        if delay_s is None:
            return
        taken = getattr(self, "_pvp_shots_taken", None)
        if taken is None:
            taken = set()
            self._pvp_shots_taken = taken
        if screen in taken:
            return
        if getattr(self, "_pvp_shot_last_screen", None) != screen:
            self._pvp_shot_last_screen = screen
            self._pvp_shot_entered_s = now
        if now - getattr(self, "_pvp_shot_entered_s", now) < delay_s:
            return
        if screen == F.PVP_FOUND and not state.pvp_opponent_name:
            return
        from OpenGL import GL

        GL.glFinish()
        self._save_pvp_shot(screen)
        taken.add(screen)

    def _save_pvp_shot(self, screen: str) -> Path:
        """F12-equivalent to ``results/pvp_1116/<screen>.png`` — a separate
        destination from ``_save_album_shot``'s ``results/ui_v3_1107/``
        (that one feeds ``--ui-shots``/``HCPD_V3_LIVE`` tooling; this is
        only for the PvP gate screenshots)."""
        from src.ui_album.shots import save_framebuffer

        app = self._app
        path = app.root / "results" / "pvp_1116" / f"{screen}.png"
        app.window.restore_and_focus()
        app.window.pump_events()
        width, height = app.window.framebuffer_size()
        if width < 2 or height < 2:
            logger.warning("PvP shot bo qua: framebuffer %dx%d", width, height)
            return path
        save_framebuffer(path, width, height)
        logger.info("PvP shot: %s", path)
        return path

    # ---- --album-shots / --ui-shots: whole-UI screenshot sequence -----------

    def _album_shots_step(self) -> None:
        """
        ``--album-shots``: mỗi màn vẽ đủ 3 khung rồi mới đọc pixel.

        Ba điều đã làm hỏng lượt chụp đầu (18/50 ảnh đen, 6 nhóm trùng md5):
        đọc ngay khung đầu sau khi đổi màn, đọc theo cỡ cửa sổ đã rơi về
        1280×720 lúc cửa sổ bị thu nhỏ, và đồng hồ mô phỏng tự nhảy màn.

        TASK_perf_textures_1119 Fix 3: texture của màn giờ nạp NGOÀI
        LUỒNG — "đủ 3 khung" không còn chắc nghĩa là "đã xong" (trước
        đây ``bind_screen`` chặn luồng tới khi xong nên 3 khung luôn
        thừa). Phải tự hỏi ``current_screen_ready()``; đếm 3 khung chỉ
        bắt đầu SAU khi nó thật đã xong.
        """
        from OpenGL import GL

        app = self._app
        if app.window.is_iconified():
            logger.warning("Album shots: cua so dang thu nho — cho")
            return
        if app.album_options.is_v3 and not app.album.current_screen_ready():
            return
        self._shot_hold = getattr(self, "_shot_hold", 0) + 1
        if self._shot_hold < 3:
            return
        self._shot_hold = 0
        index = getattr(self, "_shot_index", 0)
        plan = app.album.shot_plan()
        entry = plan[index]
        # Bộ v3 chỉ có MỘT theme, nên kế hoạch của nó là danh sách tên màn
        # phẳng; bộ cũ là cặp (màn, theme).
        stem = entry if isinstance(entry, str) else f"{entry[0]}_{entry[1]}"
        GL.glFinish()
        app.save_album_shot(stem)
        if index + 1 >= len(plan):
            logger.info("UI shots: xong %d anh", len(plan))
            app.window.request_close()
            return
        self._shot_index = index + 1
        app.album.apply_shot(index + 1)

    # ---- TASK_ui_sakuga_1117 G5: 20 ảnh hover/press/focus/tilt/fx --------

    def _sakuga_shot_plan(self) -> list:
        """20 bước, mỗi bước ``(tên file, setup(), số khung giữ trước khi
        chụp)``. ``setup()`` chỉ chạy Ở KHUNG ĐẦU của bước — sau đó nhịp
        bình thường (``album.tick()``/``draw()``) tự chạy tiếp, đúng như
        người chơi thật đứng yên ở trạng thái đó."""
        app = self._app
        F = app.album.flow
        # HOVER_HOLD: đủ để spring hover/press ổn định (k=420,d=26 lắng
        # trong vài chục khung). AUTO_FX_HOLD: cho hiệu ứng tự spawn qua
        # ``_step_fx_triggers`` (loading/toast/win/primary) đủ thời gian
        # decode+spawn (~0,3-0,5s) VẪN còn sống khi chụp (đời hiệu ứng
        # 0,4-1,3s) — khác FX_HOLD cũ 150 khung (~4s ở 37fps), đủ lâu để
        # MỌI hiệu ứng chết hẳn trước khi chụp (ba ảnh fx_kill/fx_crit/
        # fx_miss ra giống hệt nhau lúc kiểm — xem báo cáo G5).
        HOVER_HOLD, AUTO_FX_HOLD = 90, 25

        def hotspot_rect(screen, name):
            from src.ui_album.v3_hit import hotspot_rect as _rect

            for hotspot in app.album.manifest.screens[screen].get(
                    "hotspots", ()):
                if hotspot["n"] == name:
                    return _rect(screen, hotspot)
            return None

        def hover_at(screen, name, press=False):
            def setup():
                app.album.state.go(screen)
                app.album.state.focus = None
                rect = hotspot_rect(screen, name)
                if rect is not None:
                    app.album.mouse = (rect[0] + rect[2] * 0.5,
                                        rect[1] + rect[3] * 0.5)
                app.album.mouse_down = press
            return setup

        def focus_at(screen, name):
            def setup():
                app.album.state.go(screen)
                app.album.mouse = (-1000.0, -1000.0)
                app.album.mouse_down = False
                app.album.state.focus = name
            return setup

        def tilt_state(face, left, right):
            def setup():
                app.album.state.go(F.TILT)
                app.album.state.face_seen = face
                app.album.state.tilt_left_done = left
                app.album.state.tilt_right_done = right
                app.album.mouse = (-1000.0, -1000.0)
            return setup

        def fx_shot(screen, name, point, **kwargs):
            def setup():
                app.album.state.go(screen)
                app.album.mouse = (-1000.0, -1000.0)
                # TASK_game_feel_1118 P3: fx_kill/fx_crit/fx_miss all fire
                # on the SAME screen back to back — a screen change is
                # what normally clears the previous shot's effect, so
                # without this the next shot's screenshot still shows it
                # (fx_miss.png showing fx_crit's red DON).
                app.album.clear_fx()
                self._fx_shot_pending = (name, point, kwargs)
            return setup

        def loading_state(pct):
            def setup():
                app.album.state.go(F.LOADING)
                app.album.state.pipeline_pct = pct
                app.album.state.layer_count = 4
                app.album.mouse = (-1000.0, -1000.0)
            return setup

        def primary_click(screen):
            def setup():
                app.album.state.go(screen)
                app.album.mouse = (-1000.0, -1000.0)
                app.album.submit()
            return setup

        def toast_shot():
            def setup():
                app.album.state.go(F.MENU)
                app.album.mouse = (-1000.0, -1000.0)
                F.say(app.album.state, "Reloading")
            return setup

        def win_shot():
            def setup():
                # TASK_game_feel_1118 P3: prove the match-toast-leak fix —
                # a "Reloading" toast set while actually ON PLAY must NOT
                # survive the PLAY -> WIN transition (1117's own fx_win.png
                # showed it lingering; a MENU -> WIN jump never exercises
                # that path at all).
                app.album.state.go(F.PLAY)
                F.say(app.album.state, "Reloading")
                app.album.state.last_result = {
                    "found": 3, "total": 3, "score": 120,
                    "time_left": 10, "used": {}}
                app.album.state.go(F.WIN)
                app.album.mouse = (-1000.0, -1000.0)
            return setup

        return [
            # TASK_game_feel_1118 P3: MENU has no "tab_Solo" hotspot (its
            # four big cards are "btn_Solo" etc — the top tab bar only
            # exists on LEVELS/PHOTOS/LOADOUT/TILT/SETTINGS/...), so the
            # old shot never moved the mouse and just showed a plain menu.
            ("hover_tab_solo", hover_at(F.LEVELS, ID.TAB_SOLO), HOVER_HOLD),
            ("hover_display_full", hover_at(F.SETTINGS, ID.DISPLAY_FULL), HOVER_HOLD),
            ("hover_link_guest", hover_at(F.SIGN_IN, ID.LINK_GUEST), HOVER_HOLD),
            ("hover_level_row_2", hover_at(F.LEVELS, ID.LEVEL_ROW_2), HOVER_HOLD),
            ("hover_btn_sign_in", hover_at(F.SIGN_IN, ID.BTN_SIGN_IN), HOVER_HOLD),
            ("hover_slider", hover_at(F.SETTINGS, ID.SLIDER), HOVER_HOLD),
            ("press_btn_sign_in", hover_at(F.SIGN_IN, ID.BTN_SIGN_IN, press=True), HOVER_HOLD),
            ("focus_field_user", focus_at(F.SIGN_IN, ID.FIELD_USER), HOVER_HOLD),
            ("tilt_face", tilt_state(False, False, False), HOVER_HOLD),
            ("tilt_left", tilt_state(True, False, False), HOVER_HOLD),
            ("tilt_right", tilt_state(True, True, False), HOVER_HOLD),
            ("tilt_done", tilt_state(True, True, True), HOVER_HOLD),
            # fx_kill/crit/miss dùng cơ chế chờ KHÔNG GIỚI HẠN riêng
            # (``_fx_shot_pending`` trong ``_tick_sakuga_shots``) — số ở đây
            # bị bỏ qua cho ba bước này.
            ("fx_kill", fx_shot(F.PLAY, "hit_kill", (640.0, 360.0)), 0),
            ("fx_crit", fx_shot(F.PLAY, "hit_crit", (640.0, 360.0)), 0),
            ("fx_miss", fx_shot(F.PLAY, "hit_block", (640.0, 360.0)), 0),
            ("fx_loading_50", loading_state(50.0), AUTO_FX_HOLD),
            ("fx_loading_done", loading_state(100.0), AUTO_FX_HOLD),
            # TASK_game_feel_1118 P3: redo on LEVELS — "btn_Next: choose a
            # photo" is an obvious advancing CTA; SETTINGS' primary button
            # is the much less representative "btn_recalibrate".
            ("fx_button_primary", primary_click(F.LEVELS), AUTO_FX_HOLD),
            ("fx_toast", toast_shot(), AUTO_FX_HOLD),
            ("fx_win", win_shot(), AUTO_FX_HOLD),
        ]

    def _tick_sakuga_shots(self) -> None:
        """``HCPD_UI_SHOTS_1117=1``: chạy ``_sakuga_shot_plan`` rồi thoát —
        20 ảnh vào ``results/ui_sakuga_1117/`` (TASK_ui_sakuga_1117 G5).

        ``_sakuga_shot_started`` (KHÔNG phải ``hold == 0``) là cờ "đã gọi
        ``setup()`` cho bước này chưa" — dùng ``hold`` làm cờ đó từng là
        lỗi thật: trong lúc CHỜ ``_fx_shot_pending`` (có thể nhiều khung),
        ``hold`` đứng ở 0, nên ``setup()`` bị gọi lại MỖI KHUNG, ghi đè
        ``_fx_shot_pending`` vừa chốt ``None`` trở lại y lúc đầu — kẹt vô
        hạn ở bước ``fx_kill`` (texture đã lên GPU từ lâu, vẫn không bao
        giờ chụp được vì cờ chốt xong bị xoá lại đúng khung sau đó).
        """
        from OpenGL import GL

        app = self._app
        if app.window.is_iconified():
            return
        plan = self._sakuga_shot_plan()
        index = getattr(self, "_sakuga_shot_index", 0)
        if index >= len(plan):
            return
        stem, setup, hold_frames = plan[index]
        if not getattr(self, "_sakuga_shot_started", False):
            setup()
            self._sakuga_shot_started = True
            self._sakuga_shot_hold = 0
            self._sakuga_shot_post = None
        pending = getattr(self, "_fx_shot_pending", None)
        if pending is not None:
            name, point, kwargs = pending
            if app.album.spawn_fx(name, point[0], point[1], **kwargs):
                self._fx_shot_pending = None
                self._sakuga_shot_post = 0
            return      # moi spawn xong (chua du khung de chup) HOAC van
                        # cho decode — ca hai deu chua san sang chup ngay.
        post = getattr(self, "_sakuga_shot_post", None)
        if post is not None:
            post += 1
            self._sakuga_shot_post = post
            ready = post >= self._FX_PENDING_POST_FRAMES
        else:
            hold = getattr(self, "_sakuga_shot_hold", 0) + 1
            self._sakuga_shot_hold = hold
            ready = hold >= hold_frames
        if not ready:
            return
        GL.glFinish()
        self._save_sakuga_shot(stem)
        self._sakuga_shot_started = False
        self._sakuga_shot_hold = 0
        self._sakuga_shot_post = None
        self._fx_shot_pending = None
        self._sakuga_shot_index = index + 1
        if self._sakuga_shot_index >= len(plan):
            logger.info("Sakuga UI shots: xong %d anh", len(plan))
            app.window.request_close()

    def _save_sakuga_shot(self, stem: str) -> None:
        from src.ui_album.shots import save_framebuffer

        app = self._app
        app.window.restore_and_focus()
        app.window.pump_events()
        width, height = app.window.framebuffer_size()
        if width < 2 or height < 2:
            logger.warning("Sakuga shot bo qua: framebuffer %dx%d", width, height)
            return
        path = app.root / "results" / "ui_sakuga_1117" / f"{stem}.png"
        save_framebuffer(path, width, height)
        logger.info("Sakuga UI shot: %s (%dx%d)", path, width, height)

    # ---- TASK_game_feel_1118 Phase 0/7: anti-aliasing before/after crops ----

    def _aa_shot_plan(self) -> list:
        """5 bước — màn/khoảnh khắc y hệt cho cả ảnh "trước" (Phase 0) và
        "sau" (Phase 7); việc cắt/zoom 3x làm ở bước riêng, ngoài app."""
        app = self._app
        F = app.album.flow
        HOLD = 30

        def go_levels():
            app.album.state.go(F.LEVELS)
            app.album.state.level = 2
            app.album.mouse = (-1000.0, -1000.0)

        def go_play_static():
            app.album.state.go(F.PLAY)
            app.album.mouse = (-1000.0, -1000.0)

        def spawn_burst():
            app.album.state.go(F.PLAY)
            app.album.mouse = (640.0, 360.0)
            self._aa_burst_pending = True

        return [
            ("aa_level_row", go_levels, HOLD),
            ("aa_tab", go_levels, HOLD),
            ("aa_burst", spawn_burst, 6),        # ~0.1s @ ~60fps cua so nho
            ("aa_layer_edge", go_play_static, HOLD),
            ("aa_bug", go_play_static, HOLD),
        ]

    def _tick_aa_baseline_shots(self) -> None:
        """``HCPD_AA_BASELINE_1118=1`` (dùng cùng
        ``HCPD_ALBUM_AUTOMATCH=1``): 5 ảnh thô (chưa cắt) vào thư mục
        ``HCPD_AA_OUT`` (mặc định ``results/game_feel_1118/baseline/aa_raw``)
        — màn PLAY cần ván thật (bọ + lớp ảnh) nên đợi ``_album.game`` có
        trước khi chụp hai bước cuối."""
        from OpenGL import GL

        app = self._app
        if app.window.is_iconified():
            return
        plan = self._aa_shot_plan()
        index = getattr(self, "_aa_shot_index", 0)
        if index >= len(plan):
            return
        stem, setup, hold_frames = plan[index]
        if not getattr(self, "_aa_shot_started", False):
            setup()
            self._aa_shot_started = True
            self._aa_shot_hold = 0
        if stem in ("aa_layer_edge", "aa_bug", "aa_burst"):
            F = app.album.flow
            if (app.album.state.screen != F.PLAY or app.album.game is None
                    or app.renderer.waiting_for_depth):
                return          # cho ván thật mở xong (lop + bo)
        if getattr(self, "_aa_burst_pending", False):
            if app.album.spawn_fx("hit_kill", 640.0, 360.0):
                self._aa_burst_pending = False
            else:
                return
        hold = getattr(self, "_aa_shot_hold", 0) + 1
        self._aa_shot_hold = hold
        if hold < hold_frames:
            return
        GL.glFinish()
        self._save_aa_shot(stem)
        self._aa_shot_started = False
        self._aa_shot_hold = 0
        self._aa_burst_pending = False
        self._aa_shot_index = index + 1
        if self._aa_shot_index >= len(plan):
            logger.info("AA baseline: xong %d anh", len(plan))
            app.window.request_close()

    def _save_aa_shot(self, stem: str) -> None:
        from src.ui_album.shots import save_framebuffer

        app = self._app
        app.window.restore_and_focus()
        app.window.pump_events()
        width, height = app.window.framebuffer_size()
        if width < 2 or height < 2:
            logger.warning("AA shot bo qua: framebuffer %dx%d", width, height)
            return
        out = self.env("HCPD_AA_OUT") or "results/game_feel_1118/baseline/aa_raw"
        path = app.root / out / f"raw_{stem}.png"
        save_framebuffer(path, width, height)
        logger.info("AA baseline shot: %s (%dx%d)", path, width, height)
