"""Main HCPD application orchestrator."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
import logging
import math
import os
import secrets
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.content.formats import detect_content_type, is_supported_media
from src.content.hidden_props import layer_planes
from src.content.persistence import media_path_for_config, save_content_settings
from src.game.levels import LEGACY_PREFER
from src.game.levels import rules as level_rules
from src.content.prop_game import (
    PROP_COUNT,
    bug_radius_mm,
    GameState,
    is_exposed,
    append_csv,
    csv_row,
    prop_items,
    start_game,
    status_line,
    update as game_update,
)
from src.content.prop_probe import probe_at, probe_props
from src.content.prop_rest_eye import REST_MEASURE_S, rest_eye, spread_mm, z_rest_mm
from src.content import media_gate, shared_album
from src.content.shared_album_worker import SharedAlbumWorker, PENDING as SHARE_PENDING
from src.config.schema import AppConfig, ContentSettings, UiSettings, _merge_dataclass
from src.gui.file_dialog import pick_media_file
from src.engine.adaptive_quality import AdaptiveQualityController
from src.engine.camera_worker import CameraWorker
from src.engine.pose_bus import PoseBus
from src.engine.profiler import Profiler
from src.common import stage_profile
from src.engine.tracking_worker import TrackingWorker
from src.filters import PoseFilterPipeline
from src.renderer import DebugMetrics, GlfwWindow, SceneRenderer, VirtualCamera
from src.renderer.webcam_preview import WebcamPreviewSettings
from src.scene import build_room_scene
from src.scene.window_box import build_scene_for
from src.ui_album.options import AlbumOptions

logger = logging.getLogger(__name__)

#: Cửa sổ nghiệm thu sau khi mở ván: ngồi yên ngần này giây, phải 0 khung lộ vật.
WATCH_S = 5.0

#: Chờ tách lớp (PB-051): nhịp toast (bằng thời gian sống 2 s của toast), và
#: trần để một ảnh không bao giờ có lớp (vd. ``_apply_layers`` bỏ qua vì lệch
#: cỡ) không hiện toast "đang tách lớp" mãi.
LAYER_WAIT_TOAST_S = 2.0
LAYER_WAIT_MAX_S = 120.0

#: TASK_perf_textures_1119 Fix 8: every bench/evidence/shot env-var hook
#: this codebase has — checked by ``HcpdApplication._is_scripted_run``
#: so NONE of them can ever write the user's real config files. Keep this
#: in sync when a new hook is added.
HOOK_ENV_VARS = (
    "HCPD_FX_BENCH", "HCPD_SCOPE_BENCH", "HCPD_ALBUM_AUTOMATCH",
    "HCPD_EVIDENCE_1118", "HCPD_ACCOUNTS_SHOTS", "HCPD_NAV_BENCH",
    "HCPD_UI_SHOTS_1117", "HCPD_AA_BASELINE_1118", "HCPD_V3_LIVE",
    "HCPD_PERF_PROFILE", "HCPD_PVP_AUTO", "HCPD_PVP_SHOTS",
    "HCPD_PROP_PROBE", "HCPD_VIDEO_STATS", "HCPD_HUD_FIX_1119_SHOTS",
    "HCPD_ALBUM_AUTOWIN", "HCPD_BUG_SEED", "HCPD_ECONOMY_1124",
    "HCPD_HARDENING_1124_SHOTS", "HCPD_BAD_PHOTO_1124",
)

#: Bug-placement seed an automatch run uses unless ``HCPD_BUG_SEED`` says
#: otherwise, so scripted runs stay repeatable. Real play draws a fresh seed.
AUTOMATCH_BUG_SEED = 1123


def is_scripted_run(*, shots: bool, ui_shots: bool) -> bool:
    """TASK_perf_textures_1119 Fix 8: standalone so it's checkable BEFORE
    ``HcpdApplication`` exists (account storage redirect needs to happen
    before ``V3UI.__init__`` ever reads ``accounts.json``/``session.json`` —
    see the account-store redirect in ``run()``), not just from
    ``HcpdApplication._is_scripted_run``.

    TASK_account_persist_1121: neither ``--demo`` nor ``--album`` makes a
    run "scripted" — only the shot/automatch sequencers and the
    bench/evidence hooks do. Demo and album are real, interactive modes
    whose accounts/session/prefs must persist like any other real run."""
    if shots or ui_shots:
        return True
    return any(os.environ.get(var) for var in HOOK_ENV_VARS)


class _NullDevTools:
    """TASK_arch_refactor_1120 R3: ``HcpdApplication._devtools`` when no
    ``HCPD_*`` flag is set — every per-frame call site in this file is
    unconditional, so this stub exists purely to avoid an ``is not
    None`` check at each of them. Defined here (not in ``src.devtools``)
    so a normal run never imports that package at all."""

    def env(self, name: str, default: str = "") -> str:
        return default

    def tick_pre_match(self, now: float) -> None:
        pass

    def tick_pvp(self, now: float) -> None:
        pass

    def tick_post_render(self, now: float) -> None:
        pass


class HcpdApplication:
    """
    Head-coupled perspective display application.

    Three threads:
    1. CameraWorker — async webcam (internal only, never displayed)
    2. TrackingWorker — pose estimation + filtering
    3. Main thread — OpenGL rendering

    The only visible window is the virtual 3D room.
    """

    #: Seed of the match being placed; ``_start_prop_game`` sets it each match.
    _bug_seed = 0
    _bug_mover = None
    #: Seed drawn on the loading screen, used by the next match (or None).
    _pending_bug_seed = None

    #: Mặc định ở mức class để bản dựng bằng ``__new__`` (test) vẫn là sandbox.
    _demo = False
    #: TASK_album_ui_1106: cùng lý do — bản dựng bằng ``__new__`` không có
    #: giao diện album, và không được vì thế mà hỏng.
    _album_options = AlbumOptions()
    _album = None
    #: TASK_pvp_lan_1116 P2: cùng lý do — bản dựng bằng ``__new__`` vẫn phải
    #: mở webcam thật như hôm nay, trừ khi ``__init__`` nói khác.
    _no_webcam = False
    #: TASK_arch_refactor_1120 R3: cùng lý do — bản dựng bằng ``__new__``
    #: vẫn gọi được mọi ``self._devtools.xxx(...)`` mà không phải lo
    #: ``None``.
    _devtools = _NullDevTools()

    def __init__(self, config: AppConfig, project_root: Path, *,
                 demo: bool = False, album: AlbumOptions | None = None,
                 no_webcam: bool = False) -> None:
        self._config = config
        self._root = project_root
        # TASK_album_ui_1106: giao diện album. ``None`` hoặc ``enabled=False``
        # = không đổi một hành vi nào so với hôm nay.
        self._album_options = album or AlbumOptions()
        # TASK_arch_refactor_1120 R3: the ONE entry point — decides whether
        # devtools (bench/evidence/shots/automatch hooks) is needed at all,
        # BEFORE anything else below reads an HCPD_ flag. Importing
        # ``src.devtools`` is the only thing gated on this: with no flag
        # (and no ``--album-shots``/``--ui-shots``, which drive the shot
        # sequencer the same way) set, it never enters ``sys.modules``,
        # and every per-frame call below goes through ``_NullDevTools``.
        if (self._album_options.shots or self._album_options.ui_shots
                or any(os.environ.get(var) for var in HOOK_ENV_VARS)):
            from src import devtools as _devtools_mod

            self._devtools = _devtools_mod.DevTools(self)
        self._album = None
        # TASK_game_feel_1118 P1: closure ``scene`` của khung vẽ GẦN NHẤT
        # (chụp sẵn pose/khung webcam lúc đó) — ``_draw_frame_only`` dùng
        # lại y hệt lúc cửa sổ cần vẽ lại giữa chừng kéo cạnh, vì nó
        # KHÔNG được tự đọc webcam/pose mới. ``_drawing`` chặn vẽ lồng
        # nhau (refresh trong lúc refresh, hoặc trong lúc main loop đang
        # tự vẽ).
        self._last_scene = None
        self._drawing = False
        # TASK_game_feel_1118 P5: full-auto (Heavy-mag gun) — tracks
        # whether the PREVIOUS frame's trigger-held check fired, so
        # releasing the button is detected exactly once (see
        # ``_tick_auto_fire``).
        self._heavy_mag_was_held = False
        # TASK_game_feel_1118 P6: Precision-gun scope — dt tracker for
        # stepping the zoom spring (``_tick_scope``).
        self._scope_last_t = None
        # PB-055: demo đọc content.demo.json, không ghi file config nào, chỉ nhận
        # G, T, F6, kéo-thả, F4, Esc (lọc ở GlfwWindow).
        self._demo = bool(demo)
        # TASK_pvp_lan_1116 P2: khong mo CameraWorker.start() — TrackingWorker
        # da san co duong lui "khong co khung hinh -> dau mo phong" (xem
        # ``_hold_or_simulate``/``_simulated_pose``), nen chi can khong bao gio
        # dua khung hinh that vao la du, khong phai dung lai co che mo phong
        # rieng.
        self._no_webcam = bool(no_webcam)
        self._pose_bus = PoseBus()
        self._filters = PoseFilterPipeline(config.filter)
        opt = config.optimization
        self._profiler = Profiler(target_latency_ms=opt.target_latency_ms)
        self._profiler.budget._alpha = opt.latency_ema_alpha
        self._adaptive = AdaptiveQualityController(config.adaptive_quality)

        # Load screen calibration FIRST so camera_offset_mm reaches TrackingWorker.
        self._screen_cal = self._load_screen_calibration(project_root)
        self._ui_settings = self._load_ui_settings(project_root)
        if self._album_options.is_v3:
            # Giao diện v3 không có F4, nên ô webcam PiP không bao giờ bật.
            # Tắt từ đầu để nó không vẽ và không giữ texture (xem V3_KEYS).
            from dataclasses import replace as _replace

            self._ui_settings = _replace(self._ui_settings, enabled=False)
        self._content = self._load_content(project_root)
        self._log_calibration()

        camera_cfg = config.camera
        from src.camera.capture import CaptureConfig

        self._camera = CameraWorker(
            CaptureConfig(
                device_id=camera_cfg.device_id,
                width=camera_cfg.width,
                height=camera_cfg.height,
                fps=camera_cfg.fps,
                buffer_size=camera_cfg.buffer_size,
                fourcc=camera_cfg.fourcc,
            ),
            queue_size=camera_cfg.queue_size,
        )
        self._tracking = TrackingWorker(
            self._pose_bus,
            self._filters,
            self._camera,
            config,
            webcam_offset_mm=self._screen_cal["camera_offset_mm"],
        )

        # D1: chế độ hiển thị và khung cửa sổ lần trước nằm ở user_prefs.json.
        from src.config import prefs as _prefs

        self._prefs = _prefs.load(project_root)
        self._window = GlfwWindow(config.renderer, demo=self._demo,
                                  album=self._album_options.enabled,
                                  ui_v3=self._album_options.is_v3,
                                  prefs=self._prefs)
        self._virtual_camera = VirtualCamera(
            screen_width_m=self._screen_cal["screen_width"],
            screen_height_m=self._screen_cal["screen_height"],
            screen_center_m=self._screen_cal["screen_center"],
            cache_projection=opt.cache_projection,
        )
        self._shader_dir = project_root / config.renderer.shader_path
        self._renderer: SceneRenderer | None = None
        self._render_extrapolation = opt.render_extrapolation
        # Trò tìm vật ẩn (PB-015 C2). ``_game_layers_revision`` ghi bộ lớp mà
        # ván hiện tại được đặt trên, để biết khi nào hình học đã đổi.
        self._game = None
        self._game_layers_revision = 0
        self._game_planes = ()
        # Mắt nghỉ ĐO ĐƯỢC (bổ sung 15/10), không phải hằng số trong config.
        self._rest_samples = None
        self._rest_started_s = 0.0
        self._rest_eye = None
        self._rest_spread = (0.0, 0.0)
        self._tracking_off = False
        self._watch_until_s = None
        self._watch_frames = self._watch_seen = 0
        self._layer_wait_since = None
        self._layer_wait_toast_s = 0.0
        self._bugs = None
        #: U3: lượt dựng cảnh chụp ảnh luận văn. Nhịp mô phỏng và tracker
        #: **không** được ghi đè trạng thái mà preset vừa đặt.
        self._v3_live = self._devtools.env("HCPD_V3_LIVE") == "1"
        #: TASK_shared_album_1111: upload/list/download chạy ở luồng riêng —
        #: ``urllib.request`` có thể chặn tới 15 s (shared_album.TIMEOUT_S).
        self._share_worker = SharedAlbumWorker()
        #: "upload" hoặc "random" — ``_tick_shared_album`` đọc kết quả khác
        #: nhau (URL/lỗi so với Path/lỗi) nên phải nhớ việc nào đang chạy.
        self._share_job_kind: str | None = None
        #: TASK_pvp_lan_1116: kết nối PvP (``src/net/client.py``), và trạng
        #: thái phiên hiện tại. ``None`` tới khi người chơi bấm Find match /
        #: Invite / Join lần đầu — PvP không mở socket nào trước đó.
        self._pvp = None
        self._pvp_pending: dict | None = None
        self._pvp_my_name = ""
        self._pvp_is_host = False
        self._pvp_photo_sent = False
        #: Bật khi MÌNH đã bắt đầu tải đúng ảnh của trận này (host: ngay khi
        #: ``matched`` tới; khách: khi ``photo`` tới) — chặn
        #: ``_pvp_tick_lobby`` đọc nhầm hình học lớp CŨ (ảnh trước đó, nếu
        #: còn &gt;= 2 lớp từ phiên solo trước) và gửi ``ready`` quá sớm.
        self._pvp_photo_ready = False
        self._pvp_bugs_sent = False
        self._pvp_bugs_attempted = False
        self._pvp_ready_sent = False
        #: Hunt & Hide: mốc gửi ``eye`` gần nhất của hunter (chặn 10 Hz).
        self._pvp_last_eye_sent_s = 0.0
        #: Bật khi server gửi ``start`` với bọ thật (khác pha "setup" của
        #: Hunt & Hide) — ``_album_shoot`` chỉ gửi ``hit`` khi cờ này bật.
        self._pvp_active = False

    def _load_screen_calibration(self, root: Path) -> dict:
        calibration_path = root / "assets" / "config" / "screen.json"
        sc = self._config.screen
        defaults: dict = {
            "screen_width": sc.width_mm * 0.001,
            "screen_height": sc.height_mm * 0.001,
            "screen_center": np.array(
                [sc.center_x_m, sc.center_y_m, sc.center_z_m], dtype=np.float64
            ),
            "eye_offset": np.zeros(3, dtype=np.float64),
            # camera_offset: webcam position in world space (meters, HCPD axes).
            # X=0 (centered), Y=screen_height/2 + bezel (~6mm), Z≈0 (flush).
            # Stored in mm for HeadPoseEstimator.
            "camera_offset_mm": np.array(
                [0.0, sc.height_mm * 0.5 + 6.0, 0.0], dtype=np.float64
            ),
        }
        if not calibration_path.exists():
            logger.warning(
                "screen.json not found — using config defaults. "
                "TODO: measure your screen and create assets/config/screen.json"
            )
            return defaults

        payload = json.loads(calibration_path.read_text(encoding="utf-8"))
        # camera_offset in screen.json is in meters; convert to mm for the estimator.
        raw_offset_m = np.asarray(
            payload.get("camera_offset", [0.0, 0.0, 0.0]), dtype=np.float64
        )
        return {
            "screen_width": float(payload.get("screen_width", defaults["screen_width"])),
            "screen_height": float(
                payload.get("screen_height", defaults["screen_height"])
            ),
            "screen_center": np.asarray(
                payload.get("screen_center", defaults["screen_center"]),
                dtype=np.float64,
            ),
            "eye_offset": np.asarray(
                payload.get("eye_offset", [0.0, 0.0, 0.0]), dtype=np.float64
            ),
            "camera_offset_mm": raw_offset_m * 1000.0,
        }

    def _log_calibration(self) -> None:
        cal = self._screen_cal
        w_mm = cal["screen_width"] * 1000.0
        h_mm = cal["screen_height"] * 1000.0
        off = cal["camera_offset_mm"]
        logger.info(
            "Screen calibration loaded: %.0f × %.0f mm  "
            "webcam_offset=[%.1f, %.1f, %.1f] mm",
            w_mm, h_mm, off[0], off[1], off[2],
        )
        logger.info(
            "TODO: verify screen dimensions with a ruler against your physical "
            "display area (not bezel). Edit assets/config/screen.json to update."
        )

    def _load_ui_settings(self, root: Path) -> WebcamPreviewSettings:
        """Load PiP preview settings from config and optional ui.json override."""
        ui = self._config.ui
        ui_path = root / "assets" / "config" / "ui.json"
        if ui_path.exists():
            payload = json.loads(ui_path.read_text(encoding="utf-8"))
            ui = _merge_dataclass(UiSettings, {**asdict(self._config.ui), **payload})

        return WebcamPreviewSettings(
            enabled=ui.webcam_preview_enabled,
            corner=ui.webcam_preview_corner,
            scale=ui.webcam_preview_scale,
            margin_px=ui.webcam_preview_margin_px,
            mirror=ui.webcam_preview_mirror,
            draw_landmarks=ui.webcam_preview_landmarks,
        )

    def _load_content(self, root: Path) -> ContentSettings:
        """
        Load video-window content from assets/config/content.json.

        Parsing is driven by the ContentSettings dataclass itself
        (ContentSettings.from_dict), so a new knob needs no change here.
        Demo (PB-055) đọc ``content.demo.json`` — bản đã đóng băng và commit.

        L2 (TASK_game_1110): ``--album`` đọc ``content.game.json`` — hồ sơ
        RIÊNG cho ván chơi, để ``content.demo.json`` (bản trình chiếu ảnh)
        đứng yên không đổi một khoá nào. Album thắng cả cờ demo: ván chơi
        luôn cần hồ sơ game dù demo có bật hay không (TASK_account_persist_1121:
        ``run_game.bat`` nay chỉ gọi ``--album``, không còn ``--demo``).
        """
        if self._album_options.enabled:
            name = "content.game.json"
        elif self._demo:
            name = "content.demo.json"
        else:
            name = "content.json"
        path = root / "assets" / "config" / name
        logger.info("Content config: %s%s", path.name,
                   " (ALBUM)" if self._album_options.enabled else
                   " (DEMO)" if self._demo else "")
        if not path.exists():
            logger.info("content.json not found — using content defaults")
            return ContentSettings()
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.error(
                "Cannot read %s (%s) — using content defaults", path, exc
            )
            return ContentSettings()
        if not isinstance(payload, dict):
            logger.error("%s root must be a JSON object — using defaults", path)
            return ContentSettings()
        return ContentSettings.from_dict(payload)

    def _build_scene(self):
        if self._content.enabled and self._content.mode == "video":
            logger.info(
                "Content mode: video window — frame Z=%.2f, plane Z=%.2f, mid Z=%.2f, backdrop Z=%.2f",
                self._content.frame_z_m,
                self._content.plane_position_m[2],
                self._content.mid_layer_z_m,
                self._content.backdrop_z_m,
            )
            # PB-074: in đủ bốn khoá hình ảnh mới cùng độ sâu lớp xa, để một
            # lượt chạy tự nói nó đang ở phương án nào — không phải mở config ra đoán.
            logger.info(
                "PB-074 keys: edge_style=%s · layer_shadow=%s · fill_method=%s "
                "· surround=%s · layer_z_far_m=%.2f",
                self._content.edge_style, self._content.layer_shadow,
                self._content.fill_method, self._content.surround,
                self._content.layer_z_far_m)
            return build_scene_for(
                self._content, self._screen_cal["screen_width"],
                self._screen_cal["screen_height"])
        return build_room_scene()

    def _load_error(self, text: str) -> None:
        """A photo that could not be opened: the toast of the 3D pass, plus the
        album's own message line (the toast alone is not visible in v3)."""
        self._renderer.show_toast(text, is_error=True)
        album = self._album
        if album is not None and self._album_options.is_v3:
            album.flow.say(album.state, text, 5.0)

    def _try_load_media(self, path: Path, now: float, *, clean=None) -> None:
        """
        Load image/video via picker or drag-drop; persist on success.

        ``now`` là đầu khung bấm F6 / thả file — trước khi hộp chọn file chặn
        vòng lặp — để số giây của ván bị bỏ dở không gồm thời gian chọn file.
        """
        if not path.exists():
            self._load_error(f"File not found: {path.name}")
            return
        if not is_supported_media(path):
            self._load_error("Unsupported file type")
            return

        # An image goes through ``media_gate`` inside ``load_content_file``
        # (``clean``: the caller already did it, e.g. for a PvP photo).
        ok, msg = self._renderer.load_content_file(
            clean if clean is not None else path, label=path.name)
        if not ok:
            self._load_error(msg)
            return

        kind = detect_content_type(path) or "video"
        self._content.content_path = media_path_for_config(path, self._root)
        self._content.content_type = kind
        self._content.mode = "video"
        # PB-055: demo không ghi file config. TASK_account_persist_1121:
        # album không ghi content.json (research config) nữa — dù demo có
        # bật hay không; ván chơi không cần nhớ ảnh vừa mở qua file này.
        if not self._demo and not self._album_options.enabled:
            config_path = self._root / "assets" / "config" / "content.json"
            save_content_settings(
                config_path, self._content, path, kind, self._root
            )
        self._abandon_game(now, "new_image")
        # FR-DEPTH-03: gộp thông báo "video chỉ lấy depth khung đầu" vào cùng
        # một toast, vì toast sau sẽ đè toast trước.
        notice = self._renderer.take_content_notice()
        self._renderer.show_toast(msg if notice is None else f"{msg} — {notice}")

    def _draw_album_bugs(self, view_flat, proj_flat) -> None:
        """Vẽ bọ của ván đang chơi (F3). Không có ván thì không vẽ gì."""
        if self._game is None or not self._game.props:
            return
        if self._bugs is None:
            return
        self._bugs.draw(view_flat, proj_flat, self._game.props,
                        self._game.found, self._album.state.level,
                        self._album.bug_theme)

    def _tick_album(self, input_state, now: float) -> None:
        """
        Một khung của giao diện album: chuột, chữ gõ, nhịp mô phỏng, hành động.

        Giao diện chỉ *yêu cầu*; mọi việc đụng tới app (mở file, tách lớp, đo
        mắt nghỉ, mở ván) làm ở đây — xem ``src/ui_album/routes.py``.
        """
        album = self._album
        sakuga_shots = self._devtools.env("HCPD_UI_SHOTS_1117") == "1"
        scope_bench = self._devtools.env("HCPD_SCOPE_BENCH") == "1"
        evidence_1118 = self._devtools.env("HCPD_EVIDENCE_1118") == "1"
        # TASK_hud_fix_1119 Fix 4 evidence: same reasoning as
        # ``evidence_1118`` right below — ``_build_hud_fix_1119_steps``
        # also drives its own fixed ``album.mouse`` per step.
        hud_fix_1119_shots = self._devtools.env("HCPD_HUD_FIX_1119_SHOTS") == "1"
        hardening_1124_shots = self._devtools.env("HCPD_HARDENING_1124_SHOTS") == "1"
        if (not sakuga_shots and not scope_bench and not evidence_1118
                and not hud_fix_1119_shots and not hardening_1124_shots):
            # TASK_ui_sakuga_1117: lượt chụp 20 ảnh tự đặt ``album.mouse``
            # cho từng bước (hover_*/press_*) — con trỏ OS thật ghi đè lên
            # đúng mỗi khung thì KHÔNG BAO GIỜ thấy hover, bất kể bước nào
            # (``album.mouse`` luôn là vị trí chuột thật, không phải vị trí
            # hotspot bước đó muốn trỏ tới). TASK_game_feel_1118 P7:
            # ``HCPD_SCOPE_BENCH=1`` drives its OWN circular crosshair —
            # see ``_tick_scope_bench``. TASK_game_feel_fix_1118:
            # ``HCPD_EVIDENCE_1118=1`` drives its own FIXED crosshair per
            # step — see ``_build_evidence_steps``, which calls
            # ``_tick_scope`` itself with its own chosen raw mouse.
            raw_mouse = album.design_mouse(*self._window.cursor_pos())
            album.mouse = self._tick_scope(now, raw_mouse)
        fx_bench_heavy = (self._devtools.env("HCPD_FX_BENCH") == "1"
                         and self._devtools.env("HCPD_BENCH_WEAPON") == "Heavy-mag gun")
        if self._album_options.is_v3 and not sakuga_shots and not fx_bench_heavy:
            # TASK_ui_sakuga_1117 Phase 5: press feedback needs HOLD state,
            # not just the edge-triggered clicks ``pop_clicks()`` gives.
            # (``sakuga_shots`` mode sets its OWN ``album.mouse_down`` per
            # step — same reasoning as ``album.mouse`` above.)
            #
            # TASK_game_feel_1118 P7: ``fx_bench_heavy`` also skips this —
            # ``_tick_fx_bench`` (called LATER this same frame) holds
            # ``album.mouse_down`` down itself; running this line
            # unconditionally every frame would reset it to the real
            # (unpressed) mouse state before ``_tick_auto_fire`` (called
            # EARLIER than ``_tick_fx_bench``) ever saw it held.
            album.mouse_down = self._window.is_mouse_down()
        for char in self._window.pop_typed():
            album.key(char)
        if input_state.backspace:
            album.backspace()
        if input_state.focus_next:
            album.focus_next()
        if input_state.submit:
            self._album_action(album.submit(), now)
        for _click in self._window.pop_clicks():
            if self._album_shoot(now):
                continue                     # đang trong trận: bấm = bắn
            self._album_action(album.click(), now)
        self._tick_auto_fire(now)
        if input_state.album_centre and not self._v3_field_focused():
            self._album_action("rest_eye", now)
        if input_state.screenshot:
            self._save_album_shot()
        if not self._v3_live and self._devtools.env("HCPD_UI_SHOTS_1117") != "1":
            # TASK_ui_sakuga_1117: lượt chụp 20 ảnh tự đặt face_seen/tilt_*
            # cho từng bước (tilt_face/left/right/done) — tracker thật ghi
            # đè lên đúng y mỗi khung thì bốn ảnh đó không bao giờ khác nhau.
            album.state.face_seen = bool(self._tracking.face_detected)
            album.state.tracker_simulated = not bool(
                self._tracking.tracker_available)
        if self._album_options.is_v3:
            self._tick_v3_input(input_state, now)
        album.tick()
        self._devtools.tick_pre_match(now)
        self._tick_album_match(now)
        self._tick_pvp(now)
        self._devtools.tick_pvp(now)
        self._tick_shared_album()
        self._log_album_fps(now)

    def _start_share_upload(self, path) -> None:
        """P2.4: nút/phím U đã xác nhận — tải ảnh lên, không chặn luồng GL."""
        if self._share_worker.submit(shared_album.upload, path):
            self._share_job_kind = "upload"
        else:
            logger.warning("Shared album: dang ban mot viec khac, bo qua")

    def _start_random_photo(self) -> None:
        """P3: nút/phím R — xin, chọn ngẫu nhiên và tải một ảnh, ngoài luồng GL."""
        dest_dir = self._root / "assets" / "cache" / "shared"
        if self._share_worker.submit(shared_album.pick_random_and_download, dest_dir):
            self._share_job_kind = "random"
        else:
            logger.warning("Shared album: dang ban mot viec khac, bo qua")

    def _tick_shared_album(self) -> None:
        """Đọc kết quả upload/random (P2.4/P3.2: toast, hoặc chọn ảnh mới)."""
        result = self._share_worker.pop_result()
        if result is SHARE_PENDING:
            return
        kind, self._share_job_kind = self._share_job_kind, None
        state = self._album.state
        if kind == "upload":
            ok = isinstance(result, str) and result.startswith("https://")
            self._album.flow.say(state, "Shared" if ok else str(result))
        elif kind == "random":
            if isinstance(result, Path):
                state.photo = result
                if result not in state.recent:
                    state.recent.insert(0, result)
                    del state.recent[4:]
                self._album_action("start_pipeline", time.perf_counter())
            else:
                self._album.flow.say(state, str(result))  # anh hien tai giu nguyen

    def _layer_progress(self) -> tuple[int, int]:
        """
        ``(lớp đang xét, tổng số lớp)`` cho ô ``LAYER a / N`` của HUD.

        ``N`` là số lớp THẬT mà ảnh này tách ra. ``a`` là lớp **nông nhất còn
        giấu một con bọ chưa tìm**, đánh số 1 = lớp gần người chơi nhất — tức
        "chỗ nông nhất vẫn còn thứ để tìm". Hết bọ ở lớp nông thì con số tự
        tụt sâu dần, nên dãy pip đầy lên theo đúng tiến độ ván chơi.
        """
        planes = self._game_planes
        total = len(planes)
        if not total:
            return (1, 0)
        game = self._game
        if game is None or not game.props:
            return (1, total)
        # ``layer_planes`` xếp xa → gần, nên lớp gần nhất là chỉ số cuối.
        # Trường là ``z_m``; ``z_world_m`` là tên bên ``_LayerRenderable`` của
        # renderer, không phải bên ``LayerPlane`` — gọi nhầm thì cả vòng vẽ
        # chết ngay khi vào trận.
        depths = [float(plane.z_m) * 1000.0 for plane in planes]
        shallowest = total
        for index, prop in enumerate(game.props):
            if game.found[index]:
                continue
            z_mm = float(prop["pos_mm"][2])
            nearest = min(range(total), key=lambda i: abs(depths[i] - z_mm))
            shallowest = min(shallowest, total - nearest)
        return (max(1, shallowest), total)

    def _glass_mm(self):
        """
        Tấm kính ĐANG vẽ: ``((rộng, cao) mm, (tâm x, y) mm)``.

        Chạy toàn màn hình thì bằng hiệu chỉnh; chạy trong cửa sổ thì là vùng
        khách của cửa sổ (D2). Không gian thiết kế 1280×720 phủ đúng tấm kính
        này, nên mọi phép đổi mm ↔ pixel thiết kế (chiếu bọ, tâm ngắm) phải
        quy theo nó.
        """
        camera = self._virtual_camera
        return ((float(camera.screen_width_m) * 1000.0,
                 float(camera.screen_height_m) * 1000.0),
                (float(camera.screen_center_m[0]) * 1000.0,
                 float(camera.screen_center_m[1]) * 1000.0))

    def _viewport_rect(self, screen: str | None = None):
        """
        Ô viewport của **màn sẽ chơi**, trong toạ độ thiết kế.

        Mặc định lấy màn chơi chứ KHÔNG lấy màn đang mở: ``_start_album_match``
        chạy lúc người chơi còn đứng ở màn thử nghiêng, mà ô viewport ở đó chỉ
        là [78,166,632,380] — nhỏ hơn hẳn ô [28,28,1224,664] của màn chơi. Lấy
        nhầm ô nhỏ thì bộ lọc vứt gần hết bọ: log 07/11 "xin 6 · đặt 1 · ngoài
        ô 7", người chơi vào trận cấp 1 chỉ có đúng một con.
        """
        from src.game.viewport_filter import VIEWPORT_RECT

        if not self._album_options.is_v3:
            return VIEWPORT_RECT
        name = screen or self._album.flow.PLAY
        record = self._album.manifest.screens.get(name, {})
        for photo in record.get("photos", ()):
            if photo.get("kind") == "viewport":
                return tuple(photo["rect"])
        return VIEWPORT_RECT

    def _build_bug_painter(self):
        """
        ``BugPainter`` dùng hình bọ của bộ asset **cũ** — bộ v3 không có.

        Thiếu ``assets/ui_album/`` (nó nằm trong .gitignore) thì chỉ mất con
        bọ vẽ ra, không được làm cả app không mở lên được.
        """
        from src.ui_album.bugs import BugPainter
        from src.ui_album.manifest import AlbumManifest

        manifest = getattr(self._album, "manifest", None)
        if self._album_options.is_v3:
            try:
                manifest = AlbumManifest(self._root)
            except (OSError, ValueError, KeyError) as err:
                logger.warning("Khong nap duoc hinh bo (%s) — man choi khong "
                               "co bo ve ra", err)
                return None
        painter = BugPainter(manifest, self._shader_dir)
        painter.initialize()
        return painter

    def _v3_field_focused(self) -> bool:
        """TASK_hkey_profile_1119: a v3 text field (sign-in/up, the
        forgot-password dialog's own four fields, the PvP join-code
        field) has keyboard focus right now.

        The char callback (``pop_typed``/``album.key``) and the single-
        letter shortcut keys (H, C, 1-3 — see ``_on_key``'s ``V3_KEYS``)
        fire from TWO INDEPENDENT GLFW callbacks for the SAME keypress:
        typing "h" into a username both inserts the letter AND used to
        toggle the HUD. ``False`` for the legacy album UI unconditionally
        — it has no such focus concept at all, and its input handling
        must not change (it is the thesis-defense fallback)."""
        if not self._album_options.is_v3:
            return False
        state = self._album.state
        return state.focus is not None or state.forgot_open

    def _tick_v3_input(self, input_state, now: float) -> None:
        """
        Phần nhập liệu CHỈ giao diện v3 có: Esc lùi, F11, ↑/↓, R, 1–3, U.

        Để riêng một hàm chứ không trộn vào ``_tick_album``: bộ asset cũ là
        đường lùi cho buổi bảo vệ, nên nó không được đổi một hành vi nào.
        """
        album = self._album
        state = album.state
        if input_state.escape:
            action = album.back()
            if action == "quit":
                self._window.request_close()
            else:
                self._album_action(action, now)
        if input_state.display_toggle:
            self._apply_display_mode(self._window.toggle_display())
        if input_state.focus_up:
            album.move_focus(-1)
        if input_state.focus_down:
            album.move_focus(1)
        if input_state.replay:
            if state.screen in (album.flow.WIN, album.flow.LOSE):
                state.go(album.flow.TILT)
            elif state.screen == album.flow.PHOTOS:
                # TASK_shared_album_1111 P3: cung phim R, nghia khac theo man —
                # y het Enter (album.submit()) da doi nghia theo man tu truoc.
                self._album_action(album.activate("btn_Random from album"), now)
        if input_state.item_slot and not self._v3_field_focused():
            state.slot_selected = int(input_state.item_slot)
        if input_state.share_to_album and state.screen == album.flow.PHOTOS:
            self._album_action(album.activate("btn_Share to album"), now)
        if (input_state.hud_hide_toggle and not self._v3_field_focused()
                and state.screen in (album.flow.PLAY, album.flow.PVP_HUNT,
                                     album.flow.PVP_HIDE)):
            album.toggle_hud_hidden()
        if album.pop_account_corrupt_toast():
            album.flow.say(state, "Your saved accounts file was unreadable — started a fresh one.")
        # TASK_small_fixes_1120: caret move/delete inside the focused
        # text field — same guard pattern as every other key here, no
        # ``_v3_field_focused()`` check needed (``move_caret``/``caret_home``/
        # ``caret_end``/``delete_char`` already no-op with nothing focused).
        if input_state.caret_left:
            album.move_caret(-1)
        if input_state.caret_right:
            album.move_caret(1)
        if input_state.caret_home:
            album.caret_home()
        if input_state.caret_end:
            album.caret_end()
        if input_state.delete_forward:
            album.delete_char()
        if input_state.paste_requested:
            # TASK_accounts_1119: dán vào ô đang chọn, kể cả ô mật khẩu —
            # đi qua ĐÚNG đường ``album.key()`` mỗi ký tự gõ tay cũng đi,
            # nên trần ký tự từng ô (``F.FIELD_MAXLEN``) và việc đánh dấu
            # "đã gõ vào" (gợi ý lỗi) áp dụng y hệt dán tay.
            for char in self._window.get_clipboard():
                if char not in ("\n", "\r", "\t"):
                    album.key(char)
        # Thanh âm lượng: bấm HOẶC kéo. Kéo phải đọc trạng thái phím chuột,
        # không đọc được qua danh sách cú bấm.
        if (state.screen == album.flow.SETTINGS
                and self._window.is_mouse_down()
                and album.hover() == "slider"):
            self._album_action(album.click(), now)
        # Màn thử nghiêng cần góc đầu thật để chấm hai dấu trái/phải.
        if state.screen == album.flow.TILT and not self._v3_live:
            eye = self._virtual_camera.eye_position_mm
            album.flow.tilt_angle(state, float(eye[0]), float(eye[2]))
        album.set_camera_frame(self._camera.peek_frame())
        state.webcam_name = f"Webcam #{state.webcam_index}"
        state.display = self._window.display_mode
        # Số lớp THẬT của ảnh đang mở — mỗi ảnh tách ra một số khác nhau
        # (adaptive K 2–4), nên không màn nào được in hằng 4 của bản vẽ.
        state.layer_count = len(self._renderer.layer_records())
        state.layer_target = int(self._content.layer_count)

    def _apply_display_mode(self, mode: str) -> None:
        """Ghi lại chế độ hiển thị và dời mặt phẳng Kooima theo cửa sổ mới."""
        from src.config import prefs as P

        if self._album is not None:
            self._album.state.display = mode
        if not self._is_scripted_run():
            P.save(self._root, mode, self._window.remembered_rect())
        self._refresh_window_plane()

    def _refresh_window_plane(self) -> None:
        """
        D2 — mặt phẳng Kooima = vùng khách của cửa sổ trên màn hình.

        Gắn vào callback dời chỗ / đổi cỡ của GLFW, nên nó chạy đúng lúc hình
        chữ nhật thật đổi, không phải mỗi khung.
        """
        from src.renderer.window_plane import window_plane

        window = self._window
        _monitor, is_primary = window.window_centre_monitor()
        if not is_primary and self._renderer is not None:
            # Hiệu chỉnh chỉ đo cho màn chính; nói thẳng chứ đừng lặng lẽ áp
            # số của màn này lên một màn khác kích thước.
            if not getattr(self, "_warned_monitor", False):
                self._warned_monitor = True
                self._renderer.show_toast(
                    "HCPD is calibrated for the primary monitor", is_error=True)
        # Luôn quy chiếu theo màn CHÍNH (D2), kể cả khi cửa sổ đã sang màn khác.
        position, size, _refresh = window.monitor_geometry(None)
        rect = window.window_rect()
        if rect[2] < 2 or rect[3] < 2:
            # Cửa sổ thu nhỏ: GLFW trả 0×0. Tấm kính rộng 0 làm hai góc pa/pb
            # trùng nhau, và Kooima chuẩn hoá một vector dài 0 → ValueError
            # giết cả khung vẽ ("Cannot normalize zero-length vector", 07/11).
            # Giữ nguyên tấm kính cũ cho tới khi cửa sổ hiện lại.
            logger.debug("D2: bo qua khung cua so %dx%d", rect[2], rect[3])
            return
        window.framebuffer_ratio()
        width_m, height_m, centre = window_plane(
            self._screen_cal["screen_width"], self._screen_cal["screen_height"],
            self._screen_cal["screen_center"], position, size,
            (rect[0], rect[1]), (rect[2], rect[3]))
        self._virtual_camera.set_screen(width_m, height_m, centre)
        # Kéo một góc cửa sổ sinh hàng trăm lượt gọi (đo được 591 lượt trong
        # một lần kéo), nên chỉ lượt ĐẦU ở mức INFO; còn lại xuống DEBUG để
        # log một phiên chơi không bị một cú kéo chuột nhấn chìm.
        level = logging.DEBUG if getattr(self, "_logged_plane", False) else logging.INFO
        self._logged_plane = True
        logger.log(level,
                   "D2 man kinh: cua so %dx%d tai (%d, %d) -> %.1f x %.1f mm, "
                   "tam (%.1f, %.1f) mm", rect[2], rect[3], rect[0], rect[1],
                   width_m * 1000.0, height_m * 1000.0,
                   centre[0] * 1000.0, centre[1] * 1000.0)

    def _log_album_fps(self, now: float) -> None:
        """
        Mỗi 5 giây in FPS **đếm khung** của giao diện album.

        Đếm khung chứ không lấy EMA của HUD F1: cả dự án dùng cùng một định
        nghĩa FPS, nếu không thì số của hai lần đo không so được với nhau.
        """
        self._album_frames = getattr(self, "_album_frames", 0) + 1
        start = getattr(self, "_album_fps_t0", None)
        if start is None:
            self._album_fps_t0 = now
            return
        if now - start < 5.0:
            return
        logger.info("Album FPS 5s: real %.1f (%d khung) — man %s",
                    self._album_frames / (now - start), self._album_frames,
                    self._album.state.screen)
        self._album_frames, self._album_fps_t0 = 0, now

    def _album_draw_guarded(self, scene) -> None:
        """
        Vẽ giao diện album; **một lỗi vẽ không được làm sập ván đang chơi**.

        Một ô bầu dục co về 0 px đã ném ``ValueError`` từ PIL và giết cả vòng
        vẽ giữa trận (06/11 23:51). Lỗi vẽ nay chỉ ghi log **một lần mỗi
        loại** rồi bỏ qua khung đó.
        """
        try:
            self._album.draw((self._window.width, self._window.height),
                             scene_draw=scene)
        except Exception as err:                       # noqa: BLE001
            seen = getattr(self, "_album_draw_errors", set())
            key = f"{type(err).__name__}: {err}"
            if key not in seen:
                seen.add(key)
                self._album_draw_errors = seen
                logger.exception("Album: loi ve mot khung (%s) — bo qua", key)

    def _draw_frame_only(self, scene) -> None:
        """
        Vẽ ĐÚNG MỘT khung toàn cảnh ở cỡ framebuffer HIỆN TẠI bằng ``scene``
        đã dựng sẵn (đóng gói pose/khung webcam từ lúc dựng, không tự đọc
        gì mới) — dùng cho khung bình thường của main loop. Không đọc
        input, không tick ván/PvP, không đọc webcam, không gọi
        ``pump_events``.

        TASK_game_feel_fix_1118 Fix 9: trước đây đường mã này CŨNG là
        đường vẽ-lại khi kéo cạnh cửa sổ (``_on_refresh`` gọi thẳng vào
        đây) — chạy lại TOÀN BỘ full draw (bake HUD bằng PIL, upload lại
        texture, v.v.) mỗi bước cỡ trong vòng modal resize của Windows là
        quá chậm, đúng như báo lại của người dùng (khung rớt, không thấy
        cửa sổ đã to tới đâu). Giờ ``_on_refresh`` không gọi hàm này nữa —
        nó chỉ blit lại khung ĐÃ vẽ (xem ``Window.draw_last_frame_letterboxed``).
        Hàm này vẫn là full draw duy nhất, và chụp lại khung vừa vẽ (TRƯỚC
        khi swap) để đường blit đó có cái mà dùng.
        """
        if self._album is not None:
            if self._should_freeze_on_screen_switch():
                if self._window.draw_last_frame_letterboxed():
                    return
                # Nothing captured yet (very first frame ever) — fall
                # through to a real draw instead of a black window.
            if not self._album.has_viewport():
                # Màn không có ô viewport: vẫn chạy đúng đường vẽ demo
                # (render đầy đủ) nhưng cắt xuống 1 pixel — xem chú thích
                # gốc ở chỗ gọi cũ trong main loop.
                self._album.draw_hidden_scene(scene)
            self._album_draw_guarded(scene)
        else:
            scene()
        self._window.capture_last_frame()
        with stage_profile.stage("swap_vsync"):
            self._window.swap_buffers()

    def _should_freeze_on_screen_switch(self) -> bool:
        """TASK_perf_textures_1119 Fix 3: "swap only when ready" — up to
        500 ms, keep the window showing the LAST drawn frame while the
        screen just switched TO is still decoding its textures in the
        background, instead of a broken partial draw (missing base/top
        layer). ``V3UI.prepare_screen`` runs regardless of the verdict —
        including on frames we end up freezing — so loading keeps
        progressing even while frozen; skipping it there would mean a
        screen that's still loading at the 500 ms mark NEVER finishes.

        v3-only: the old album UI never got this cache/loader rework."""
        if not self._album_options.is_v3:
            return False
        ready = self._album.prepare_screen()
        screen = self._album.state.screen
        now = time.perf_counter()
        if screen != getattr(self, "_screen_switch_screen", None):
            self._screen_switch_screen = screen
            self._screen_switch_t = now
        if ready:
            return False
        return (now - self._screen_switch_t) < 0.5

    def _on_refresh(self) -> None:
        """
        ``window.set_refresh_handler``: GLFW báo cửa sổ cần vẽ lại NGAY —
        đang kéo cạnh (modal loop bên trong ``glfw.poll_events()``, main
        loop không chạy tới lượt) hoặc vừa đổi cỡ.

        TASK_game_feel_fix_1118 Fix 9: KHÔNG chạy lại full draw nữa (xem
        ``_draw_frame_only``'s docstring) — chỉ blit khung đã chụp sẵn,
        letterbox 16:9 vào cỡ framebuffer MỚI. Khung đầy đủ, đúng cỡ
        cuối cùng, vẽ ở lượt KẾ TIẾP của main loop, một lần, sau khi
        ``poll_events()`` trả về — không cần gọi gì thêm ở đây cho việc
        đó, main loop tự nhiên làm điều này.

        Chặn lồng nhau: một refresh nổ ra TRONG một refresh khác, hoặc
        trong lúc main loop đang tự vẽ khung của nó, thì bỏ qua — không
        có gì để vẽ thêm mà an toàn hơn vẽ chồng.
        """
        if self._drawing or self._last_scene is None:
            return
        self._drawing = True
        try:
            self._window.draw_last_frame_letterboxed()
        finally:
            self._drawing = False

    def _photo_rgb_for_camo(self):
        """
        Ảnh gốc (RGB, CPU) để lấy màu nguỵ trang — KHÔNG đọc lại GL.

        Đọc thẳng từ đĩa đúng file ảnh đang mở; lỗi đọc (file mất, định dạng
        lạ) trả ``None`` và ``assign_bug_camo`` bị bỏ qua — con bọ khi đó vẽ
        không nguỵ trang, không phải lỗi chết ván.
        """
        from src.content import media_gate

        photo = self._renderer.current_clean_photo
        if photo is None:
            return None
        try:
            return media_gate.load_rgb(photo)
        except (OSError, ValueError):
            return None

    def _aim_context(self):
        """
        ``AimContext`` của khung này: mắt thật, hình học kính, lớp, và sprite.

        F1: hộp trúng là hình bóng THẬT của sprite, nên nó cần đúng mặt nạ
        alpha mà ``BugPainter`` vẽ. Thiếu bộ asset (không có sprite) thì trả
        ``None`` — không có bọ vẽ ra thì cũng không có gì để bắn.

        TASK_hit_0200: tập bắn được PHẢI bằng tập vẽ ra. ``BugPainter.draw``
        (``app.py:341`` gọi với ``self._game.props`` nguyên vẹn) vẽ MỌI con
        chưa diệt, không lọc ``is_exposed`` — xem ``renderer.py:391-392``: pass
        bọ chạy TRƯỚC pass lớp, và ``content_plane.py:944`` tắt depth WRITE
        (không tắt depth TEST, bật một lần ở ``window.py:346``, không ai tắt
        lại) cho pass lớp, nên một lớp gần hơn (alpha ≥ ngưỡng vẽ) vẫn đứng
        TRƯỚC con bọ trong depth buffer và vẽ đè lên nó trên màn — lớp gần
        CÓ THỂ che bọ thật. Vì vậy ``bugs`` ở đây lấy MỌI con chưa diệt (không
        lọc ``is_exposed``), và việc loại bỏ chuyển hẳn cho
        ``pick_target``/``alpha_threshold`` xử lý bằng đúng ngưỡng vẽ/bỏ của
        ``video_rgba.frag`` (``content.layer_alpha_threshold``).
        """
        from src.ui_album.bugs import SIZE_FACTOR
        from src.game.match import AimContext

        mask = self._bugs.sprite_mask(self._album.state.level) \
            if self._bugs is not None else None
        if mask is None:
            return None
        # TASK_game_feel_1118 P6: the "steady eye" while scoped — equals
        # ``eye_position_mm`` whenever not scoped, so this is a no-op
        # change for every OTHER weapon/screen. Rendering reads the same
        # property (``VirtualCamera.projection_matrix``/``view_matrix``),
        # so a shot never hits something different from what's on screen.
        eye = self._virtual_camera.effective_eye_position_mm
        glass, centre = self._glass_mm()
        radius = bug_radius_mm(self._content)
        bugs = {
            # TASK_bugs_look_0130: đọc ĐÚNG size_k đã ghi lúc mở ván — cùng
            # giá trị mà BugPainter.draw dùng để vẽ, không tính lại.
            index: (tuple(prop["pos_mm"]), float(prop.get("radius_mm", radius)),
                    float(prop.get("size_k", SIZE_FACTOR)), mask)
            for index, prop in enumerate(self._game.props)
            if not self._game.found[index]
        }
        exposed = {index: is_exposed(self._game, index) for index in bugs}
        return AimContext(eye_mm=tuple(float(v) for v in eye), bugs=bugs,
                          planes=tuple(self._game_planes or ()),
                          glass_mm=glass, centre_mm=centre,
                          alpha_threshold=float(
                              self._content.layer_alpha_threshold),
                          is_exposed=exposed)

    def _album_shoot(self, now: float) -> bool:
        """
        Bấm chuột trong trận = bắn. Trả ``True`` nếu đã xử lý phát bắn.

        F1: ``game_bridge.fire`` nhận cả bối cảnh ngắm, nên nó tính được phát
        bắn găm vào hình bóng nào — kể cả khi tâm ngắm nằm trên cái chân, và
        kể cả khi một lớp gần hơn đang che chỗ đó.
        """
        F = self._album.flow
        from src.game.match import fire

        match = self._album.game
        if match is None or self._album.state.screen not in (F.PLAY, F.PVP_HUNT):
            return False
        if self._game is None or not self._game.props:
            # Ván giao diện còn sống mà ván vật ẩn đã bị dọn: xảy ra khi hình
            # học lớp dựng lại giữa trận (``_sync_prop_game`` gọi
            # ``_abandon_game``). Không chặn ở đây thì cú bấm kế tiếp đọc
            # ``self._game.props`` trên ``None`` và giết cả vòng vẽ (07/11).
            return True              # vẫn nuốt cú bấm: đang trong trận
        aim = self._aim_context()
        if aim is None:
            return True
        shot = fire(match, self._album.mouse, aim,
                   pvp=(self._album.state.screen == F.PVP_HUNT))
        if shot is None:
            return True          # hết băng hoặc đang nạp: nuốt cú bấm
        hit, crit = shot
        if self._album_options.is_v3:
            # TASK_game_feel_1118 P4: kick/shake/hit marker/crit impact
            # for THIS shot — every outcome (hit, crit, miss) gets
            # it; only a hit gets a marker, only a crit gets the impact
            # frame too (``ShotFeel.on_shot`` itself decides which).
            self._album.register_shot(hit >= 0, crit)
        # TASK_weapons_1119: a silhouette hit no longer always kills —
        # ``fire()`` already ran damage per pellet (the shotgun fires
        # several) and recorded which ones actually died in
        # ``last_shot_hits``. Only a KILL removes the bug, counts, sends
        # PvP's ``hit`` or plays ``hit_kill``/``hit_crit``; a non-lethal
        # contact already got its marker above via ``register_shot``, and
        # its HP bar comes straight back from ``match.bug_last_hit``
        # (``v3_values._hud``) — nothing else to do for it here.
        for index, pellet_crit, killed in match.last_shot_hits:
            if not killed:
                continue
            self._renderer.set_prop_found(index)
            if self._bugs is not None:
                self._bugs.mark_found(
                    index, [v / 1000.0 for v in self._game.props[index]["pos_mm"]])
            if self._album_options.is_v3:
                # TASK_ui_sakuga_1117 Phase 6: LOCAL hit only — this path
                # never runs for the opponent's kills, so no extra guard
                # is needed to keep their shots silent here.
                self._album.spawn_hit_fx(pellet_crit)
            if self._pvp_active and self._pvp is not None:
                # TASK_pvp_lan_1116: xác nhận trúng do SERVER quyết — đánh
                # dấu "found" cục bộ ở trên chỉ là phản hồi tức thì, điểm/
                # ai được tính hạ bọ chỉ chốt khi ``killed`` về (xem
                # ``_pvp_handle_killed``).
                from src.net import protocol as proto

                self._pvp.send({"t": proto.HIT, "bug": index})
            eye = aim.eye_mm
            logger.info("Prop BAN TRUNG #%d%s — eye (%.0f, %.0f, %.0f) mm",
                        index, " CHI MANG" if pellet_crit else "",
                        eye[0], eye[1], eye[2])
        if hit < 0 and self._album_options.is_v3:
            self._album.spawn_miss_fx()
        return True

    def _tick_auto_fire(self, now: float) -> None:
        """
        TASK_game_feel_1118 P5: full-auto for the Heavy-mag gun only.

        The first shot of a burst already happened through the normal
        click path (``pop_clicks()`` above, same frame). This fires AGAIN
        every subsequent frame the left button is still held, as long as
        ``match.ready_to_fire(now)`` holds and the gun isn't reloading —
        checked BEFORE calling ``_album_shoot`` (which calls ``fire()``),
        so an unready frame never even logs "Bam co dan". Releasing the
        gun — button up, weapon switch, or leaving the match screen — any
        of those ends the bloom bump started below.

        TASK_game_feel_fix_1118 Fix 3: the reload re-trigger race (fire
        landing on the exact frame the reload deadline passes, before
        anything has refilled ``rounds_left`` yet this frame) used to be
        patched HERE with an explicit ``match.tick_reload(now)`` call —
        but that only covered THIS call path, not a regular click. Fixed
        at the root instead: ``AlbumMatch.take_round()`` calls
        ``tick_reload(now)`` itself now, so every firing path (this one,
        and ``pop_clicks()`` above) is covered without a duplicate call.
        """
        if not self._album_options.is_v3:
            return
        F = self._album.flow
        match = self._album.game
        firing = (match is not None
                 and self._album.state.screen in (F.PLAY, F.PVP_HUNT)
                 and match.weapon_name == "Heavy-mag gun"
                 and bool(self._album.mouse_down))
        if firing and match.ready_to_fire(now) and not match.reloading(now):
            if self._album_shoot(now):
                self._album.bump_auto_bloom()
        elif self._heavy_mag_was_held and not firing:
            self._album.release_auto_bloom()
        self._heavy_mag_was_held = firing

    def _tick_scope(self, now: float, raw_mouse: tuple) -> tuple:
        """
        TASK_game_feel_1118 P6: Precision-gun scope — right mouse, held.

        Called once per frame, BEFORE anything else reads
        ``self._album.mouse`` — returns the crosshair this frame should
        actually use: the raw mouse position unscoped, or the finer-aim
        crosshair while scoped. Also keeps ``self._virtual_camera``'s
        scope fields in sync, so rendering (``projection_matrix``/
        ``view_matrix``) and the hit test (``_aim_context``, which reads
        ``effective_eye_position_mm``) always agree.
        """
        if not self._album_options.is_v3:
            return raw_mouse
        F = self._album.flow
        match = self._album.game
        on_screen = (match is not None
                    and self._album.state.screen in (F.PLAY, F.PVP_HUNT))
        is_precision = match is not None and match.weapon_name == "Precision gun"
        # TASK_game_feel_1118 P7: HCPD_SCOPE_BENCH=1 holds the (fake)
        # trigger the whole run, exactly as a real right-mouse hold would.
        held = (self._devtools.env("HCPD_SCOPE_BENCH") == "1"
               or self._window.is_right_mouse_down())
        reloading = match is not None and match.reloading(now)
        want = on_screen and is_precision and held and not reloading
        if want and not self._album.is_scoped:
            self._album.enter_scope()
            self._virtual_camera.scope_eye_anchor_mm = \
                self._virtual_camera.eye_position_mm.copy()
        elif not want and self._album.is_scoped:
            self._end_scope()
        dt = (0.0 if self._scope_last_t is None
             else max(0.0, now - self._scope_last_t))
        self._scope_last_t = now
        self._album.step_scope(dt)
        crosshair = raw_mouse
        if self._album.is_scoped:
            crosshair = self._album.apply_scope_aim(raw_mouse)
        self._sync_scope_camera(crosshair)
        return crosshair

    def _end_scope(self) -> None:
        """Right mouse released (or reload started, or weapon/screen
        changed) — warp the OS cursor to the crosshair's CURRENT window
        pixel so the next frame's raw mouse read doesn't jump it."""
        wx, wy = self._album.painter.map(*self._album.mouse)
        self._window.set_cursor_pos(wx, wy)
        self._album.exit_scope()
        self._virtual_camera.scope_eye_anchor_mm = None

    def _sync_scope_camera(self, crosshair: tuple) -> None:
        from src.ui_album.scope import design_to_glass_m

        camera = self._virtual_camera
        if self._album.scope_visually_active:
            glass, centre = self._glass_mm()
            sx, sy = design_to_glass_m(crosshair, glass, centre)
            camera.scope_active = True
            camera.scope_zoom = self._album.scope_zoom
            camera.scope_crosshair_m = np.array(
                [sx, sy, float(camera.screen_center_m[2])], dtype=np.float64)
        else:
            camera.scope_active = False

    def _album_action(self, action: str, now: float) -> None:
        """Thực hiện một hành động mà giao diện album yêu cầu."""
        F = self._album.flow

        state = self._album.state
        if action == "pick_file":
            picked = pick_media_file()
            # Hộp thoại vừa đóng: cửa sổ game đang bị thu nhỏ, kéo nó về NGAY
            # (kể cả khi người dùng bấm Huỷ) rồi mới làm gì tiếp.
            self._window.restore_and_focus()
            if picked is not None:
                state.photo = picked
                if picked not in state.recent:
                    state.recent.insert(0, picked)
                    del state.recent[4:]
                # Chọn xong là chạy luôn bước tách lớp: trước đây app đứng
                # yên ở màn chọn ảnh, người dùng tưởng nó không nhận ảnh.
                self._album_action("start_pipeline", now)
        elif action == "start_pipeline":
            if state.photo is None:
                state.photo = self._content.resolved_media_path(self._root)
            self._try_load_media(Path(state.photo), now)
            state.pipeline_pct = 0.0
            self._album_pipeline_started_s = now
            state.go(F.PIPE0)
        elif action == "rest_eye":
            self._begin_rest_measure(now)
        elif action == "start_game":
            self._start_album_match(now)
        elif action == "next_level":
            state.level = min(5, state.level + 1)
            state.go(F.LOADOUT)
        elif action == "logout":
            # TASK_accounts_1119 Phase 4: a "remember me" session (if any)
            # must not outlive a deliberate sign-out.
            if self._album_options.is_v3:
                self._album.sign_out()
            state.player = state.password = state.password2 = ""
            state.account_key = ""
            if self._album_options.is_v3:
                self._album.sync_economy()
            state.focus = None
            state.go(F.SIGN_IN if hasattr(F, "SIGN_IN") else F.LOGIN)
        elif action == "copy_code":
            self._window.set_clipboard(f"PM-{state.room_code[:4] or '0000'}")
            F.say(state, "Code copied")
        elif action == "share_confirmed":
            path = state.shared_album_upload_path
            state.shared_album_upload_path = ""
            self._start_share_upload(path)
        elif action == "random_from_album":
            self._start_random_photo()
        elif action in ("display_full", "display_window"):
            mode = "fullscreen" if action == "display_full" else "window"
            if self._window.set_display_mode(mode):
                self._apply_display_mode(mode)
        elif action.startswith("pvp_"):
            self._pvp_action(action, now)
        elif action == "account_sign_in":
            self._album.sign_in_submit()
        elif action == "account_sign_up":
            self._album.sign_up_submit()
        elif action == "account_forgot_submit":
            self._album.forgot_submit()
        elif action == "account_copy_recovery_code":
            self._album.copy_recovery_code()
            F.say(state, "Code copied")

    def _tick_album_match(self, now: float) -> None:
        """Nhịp của màn tách lớp và của ván đang chơi."""
        F = self._album.flow

        if self._album_options.shots or self._album_options.ui_shots:
            return          # lượt chụp: kế hoạch màn là của _album_shots_step
        if self._devtools.env("HCPD_UI_SHOTS_1117") == "1":
            return          # TASK_ui_sakuga_1117: _tick_sakuga_shots so sanh man
        if self._v3_live:
            return          # U3: màn và ván do preset đặt, đừng nhịp đè lên
        state = self._album.state
        if state.screen in (F.PIPE0, F.PIPE, F.PIPE100):
            if self._renderer.waiting_for_depth:
                elapsed = now - getattr(self, "_album_pipeline_started_s", now)
                state.pipeline_pct = min(95.0, 100.0 * elapsed / 45.0)
                state.go(F.PIPE if state.pipeline_pct > 1.0 else F.PIPE0)
            else:
                state.pipeline_pct = 100.0
                state.go(F.PIPE100)
                self._refresh_bug_capacity()
        match = self._album.game
        if match is None or state.screen != F.PLAY:
            return
        if self._album_options.is_v3:
            # Ô "LAYER a / N" là chữ trang trí; một lỗi tính nó KHÔNG được
            # giết ván đang chơi. Đúng bài học của ``_album_draw_guarded``:
            # một ô bầu dục co về 0 px từng làm app văng giữa trận (06/11).
            # Lần này là gọi sai tên trường và cả vòng vẽ chết ngay lúc vào
            # trận — ghi log một lần mỗi loại rồi chơi tiếp.
            try:
                match.layer_index, match.layer_total = self._layer_progress()
            except Exception as err:                      # noqa: BLE001
                seen = getattr(self, "_layer_errors", set())
                key = f"{type(err).__name__}: {err}"
                if key not in seen:
                    seen.add(key)
                    self._layer_errors = seen
                    logger.exception("Khong tinh duoc so lop (%s) — bo qua", key)
        ended = match.tick(self._tracking.face_detected)
        if ended:
            state.last_result = {"score": match.score(),
                                 "found": match.bugs_found,
                                 "total": match.bugs_total}
            if self._album_options.is_v3:
                # Màn kết quả của bộ v3 in từng DÒNG điểm, nên nó cần đủ các
                # số hạng của phép tính, không chỉ tổng.
                state.last_result.update({
                    "time_left": max(0, int(round(match.time_left_s))),
                    "used": dict(match.used),
                    "overtime_s": match.overtime_s,
                    "eaten": match.eaten_pct})
                if state.photo is not None:
                    state.history[str(state.photo)] = {
                        "date": time.strftime("%d/%m/%Y"),
                        "eaten": match.eaten_pct,
                        "score": match.score()}
                # TASK_accounts_1119 Phase 5: save right after a match
                # result (a signed-in account) or the one-time guest
                # toast (never saved) — see V3UI.note_match_finished.
                self._album.note_match_finished(
                    won=(ended == "win"), level=match.level,
                    score=match.score())
            state.go(F.WIN if ended == "win" else F.LOSE)
            self._album.game = None

    # ---- TASK_pvp_lan_1116: PvP over LAN --------------------------------
    # Network I/O itself never runs here — ``PvpClient`` owns a background
    # thread (same shape as ``SharedAlbumWorker``); everything below only
    # ever calls its non-blocking ``send``/``poll``. ``v3_routes._pvp``
    # only changes screens and returns an ``ACT_PVP_*`` action; the actual
    # connect/queue/room/ready/hit calls happen here, via ``_pvp_action``.

    def _pvp_connect(self, state) -> None:
        if self._pvp is not None:
            return
        from src.net.client import PvpClient

        self._pvp_my_name = state.player or "guest"
        self._pvp = PvpClient(self._pvp_my_name)
        self._pvp.connect_auto(self._root)

    def _pvp_send_when_ready(self, msg: dict) -> None:
        """Queue a lobby message; flushed by ``_tick_pvp`` once connected.

        ``PvpClient.send`` itself queues to an outbox regardless of
        connection state, but its background thread sends ``hello`` first
        only once the socket is up — sending straight through here could
        race ahead of that handshake and reach the server before ``hello``.
        """
        if self._pvp is not None and self._pvp.is_connected:
            self._pvp.send(msg)
        else:
            self._pvp_pending = msg

    def _pvp_action(self, action: str, now: float) -> None:
        from src.net import protocol as proto
        from src.ui_album import v3_routes

        state = self._album.state
        if action == v3_routes.ACT_PVP_FIND:
            self._pvp_connect(state)
            self._pvp_send_when_ready({"t": proto.QUEUE, "mode": state.mode,
                                       "level": state.level})
        elif action == v3_routes.ACT_PVP_CREATE_ROOM:
            self._pvp_connect(state)
            self._pvp_send_when_ready({"t": proto.CREATE_ROOM, "mode": state.mode,
                                       "level": state.level})
        elif action == v3_routes.ACT_PVP_JOIN_ROOM:
            self._pvp_connect(state)
            self._pvp_send_when_ready({"t": proto.JOIN_ROOM,
                                       "code": state.join_typed})
        elif action == v3_routes.ACT_PVP_CANCEL:
            if self._pvp is not None and self._pvp.is_connected:
                self._pvp.send({"t": proto.CANCEL})
            self._pvp_leave_match("cancel")
        elif action == v3_routes.ACT_PVP_REMATCH:
            if self._pvp is not None and self._pvp.is_connected:
                self._pvp.send({"t": proto.REMATCH})
                self._album.flow.say(state, "Waiting for a rematch...")
        elif action == v3_routes.ACT_PVP_RECONNECT:
            if self._pvp is not None:
                self._pvp.close()
                self._pvp = None
            self._pvp_connect(state)
            if state.pvp_token:
                # Match/local state (self._album.game, self._game) survived
                # the disconnect untouched — rejoin just needs to re-attach
                # this new socket to that same seat, not restart the whole
                # lobby handshake (see _pvp_handle_matched's "resumed" path).
                self._pvp_send_when_ready({"t": proto.REJOIN,
                                           "token": state.pvp_token})

    def _pvp_leave_match(self, reason: str = "left") -> None:
        """Dọn phiên PvP hiện tại (huỷ/rời/lỗi) — KHÔNG đóng kết nối, chỉ
        dọn trạng thái trận. ``reason`` chỉ để log."""
        state = self._album.state
        state.pvp_role = state.pvp_token = state.pvp_match_id = ""
        state.pvp_opponent_name = ""
        state.pvp_started = False
        self._pvp_is_host = False
        self._pvp_photo_sent = False
        self._pvp_photo_ready = False
        self._pvp_bugs_sent = False
        self._pvp_bugs_attempted = False
        self._pvp_ready_sent = False
        self._pvp_active = False
        self._album.game = None
        self._game, self._game_planes = None, ()
        self._renderer.set_props([])
        logger.info("PvP: roi tran (%s)", reason)

    def _tick_pvp(self, now: float) -> None:
        """Nhịp mỗi khung: rút hộp thư ``PvpClient`` — không bao giờ chặn."""
        self._album.state.pvp_lobby_message = self._pvp_lobby_message()
        if self._pvp is None:
            return
        if self._pvp_pending is not None and self._pvp.is_connected:
            self._pvp.send(self._pvp_pending)
            self._pvp_pending = None
        for msg in self._pvp.poll():
            self._pvp_handle(msg, now)
        self._pvp_tick_lobby(now)
        self._pvp_tick_eye(now)

    def _pvp_tick_eye(self, now: float) -> None:
        """Hunt & Hide: mắt của hunter gửi tới 10 Hz cho hider tính
        exposure (task: "sends its eye position at 10 Hz")."""
        F = self._album.flow
        state = self._album.state
        if not (self._pvp_active and self._pvp is not None
                and state.mode == F.PVP_MODES[1] and state.pvp_role == "hunter"):
            return
        if now - self._pvp_last_eye_sent_s < 0.1:        # 10 Hz
            return
        self._pvp_last_eye_sent_s = now
        from src.net import protocol as proto

        eye = self._virtual_camera.eye_position_mm
        self._pvp.send({"t": proto.EYE, "x": float(eye[0]), "y": float(eye[1]),
                        "z": float(eye[2])})

    def _pvp_handle(self, msg: dict, now: float) -> None:
        from src.net import client as pvp_client_mod
        from src.net import protocol as proto

        t = msg.get("t")
        if t == pvp_client_mod.STATUS:
            self._pvp_handle_status(msg, now)
        elif t == proto.ERROR:
            self._pvp_handle_error(msg, now)
        elif t == proto.ROOM:
            self._album.state.room_code = str(msg.get("code", ""))
        elif t == proto.MATCHED:
            self._pvp_handle_matched(msg, now)
        elif t == proto.PHOTO:
            self._pvp_handle_photo(msg, now)
        elif t == proto.START:
            self._pvp_handle_start(msg, now)
        elif t == proto.KILLED:
            self._pvp_handle_killed(msg, now)
        elif t == proto.TICK:
            self._pvp_handle_tick(msg, now)
        elif t == proto.END:
            self._pvp_handle_end(msg, now)
        elif t == proto.EYE:
            self._pvp_handle_eye(msg, now)

    def _pvp_handle_status(self, msg: dict, now: float) -> None:
        F = self._album.flow
        state = self._album.state
        status = msg.get("status")
        if status == "error":
            F.say(state, f"PvP: {msg.get('message') or 'could not connect'}", 6.0)
        elif status == "disconnected":
            if state.screen in (F.PVP_SEARCH, F.PVP_INVITE, F.PVP_FOUND,
                                F.PVP_HUNT, F.PVP_HIDE):
                state.offline_since_s = now
                state.go(F.OFFLINE)

    def _pvp_handle_error(self, msg: dict, now: float) -> None:
        F = self._album.flow
        state = self._album.state
        code = msg.get("code")
        message = str(msg.get("message") or "") or "PvP error"
        if code == "version_mismatch":
            F.say(state, message, 10.0)
            if self._pvp is not None:
                self._pvp.close()
                self._pvp = None
            state.go(F.PVP_MODE)
        elif code == "room_not_found":
            F.say(state, message, 5.0)
        else:
            F.say(state, message, 6.0)

    def _pvp_handle_matched(self, msg: dict, now: float) -> None:
        F = self._album.flow
        state = self._album.state
        if msg.get("resumed"):
            # TASK_pvp_lan_1116 P5: reconnect — the match/local state
            # (self._album.game, self._game) survived the disconnect
            # untouched (see _pvp_handle_status), so resuming only needs a
            # new token and the right screen back, NOT the whole lobby
            # handshake (that would re-send the photo/bugs and restart the
            # match from scratch).
            state.pvp_token = str(msg.get("token") or state.pvp_token)
            F.say(state, "Reconnected.", 3.0)
            if self._pvp_active:
                hunting = (state.mode == F.PVP_MODES[0]
                          or state.pvp_role == "hunter")
                state.go(F.PVP_HUNT if hunting else F.PVP_HIDE)
            else:
                state.go(F.PVP_FOUND)
            return
        state.mode = str(msg.get("mode") or state.mode)
        try:
            state.level = int(msg.get("level", state.level))
        except (TypeError, ValueError):
            pass
        state.pvp_role = str(msg.get("role") or "")
        state.pvp_token = str(msg.get("token") or "")
        state.pvp_match_id = str(msg.get("room") or "")
        state.pvp_opponent_name = str(msg.get("opponent_name") or "Opponent")
        state.pvp_started = False
        self._pvp_is_host = bool(msg.get("is_host"))
        self._pvp_photo_sent = False
        self._pvp_photo_ready = False
        self._pvp_bugs_sent = False
        self._pvp_bugs_attempted = False
        self._pvp_ready_sent = False
        self._pvp_active = False
        state.found_at_s = now
        state.go(F.PVP_FOUND)
        if self._pvp_is_host:
            photo_path = (Path(state.photo) if state.photo
                         else self._content.resolved_media_path(self._root))
            self._try_load_media(photo_path, now)
            self._pvp_photo_ready = True
            self._pvp_send_photo(photo_path)

    def _pvp_send_photo(self, path: Path) -> None:
        """Host gửi ảnh cho khách — re-encode cùng luật ``shared_album._reencode``
        (nén lại, bỏ EXIF); không bao giờ gửi file gốc thẳng ra mạng."""
        if self._pvp_photo_sent or self._pvp is None:
            return
        from src.net import protocol as proto

        data = shared_album._reencode(path)
        if data is None:
            self._album.flow.say(self._album.state,
                                 "Could not read the photo to send.")
            return
        self._pvp.send({"t": proto.PHOTO,
                        "jpeg_b64": base64.b64encode(data).decode("ascii"),
                        "sha1": hashlib.sha1(data).hexdigest()})
        self._pvp_photo_sent = True

    def _pvp_handle_photo(self, msg: dict, now: float) -> None:
        """Khách nhận ảnh: lưu dưới ``assets/cache/pvp/<sha1 CỤC BỘ>.jpg`` —
        tên file không bao giờ dựng từ ``sha1`` của xa, chỉ dùng CÁI MÌNH
        TỰ TÍNH LẠI sau khi giải mã, đúng luật "never build a file path
        from remote data"."""
        state = self._album.state
        jpeg_b64, remote_sha1 = msg.get("jpeg_b64"), msg.get("sha1")
        if not isinstance(jpeg_b64, str) or not isinstance(remote_sha1, str):
            return
        try:
            data = base64.b64decode(jpeg_b64, validate=True)
        except (ValueError, binascii.Error):
            logger.warning("PvP photo: base64 hong — bo qua")
            return
        local_sha1 = hashlib.sha1(data).hexdigest()
        if local_sha1 != remote_sha1.lower():
            logger.warning("PvP photo: sha1 khong khop — bo qua")
            return
        cache_dir = self._root / "assets" / "cache" / "pvp"
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = cache_dir / f"{local_sha1}.jpg"
        path.write_bytes(data)
        # The other player's file is only ever loaded as a clean copy.
        try:
            clean = media_gate.import_photo(path, allowed_root=cache_dir)
        except media_gate.PhotoRejected as exc:
            path.unlink(missing_ok=True)
            self._album.flow.say(state, str(exc))
            return
        state.photo = path
        self._try_load_media(path, now, clean=clean)
        self._pvp_photo_ready = True

    def _pvp_lobby_message(self) -> str:
        """
        TASK_pvp_fix_1116 fix 2: chuỗi vẽ trên ``PVP_FOUND``, tính lại MỖI
        KHUNG từ trạng thái thật — không phải ``F.say`` một lần với hẹn giờ
        30 s cứng, thứ sống sót qua cả màn ``PVP_HUNT``/kết quả vì đồng hồ
        của nó không biết gì về việc người chơi đã rời màn.

        Rỗng ở MỌI màn khác ``PVP_FOUND`` (gọi ``state.go`` đổi màn là xong,
        không cần dọn gì thêm), và rỗng ngay khi ``start`` thật đã tới
        (``state.pvp_started``), đúng yêu cầu "No message may survive into
        PVP_HUNT, PVP_HIDE or PVP_RESULT."
        """
        F = self._album.flow
        state = self._album.state
        if state.screen != F.PVP_FOUND:
            return ""
        if self._renderer.waiting_for_depth:
            return "Splitting layers…"
        if self._pvp_ready_sent and not state.pvp_started:
            return f"Waiting for {state.pvp_opponent_name or 'Opponent'}…"
        return ""

    def _pvp_tick_lobby(self, now: float) -> None:
        """Trong ``PVP_FOUND``: chờ tách lớp xong, đặt bọ (CHỈ host ở
        Competitive — hider ở Hunt & Hide đặt sau, trong pha "setup", xem
        ``_pvp_handle_start``) rồi báo ``ready``. Chạy mỗi khung tới khi
        server gửi ``start`` thật (``state.pvp_started`` bật).

        Hider KHÔNG được đặt bọ ở đây: server chỉ coi ``bugs`` là bắt đầu
        pha săn khi nó tới lúc ``phase == "setup"`` (sau ``ready`` của CẢ
        hai và ``start{bugs: []}`` đã gửi) — gửi sớm hơn thì server chỉ cất
        đi, không bao giờ mở trận (xem ``src/net/server.py:_on_bugs``).
        """
        from src.net import protocol as proto

        F = self._album.flow
        state = self._album.state
        if state.screen != F.PVP_FOUND or state.pvp_started or not state.pvp_match_id:
            return
        if not self._pvp_photo_ready:
            return                       # chua bat dau tai DUNG anh cua tran nay
        if self._renderer.waiting_for_depth:
            return
        if len(self._renderer.layer_records()) < 2:
            return                       # chua co gi de bake (vd con cho anh)
        is_host_placer = state.mode == F.PVP_MODES[0] and self._pvp_is_host
        if is_host_placer and not self._pvp_bugs_attempted:
            self._pvp_bugs_attempted = True
            self._pvp_bugs_sent = self._pvp_place_bugs_or_bail(now)
        if is_host_placer and not self._pvp_bugs_sent:
            return                       # dat bo that bai nhung chua roi man (hiem)
        if not self._pvp_ready_sent:
            if self._pvp is not None:
                self._pvp.send({"t": proto.READY})
            self._pvp_ready_sent = True

    def _pvp_place_bugs_or_bail(self, now: float) -> bool:
        """``_pvp_place_bugs`` + xử lý thất bại dùng chung cho cả hai
        người đặt (host ở Competitive trong ``_pvp_tick_lobby``, hider ở
        Hunt & Hide trong ``_pvp_handle_start``'s pha setup)."""
        F = self._album.flow
        state = self._album.state
        if self._pvp_place_bugs(now):
            return True
        F.say(state, "This photo can't hide any bugs for PvP — "
                     "pick another one.", 8.0)
        if self._pvp is not None:
            from src.net import protocol as proto

            self._pvp.send({"t": proto.CANCEL})
        self._pvp_leave_match("no_bugs")
        state.go(F.PVP_MODE)
        return False

    def _pvp_place_bugs(self, now: float) -> bool:
        """Đặt bọ (host ở Competitive, hider ở Hunt & Hide): CÙNG thuật
        toán solo (``_start_prop_game`` + ``filter_props`` +
        ``assign_bug_sizes``/``assign_bug_camo``), rồi đổi sang dạng mạng
        (``pvp_sync``) và gửi ``bugs`` cho server. ``False`` nếu ảnh không
        giấu được con nào."""
        from src.game.viewport_filter import filter_props
        from src.ui_album.bugs import assign_bug_camo, assign_bug_sizes
        from src.ui_album import pvp_sync
        from src.net import protocol as proto

        F = self._album.flow
        state = self._album.state
        count = F.LEVEL_TABLE[state.level][1]
        self._start_prop_game(count=count * 3)
        if self._game is None or not self._game.props:
            return False
        rest = self._rest_eye or (
            float(self._content.overscan_design_eye_mm[0]),
            float(self._content.overscan_design_eye_mm[1]), 450.0)
        glass, centre = self._glass_mm()
        kept, _outside = filter_props(self._game.props, rest, count, glass,
                                      self._viewport_rect(F.PVP_HUNT), centre)
        if not kept:
            return False
        self._game.props = kept
        self._game.found = [False] * len(kept)
        self._game.visible_since_s = [None] * len(kept)
        assign_bug_sizes(self._game.props, self._bug_seed)
        photo_rgb = self._photo_rgb_for_camo()
        if photo_rgb is not None:
            assign_bug_camo(self._game.props, photo_rgb, rest, glass, centre)
        self._renderer.set_props([])
        if self._bugs is not None:
            self._bugs.reset()
        wire = pvp_sync.bugs_to_wire(self._game.props, self._game_planes)
        if not wire or self._pvp is None:
            return False
        self._pvp.send({"t": proto.BUGS, "bugs": wire})
        return True

    def _pvp_handle_start(self, msg: dict, now: float) -> None:
        from src.ui_album import pvp_sync
        from src.game.levels import PVP_WEAPONS_SPEC as WEAPONS_SPEC
        from src.game.match import DEFAULT_SPEC, start_match

        F = self._album.flow
        state = self._album.state
        self._game_planes = layer_planes(self._renderer.layer_records())
        bugs = msg.get("bugs") or []
        if not bugs:
            # Hunt & Hide: đây là pha "setup" — hider đặt bọ NGAY BÂY GIỜ
            # (không phải trước đó, trong ``_pvp_tick_lobby``: server chỉ
            # coi ``bugs`` là mở trận săn khi nó tới lúc phase=="setup").
            if state.mode == F.PVP_MODES[1] and state.pvp_role == "hider":
                state.pvp_setup_deadline_s = now + 20.0  # SETUP_S (src/net/server.py)
                if not self._pvp_bugs_attempted:
                    self._pvp_bugs_attempted = True
                    self._pvp_bugs_sent = self._pvp_place_bugs_or_bail(now)
            return
        radius = bug_radius_mm(self._content)
        props = pvp_sync.bugs_from_wire(bugs, self._game_planes, radius)
        found = [False] * len(props)
        for index, prop in enumerate(props):
            if prop is None:
                # Không neo được vào lớp cục bộ (khác số lớp hai máy — hiếm
                # với cùng ảnh/config): coi như đã chết, không vẽ không bắn.
                props[index] = {"pos_mm": (0.0, 0.0, -100000.0),
                                "radius_mm": 0.0, "size_k": 0.0,
                                "camo_rgb": (0.5, 0.5, 0.5), "reveal_axis": "x",
                                "reveal_450_mm": 0.0, "reveal_330_mm": 0.0}
                found[index] = True
        self._game = GameState(image="pvp", props=props, tracking_on=True,
                               started_at_s=now, found=found)
        self._game_layers_revision = self._renderer.layers_revision
        # TASK_pvp_fix_1116 fix 3: NOT set_props(prop_items(...)) — that is
        # solo's yellow DEBUG sphere path. PvP draws real bug sprites via
        # BugPainter (self._bugs), same as _start_album_match does.
        self._renderer.set_props([])
        if self._bugs is not None:
            self._bugs.reset()
        weapon = state.weapon_name() or F.WEAPONS[0]
        items = list(state.items)
        if not self._album.charge_items(items):
            F.say(state, "Not enough points for these items — going in without.")
            items = []
        match = start_match(state.level, self._game, weapon, items,
                            F.ITEM_WEIGHT)
        match.layer_total = len(self._renderer.layer_records())
        spec = WEAPONS_SPEC.get(weapon, DEFAULT_SPEC)
        match.ammo_enabled = True
        match.mag_size = spec.mag
        match.rounds_left = spec.mag
        match.fire_interval_s = spec.fire_s
        match.reload_s = spec.reload_s
        self._album.game = match
        self._pvp_active = True
        state.pvp_started = True
        state.found_at_s = now          # dem nguoc 3s cua v3_flow.update() tu day

    def _pvp_handle_killed(self, msg: dict, now: float) -> None:
        match = self._album.game
        if match is None or self._game is None:
            return
        bug_id = msg.get("bug")
        if (isinstance(bug_id, bool) or not isinstance(bug_id, int)
                or not (0 <= bug_id < len(self._game.props))):
            return
        if not self._game.found[bug_id]:
            self._game.found[bug_id] = True
            prop = self._game.props[bug_id]
            self._renderer.set_prop_found(bug_id)
            if self._bugs is not None and prop is not None:
                self._bugs.mark_found(bug_id,
                                      [v / 1000.0 for v in prop["pos_mm"]])
        if msg.get("late"):
            return
        if msg.get("by") == self._pvp_my_name:
            match.my_kills += 1
        else:
            match.bot_found += 1

    def _pvp_handle_tick(self, msg: dict, now: float) -> None:
        """Đồng hồ đếm ngược server là CHUẨN — tự sửa trôi mỗi giây thay vì
        cộng dồn lệch dần theo đồng hồ cục bộ."""
        match = self._album.game
        if match is None:
            return
        try:
            time_left = float(msg.get("time_left", match.time_left_s))
        except (TypeError, ValueError):
            return
        match.started_s = time.perf_counter() - (match.seconds - time_left)

    def _pvp_handle_end(self, msg: dict, now: float) -> None:
        F = self._album.flow
        state = self._album.state
        scores = msg.get("scores") or {}
        state.last_result = {
            "score": scores.get("you", 0), "opponent_score": scores.get("opp", 0),
            "winner": msg.get("winner"), "opp_name": msg.get("opp_name") or "Opponent",
            "rows": msg.get("rows") or [], "opponent_rows": msg.get("opponent_rows") or [],
        }
        # G2: so sánh được với log của pvp_bot.py mà không cần nhìn màn hình.
        logger.info("PVP KET QUA: %s", state.last_result)
        self._pvp_active = False
        self._album.game = None
        self._game = None
        self._renderer.set_props([])
        state.go(F.PVP_RESULT)

    def _pvp_handle_eye(self, msg: dict, now: float) -> None:
        """Hunt & Hide: mắt của hunter tới 10 Hz — hider tính % bọ còn sống
        đang lộ ra trước mắt đó, CỤC BỘ bằng hình học lớp của chính mình
        (``hidden_props.is_visible``), không phải server tính hộ."""
        state = self._album.state
        match = self._album.game
        if (match is None or self._game is None
                or state.pvp_role != "hider"):
            return
        x, y, z = msg.get("x"), msg.get("y"), msg.get("z")
        if not all(isinstance(v, (int, float)) for v in (x, y, z)):
            return
        from src.content.hidden_props import _nearer, is_visible

        eye = (float(x), float(y), float(z))
        radius = bug_radius_mm(self._content)
        alive = [i for i in range(len(self._game.props))
                if not self._game.found[i] and self._game.props[i] is not None]
        if not alive:
            match.exposure_pct = 0.0
            return
        visible = 0
        for index in alive:
            pos_mm = self._game.props[index]["pos_mm"]
            nearer = _nearer(self._game_planes, pos_mm[2] / 1000.0)
            if is_visible(eye, pos_mm, radius, nearer):
                visible += 1
        match.exposure_pct = 100.0 * visible / len(alive)

    def _start_album_match(self, now: float) -> None:
        """Mở một ván theo cấp đang chọn: số bọ lấy từ bảng cấp."""
        F = self._album.flow
        from src.game.match import start_match

        from src.content.hidden_props_search import exposure_count
        from src.game.viewport_filter import filter_props

        state = self._album.state
        count = F.LEVEL_TABLE[state.level][1]
        # Xin NHIỀU hơn rồi lọc: màn chơi chỉ hở ô viewport, con bọ chiếu ra
        # ngoài ô thì không bao giờ thấy được nên ván không thể thắng (F3.3).
        rule = level_rules(state.level)
        rest = self._rest_eye or (
            float(self._content.overscan_design_eye_mm[0]),
            float(self._content.overscan_design_eye_mm[1]), 450.0)
        glass, centre = self._glass_mm()
        planes = layer_planes(self._renderer.layer_records())
        chosen = None
        if len(planes) >= 2:
            # Fullest of LAYOUT_TRIES layouts (cached from the loading screen
            # when nothing changed since).
            chosen = self._album_layout(planes, count, rule, rest, glass,
                                        centre).layout
        self._bug_mover = None
        self._start_prop_game(count=count * 3,
                              prefer=(rule.prefer_lo, rule.prefer_hi),
                              props=chosen.kept if chosen else None)
        if self._game is not None and self._game.props:
            kept, outside = filter_props(self._game.props, rest, count, glass,
                                         self._viewport_rect(), centre)
            if chosen is not None:
                outside = chosen.outside
            # TASK_hud_fix_1119 Fix 3: ``kept`` is only "inside the
            # viewport" — place_props' own reveal scan (hidden_props_search
            # ._candidate) only walked ONE axis at a time, so a spot that
            # never reveals on the 2D peek range (both axes moving
            # together, the shape a real lean actually takes) could still
            # slip through and reach the player as an unwinnable bug,
            # exactly logs/game_20261119_150429.log's t22.png case (1 bug
            # placed behind the cat, 0/191 khung co vat lo, every shot's
            # alpha=0.00). Double-check with a real 5x5 grid over the
            # peek range, at the screen (rest) distance, and drop anything
            # that's exposed at NONE of the 25 positions.
            winnable = []
            for prop in kept:
                exposed_n = exposure_count(self._game_planes, prop["pos_mm"],
                                           prop["radius_mm"], rest[:2], rest[2])
                logger.info("Album bo tai (%.0f, %.0f, %.0f): lo o %d/25 vi tri mat",
                           *prop["pos_mm"], exposed_n)
                if exposed_n > 0:
                    winnable.append(prop)
            dropped_unwinnable = len(kept) - len(winnable)
            kept = winnable
            logger.info("Album bo: xin %d · dat %d · ngoai o %d · khong the lo %d",
                        count, len(kept), outside, dropped_unwinnable)
            if state.bug_hold is not None and state.bug_hold != len(kept):
                logger.warning("Album bo: man tai xong bao %d con nhung van dat %d "
                               "(cua so hoac mat nghi doi sau do)",
                               state.bug_hold, len(kept))
            if not kept:
                # Không còn chỗ nào THẬT SỰ lộ được — đừng mở ván không thể
                # thắng (F3.3's own gap). Ở lại màn chọn ảnh, nói thẳng tại
                # sao, như lời báo "can only hide N" bên dưới.
                F.say(state, "This photo has no spot a bug can be found "
                             "from — pick a photo with more depth", 6.0)
                state.go(F.PHOTOS)
                return
            if self._album_options.is_v3 and len(kept) < count:
                # Màn chọn cấp hứa 12 con mà ván chỉ có 1 thì người chơi
                # tưởng hỏng. Nói thẳng ra: lỗi ở tấm ảnh, không ở họ.
                # Lời báo cũ đi qua ``renderer.show_toast``, mà ở v3 toast ấy
                # vẽ trong pass cảnh 3D nên gần như không ai thấy.
                F.say(state, f"This photo can only hide {len(kept)} of "
                             f"{count} bugs — try one with more depth.", 6.0)
            self._album_place_stats = {"xin": count, "dat": len(kept),
                                       "ngoai_o": outside}
            self._game.props = kept
            self._game.found = [False] * len(kept)
            self._game.visible_since_s = [None] * len(kept)
            # TASK_bugs_look_0130: cỡ + màu nguỵ trang đặt MỘT LẦN lúc mở
            # ván, ghi thẳng vào từng prop — vẽ (BugPainter) và bắn
            # (_aim_context) sau đó đọc lại đúng cùng giá trị này.
            from src.ui_album.bugs import assign_bug_camo, assign_bug_sizes

            assign_bug_sizes(self._game.props, self._bug_seed, rule.size)
            photo_rgb = self._photo_rgb_for_camo()
            if photo_rgb is not None:
                assign_bug_camo(self._game.props, photo_rgb, rest, glass,
                               centre, strength=rule.camo)
            # Bỏ quả cầu đánh dấu cũ: nó chỉ là lưới an toàn đặt lúc 01:00
            # ngày 07/11, khi tôi chưa chứng minh được sprite có hiện lên.
            # Người dùng xác nhận 01:17 là bọ đã hiện, nên quả cầu thành thừa.
            self._renderer.set_props([])
            if self._bugs is not None:
                self._bugs.reset()
        if self._game is None or not self._game.props:
            F.say(state, "Ảnh này không giấu được con bọ nào — chọn ảnh khác "
                         "(ảnh có chiều sâu rõ, nhiều lớp)")
            state.go(F.PHOTOS)
            return
        if self._album_options.is_v3:
            # Bộ v3 đánh khoá vật phẩm bằng **tiếng Anh**, và tên vũ khí là
            # tên thật trên bản vẽ (HUD in nó ra). Truyền kèm bảng trọng số,
            # nếu không ``score()`` tra bảng tiếng Việt và ra 0 cho mọi món.
            weapon = state.weapon_name() or F.WEAPONS[0]
            items = list(state.items)
            # Items are paid for when the match starts (points economy).
            if not self._album.charge_items(items):
                F.say(state, "Not enough points for these items.")
                state.go(F.LOADOUT)
                return
        else:
            weapon = (f"Sung {state.weapon}" if state.weapon is not None
                      else "Sung")
            items = list(state.stickers)
        match = start_match(state.level, self._game, weapon, items,
                            F.ITEM_WEIGHT)
        match.layer_total = len(self._renderer.layer_records())
        if self._album_options.is_v3:
            # Đếm đạn CHỈ ở giao diện v3; bộ album giữ nguyên lối bắn không
            # giới hạn của nó. Ba khẩu khác nhau cả băng đạn, nhịp bắn lẫn
            # thời gian nạp — xem ``WEAPONS_SPEC``.
            from src.game.match import DEFAULT_SPEC, WEAPONS_SPEC

            spec = WEAPONS_SPEC.get(weapon, DEFAULT_SPEC)
            match.ammo_enabled = True
            match.mag_size = spec.mag
            match.rounds_left = spec.mag
            match.fire_interval_s = spec.fire_s
            match.reload_s = spec.reload_s
            logger.info("Sung %s: bang %d vien, nhip %.2f s, nap %.1f s",
                        weapon, spec.mag, spec.fire_s, spec.reload_s)
        self._album.game = match
        if chosen is not None and rule.move_after_s is not None:
            from src.game.bug_capacity import spare_spots
            from src.game.bug_move import BugMover

            spares = spare_spots(self._game_planes, chosen, rest,
                                 self._viewport_rect(), glass, centre)
            self._bug_mover = BugMover(self._game, spares, rule.move_after_s,
                                       bug_radius_mm(self._content),
                                       self._bug_seed)
            logger.info("Album bo: %d cho du phong, bo doi cho sau %.1f s",
                        len(spares), rule.move_after_s)
        state.go(F.PLAY)

    def _save_album_shot(self, stem: str | None = None) -> Path:
        """F12: lưu framebuffer ra ``results/album_ui_1106/<man>_<theme>.png``."""
        from src.ui_album.shots import save_framebuffer

        state = self._album.state
        if self._album_options.is_v3:
            out = self._root / "results" / "ui_v3_1107"
            out = out / ("shots" if self._album_options.ui_shots else "live")
            path = out / f"{stem or state.screen}.png"
        else:
            out = self._root / "results" / "album_ui_1106"
            path = out / f"{stem or state.screen + '_' + state.theme}.png"
        # Cửa sổ có thể đang thu nhỏ (hộp thoại, Alt-Tab): kéo về trước đã,
        # nếu không ảnh chụp ra khung đen 0x0 và bị bỏ qua.
        self._window.restore_and_focus()
        self._window.pump_events()
        # Cỡ hỏi GLFW ngay lúc này, KHÔNG dùng self._window.width/height: hai
        # số đó rơi về cấu hình khi cửa sổ thu nhỏ.
        width, height = self._window.framebuffer_size()
        if width < 2 or height < 2:
            logger.warning("Album shot bo qua: framebuffer %dx%d", width, height)
            return path
        save_framebuffer(path, width, height)
        return path

    def _begin_rest_measure(self, now: float) -> None:
        """Phím G: đo mắt nghỉ 2 giây thay vì đặt vật ngay."""
        if len(self._renderer.layer_records()) < 2:
            self._renderer.show_toast("Chua co lop — nap anh truoc", is_error=True)
            return
        # G là "mở/đặt lại": dọn ván cũ trước (ván dở vẫn ghi một dòng CSV), để
        # người chơi không vừa đo vừa tìm, và vật cũ không nằm trên màn lúc lấy mẫu.
        self._abandon_game(now, "G")
        self._rest_samples = []
        self._rest_started_s = time.perf_counter()
        self._renderer.show_toast("Ngoi yen 2 giay...")

    def _tick_rest_measure(self, now: float, eye_mm, face: bool) -> None:
        """Gom mắt khi đang đo; hết cửa sổ thì chốt hoặc từ chối."""
        if self._rest_samples is None:
            return
        if face:
            self._rest_samples.append(tuple(float(v) for v in eye_mm))
        if now - self._rest_started_s < REST_MEASURE_S:
            return
        samples, self._rest_samples = self._rest_samples, None
        measured, why = rest_eye(samples)
        if measured is None:
            logger.info("Mat nghi bi tu choi: %s", why)
            self._renderer.show_toast("Chua ngoi yen — bam G lai", is_error=True)
            return
        self._apply_rest_eye(measured, len(samples),
                             (spread_mm(samples, 0), spread_mm(samples, 1)))

    def _apply_rest_eye(self, measured, n_frames: int, spread) -> None:
        """Gán mắt thiết kế đo được (CHỈ trong bộ nhớ), dựng lại lớp, mở ván."""
        old = tuple(float(v) for v in self._content.overscan_design_eye_mm)
        self._rest_eye, self._rest_spread = measured, spread
        self._content.overscan_design_eye_mm = (measured[0], measured[1])
        rebuilt = self._renderer.refresh_layers()
        logger.info(
            "Mat nghi do duoc (%.0f, %.0f, %.0f) mm | n=%d khung co mat | "
            "dai p10-p90 x %.0f / y %.0f mm | design eye (%.0f, %.0f) -> "
            "(%.0f, %.0f), dung lai lop=%s",
            measured[0], measured[1], measured[2], n_frames,
            spread[0], spread[1], old[0], old[1], measured[0], measured[1], rebuilt,
        )
        self._start_prop_game()

    def _new_bug_seed(self) -> int:
        """One seed per match: ``HCPD_BUG_SEED`` if set, the fixed automatch
        default for scripted automatch runs, else a real random source."""
        raw = self._devtools.env("HCPD_BUG_SEED").strip()
        if raw:
            try:
                return int(raw)
            except ValueError:
                logger.warning("HCPD_BUG_SEED=%r is not an integer — ignored", raw)
        if self._devtools.env("HCPD_ALBUM_AUTOMATCH") in ("1", "2"):
            return AUTOMATCH_BUG_SEED
        return secrets.randbits(32)

    def _refresh_bug_capacity(self) -> None:
        """Loading screen: how many bugs this photo can really hold at this
        level. Recomputed when the photo, level, layer set, window glass or
        rest eye changes (the count depends on all of them; a stale number
        would disagree with the match). The seed is drawn once and kept for
        the match, so the match places the same bugs."""
        from src.content.hidden_props import layer_planes
        state = self._album.state
        rest = self._rest_eye or (
            float(self._content.overscan_design_eye_mm[0]),
            float(self._content.overscan_design_eye_mm[1]), 450.0)
        glass, centre = self._glass_mm()
        key = (str(state.photo), state.level, self._renderer.layers_revision,
               tuple(round(v) for v in (*glass, *centre, *rest)))
        if getattr(self, "_bug_hold_key", None) == key:
            return
        if key[:3] != getattr(self, "_bug_hold_base", None):
            self._bug_hold_base = key[:3]
            state.bug_hold = None    # another photo/level: never show the old count
        now = time.perf_counter()
        if now - getattr(self, "_bug_hold_t", -1.0) < 0.25:
            return                   # window being dragged: wait for it to settle
        planes = layer_planes(self._renderer.layer_records())
        if len(planes) < 2:
            return
        self._bug_hold_key, self._bug_hold_t = key, now
        F = self._album.flow
        count = F.LEVEL_TABLE[state.level][1]
        rule = level_rules(state.level)
        choice = self._album_layout(planes, count, rule, rest, glass, centre)
        state.bug_hold = len(choice.layout.kept)
        logger.info("Album bo: anh chua duoc %d / %d (seed=%d, man tai xong)",
                    state.bug_hold, count, self._pending_bug_seed)

    def _album_layout(self, planes, count, rule, rest, glass, centre):
        """The kept layout for this match (``bug_capacity.choose_layout``).

        One seed per match (``_pending_bug_seed``), so the loading screen and
        the match share one answer: the result is cached on everything it
        depends on and recomputed only when one of those changes."""
        from src.game.bug_capacity import choose_layout

        if self._pending_bug_seed is None:
            self._pending_bug_seed = self._new_bug_seed()
        seed = self._pending_bug_seed
        prefer = (rule.prefer_lo, rule.prefer_hi)
        key = (seed, count, prefer, self._renderer.layers_revision,
               tuple(round(v, 1) for v in (*glass, *centre, *rest)))
        cached = getattr(self, "_layout_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        t0 = time.perf_counter()
        choice = choose_layout(planes, count, bug_radius_mm(self._content),
                               rest, glass, self._viewport_rect(), centre,
                               seed, prefer)
        logger.info("Album bo: %d bo cuc %s · giu #%d (%d con)",
                    len(choice.counts), choice.counts, choice.index,
                    len(choice.layout.kept))
        logger.info("Chon bo cuc mat %.0f ms", (time.perf_counter() - t0) * 1000)
        self._layout_cache = (key, choice)
        return choice

    def _start_prop_game(self, count: int = PROP_COUNT,
                         prefer=LEGACY_PREFER, props=None) -> None:
        """
        Đặt vật ẩn từ hình học lớp THẬT của renderer, tại mắt nghỉ đo được.

        ``count`` mặc định ``PROP_COUNT`` = y như trước; giao diện album
        truyền số bọ của cấp đang chơi (TASK_album_ui_1106).
        """
        planes = layer_planes(self._renderer.layer_records())
        if len(planes) < 2:
            self._renderer.show_toast("Chua co lop — nap anh truoc", is_error=True)
            return
        radius_mm = bug_radius_mm(self._content)
        name = Path(self._content.content_path).name or "unknown"
        rest = self._rest_eye or (
            float(self._content.overscan_design_eye_mm[0]),
            float(self._content.overscan_design_eye_mm[1]), 450.0,
        )
        t0 = time.perf_counter()
        self._bug_seed = self._pending_bug_seed
        if self._bug_seed is None:
            self._bug_seed = self._new_bug_seed()
        self._pending_bug_seed = None
        self._bug_hold_key = None        # next loading screen recomputes
        logger.info("Album bo: seed=%d", self._bug_seed)
        probe = self._devtools.env("HCPD_PROP_PROBE")
        if props is not None:
            # Already placed (the fullest layout, ``_album_layout``).
            self._game = GameState(image=name, props=list(props),
                                   tracking_on=not self._tracking_off,
                                   started_at_s=t0)
        elif probe == "1":
            # Chẩn đoán thứ tự vẽ (C2): bỏ qua tìm chỗ, đặt ở độ sâu biết trước.
            self._game = GameState(image=name, props=probe_props(planes),
                                   tracking_on=True, started_at_s=t0)
        else:
            self._game = start_game(
                planes, radius_mm, rest[:2], z_rest_mm(rest[2]), t0, name,
                tracking_on=not self._tracking_off, count=count,
                seed=self._bug_seed, prefer=prefer,
            )
            if probe == "2":
                # Vật thật, chỉ tô màu khác nhau để biết quả nào hiện.
                self._game.props = probe_at(self._game.props)
        place_ms = (time.perf_counter() - t0) * 1000.0
        self._game_planes = planes
        # Cửa sổ nghiệm thu: 5 s đầu ngồi yên phải có 0 khung lộ vật.
        self._watch_until_s = t0 + WATCH_S
        self._watch_frames = self._watch_seen = 0
        self._renderer.set_props(prop_items(self._game, radius_mm))
        self._game_layers_revision = self._renderer.layers_revision
        logger.info("%s | dat vat %.0f ms", status_line(self._game), place_ms)
        self._toast_prop_count(len(self._game.props))

    def _toast_prop_count(self, count: int) -> None:
        """Toast theo k tìm được (C1 "any image": k < n và k = 0 đều phải báo)."""
        if count == 0:
            self._renderer.show_toast(
                "Anh nay khong giau duoc vat nao", is_error=True
            )
        elif count < PROP_COUNT:
            self._renderer.show_toast(f"Anh nay chi giau duoc {count} vat")
        else:
            self._renderer.show_toast(f"Prop game: {count} vat")

    def _sync_prop_game(self, now: float) -> None:
        """Huỷ ván nếu hình học lớp đổi giữa chừng. G mới là đường mở ván."""
        revision = self._renderer.layers_revision
        if revision == self._game_layers_revision:
            return
        self._game_layers_revision = revision
        if self._game is None:
            return
        self._abandon_game(now, "geometry")
        logger.info("Prop game huy: hinh hoc lop doi (revision %d)", revision)
        # Giao diện v3 KHÔNG còn phím G (đồ nghề sandbox đã bỏ), nên đừng bảo
        # người chơi bấm nó. Đường mở lại ván ở đó là nút Start / Enter match.
        self._renderer.show_toast(
            "Layers changed — start the match again" if self._album_options.is_v3
            else "Hinh hoc doi — bam G de choi lai")

    def _abandon_game(self, now: float, reason: str) -> None:
        """
        Dọn ván hiện tại; ván còn dở (có vật, chưa tìm đủ) vẫn ghi một dòng CSV.

        Không có dòng này thì lượt đối chứng tracking OFF — vốn không bao giờ
        tìm đủ — không để lại dấu vết nào. ``reason`` vào cột ``end_reason``:
        G / new_image / geometry / exit. Ván đã xong đã có dòng ``finished``.
        """
        game = self._game
        if game is not None and game.props and game.finished_at_s is None:
            logger.info("Prop game bo do (%s): %d/%d vat sau %.1f s", reason,
                        sum(game.found), len(game.props), now - game.started_at_s)
            self._write_game_csv(now, reason)
        self._game, self._game_planes = None, ()
        self._watch_until_s = None
        self._renderer.set_props([])
        self._abandon_ui_match(reason)

    def _abandon_ui_match(self, reason: str) -> None:
        """
        Ván vật ẩn đi thì ván GIAO DIỆN phải đi theo.

        Trước đây chỉ ``self._game`` bị dọn, còn ``self._album.game`` ở lại:
        người chơi đứng trên HUD với 0 con bọ, không bao giờ thắng được, và
        cú bấm kế tiếp đọc ``self._game.props`` trên ``None`` rồi giết cả
        vòng vẽ. Đưa họ về màn trang bị và nói rõ vì sao.
        """
        if self._album is None or self._album.game is None:
            return
        flow = self._album.flow
        self._album.game = None
        if self._album.state.screen not in (flow.PLAY, flow.PVP_HUNT):
            return
        self._album.state.go(flow.LOADOUT)
        if self._album_options.is_v3:
            # TASK_game_feel_1118 P3: ``go()`` clears any match toast when
            # LEAVING PLAY/PVP_HUNT/PVP_HIDE — must run first, or it would
            # wipe out THIS deliberate explanation right after setting it.
            flow.say(self._album.state,
                     "The photo layers changed — set up the match again.")
        logger.info("Ván giao diện dừng theo ván vật ẩn (%s)", reason)

    def _toggle_tracking(self) -> None:
        """
        Phím T. Ván đang chạy bị đánh dấu OFF ngay khi tracking tắt, và bật lại
        KHÔNG xoá dấu đó: dòng CSV ghi ON nghĩa là cả ván đều có tracking.
        """
        self._tracking_off = not self._tracking_off
        if self._game is not None and self._tracking_off:
            self._game.tracking_on = False
        state = "OFF (mat ghim o tu the nghi)" if self._tracking_off else "ON"
        logger.info("Prop game tracking: %s", state)
        self._renderer.show_toast(f"Tracking {state}")

    def _control_eye(self):
        """
        Mắt ghim của điều kiện đối chứng (T OFF), hoặc ``None`` = theo đầu thật.

        Đang đo mắt nghỉ (sau G) thì KHÔNG ghim: phép đo phải lấy đầu thật, nếu
        không G lần hai đo lại chính mắt đã ghim (dao động 0 mm) — số đo bịa.
        """
        if self._tracking_off and self._rest_eye is not None and self._rest_samples is None:
            return np.asarray(self._rest_eye, dtype=np.float64)
        return None

    def _tick_layer_wait(self, now: float) -> None:
        """
        Ảnh chưa tách lớp xong: toast đếm giây (PB-051).

        Ảnh phẳng lúc chờ nằm trên mặt kính nên nghiêng đầu không đổi gì; không
        có toast thì người dùng tưởng app treo. Đã thử nghỉ 20 ms mỗi khung cho
        DepthWorker bớt tranh GIL: A/B 3+3 lần 50,1 vs 56,8 s, hai dải chồng nhau
        → chưa chứng minh được, đã bỏ (16/10).
        """
        if not self._renderer.waiting_for_depth:
            if self._layer_wait_since is not None:
                logger.info("Cho tach lop: %.1f s", now - self._layer_wait_since)
                self._layer_wait_since = None
                # PB-055: một toast mỗi khi lớp dựng xong. TASK_account_persist_1121:
                # không hiện trong album — v3 đã bỏ phím G (xem V3_KEYS), nên
                # "Bam G de choi" sai ngay cả khi demo+album cùng bật hôm nay.
                if self._demo and not self._album_options.enabled:
                    self._renderer.show_toast("Bam G de choi")
            return
        if self._layer_wait_since is None:
            self._layer_wait_since, self._layer_wait_toast_s = now, 0.0
        waited = now - self._layer_wait_since
        if waited > LAYER_WAIT_MAX_S:
            return
        if waited - self._layer_wait_toast_s >= LAYER_WAIT_TOAST_S:
            self._layer_wait_toast_s = waited
            self._renderer.show_toast(f"Dang tach lop anh... {waited:.0f} s")

    def _tick_prop_game(self, now: float, eye_mm) -> None:
        """Việc mỗi khung: kiểm vật nào vừa đủ 0,5 giây liên tục."""
        if self._game is None or not self._game.props:
            return
        was_done = self._game.finished_at_s is not None
        radius_mm = bug_radius_mm(self._content)
        eye = (float(eye_mm[0]), float(eye_mm[1]), float(eye_mm[2]))
        auto = self._album is None          # album: phải BẮN mới diệt được
        for index in game_update(self._game, eye, self._game_planes, now,
                                 radius_mm, auto_find=auto):
            self._renderer.set_prop_found(index)
            if self._album is not None and self._bugs is not None:  # F3: tan + bụi
                self._bugs.mark_found(
                    index, [v / 1000.0 for v in self._game.props[index]["pos_mm"]])
            logger.info(
                "Prop FOUND #%d — eye (%.0f, %.0f, %.0f) mm",
                index, eye[0], eye[1], eye[2],
            )
            self._renderer.show_toast(
                f"Found {sum(self._game.found)}/{len(self._game.props)}"
            )
        self._tick_bug_moves(now, eye)
        self._tick_watch_window(now)
        if not was_done and self._game.finished_at_s is not None:
            self._write_game_csv(now, "finished")

    def _tick_bug_moves(self, now: float, eye) -> None:
        """Levels 4-5: a bug seen for long enough moves to a spare spot."""
        mover = getattr(self, "_bug_mover", None)
        match = self._album.game if self._album is not None else None
        if (mover is None or mover.game is not self._game or match is None
                or match.ended or self._pvp_active):
            return
        for index, old, new, seen_s in mover.tick(self._game_planes, eye, now):
            prop = self._game.props[index]
            photo_rgb = self._photo_rgb_for_camo()
            if photo_rgb is not None:
                from src.ui_album.bugs import assign_bug_camo

                rest = self._rest_eye or (
                    float(self._content.overscan_design_eye_mm[0]),
                    float(self._content.overscan_design_eye_mm[1]), 450.0)
                glass, centre = self._glass_mm()
                assign_bug_camo([prop], photo_rgb, rest, glass, centre,
                                strength=prop.get("camo_strength", 0.75))
            match.bug_last_hit.pop(index, None)      # its HP bar was at the old spot
            logger.info("Album bo #%d doi cho: (%.0f, %.0f, %.0f) -> (%.0f, %.0f, %.0f)",
                        index, *old, *new)
            from src.game.match import BUG_HP_BY_LEVEL

            full = BUG_HP_BY_LEVEL[match.level]
            logger.info("Album bo #%d thay lien tuc %.2f s · mau giu nguyen %.0f / %.0f"
                        " · con %d cho du phong",
                        index, seen_s, match.bug_hp.get(index, full), full,
                        len(mover.spares))

    def _tick_watch_window(self, now: float) -> None:
        """Đếm khung lộ vật trong 5 giây đầu sau G; in một dòng khi hết."""
        if self._watch_until_s is None:
            return
        if now < self._watch_until_s:
            self._watch_frames += 1
            if any(s is not None or f
                   for s, f in zip(self._game.visible_since_s, self._game.found)):
                self._watch_seen += 1
            return
        self._watch_until_s = None
        logger.info(
            "Cua so %gs sau G: %d/%d khung co vat lo (can 0) | mat nghi "
            "(%.0f, %.0f, %.0f) | k=%d",
            WATCH_S, self._watch_seen, self._watch_frames,
            *(self._rest_eye or (0.0, 0.0, 0.0)), len(self._game.props),
        )

    def _write_game_csv(self, now: float, end_reason: str) -> None:
        """Một dòng kết quả dưới ``results/``. Không bao giờ làm hỏng ván chơi."""
        rest = self._rest_eye or (0.0, 0.0, 0.0)
        row = csv_row(self._game, now, self._content.overscan_design_distance_mm,
                      rest, self._rest_spread, end_reason)
        try:
            path = append_csv(self._root / "results" / "prop_game.csv", row)
        except OSError as exc:
            logger.error("Khong ghi duoc CSV ket qua: %s", exc)
            return
        logger.info("Ket qua ghi vao %s: %s", path, row)
        verb = "Xong" if end_reason == "finished" else f"Dung ({end_reason})"
        self._renderer.show_toast(f"{verb} {row[2]}/{row[3]} trong {row[4]}s")

    def run(self) -> int:
        """Start workers and enter render loop."""
        # PB-051: lấy message Windows sau MỖI bước chậm. Trước đây từ lúc có
        # cửa sổ tới khung đầu (nạp ảnh, mở camera ≈ 2,6 s, ...) không lấy message
        # nào → dwm thay cửa sổ bằng bản ma "Not Responding" (run12), run8 bị thu
        # nhỏ tới khi có lớp. Giữ thứ tự cũ: mở camera TRƯỚC cửa sổ (run13) làm cửa
        # sổ hiện muộn hơn và phím không ăn [GIẢ ĐỊNH về cơ chế].
        self._window.create()
        # TASK_game_feel_1118 P1: GLFW chạy một modal loop bên trong
        # poll_events() khi người dùng đang kéo cạnh cửa sổ — main loop
        # không tới lượt, nên đăng ký NGAY để khung đầu tiên trong lúc kéo
        # cũng vẽ lại (``_on_refresh`` tự bỏ qua nếu chưa có khung nào).
        self._window.set_refresh_handler(self._on_refresh)
        self._window.pump_events()
        self._renderer = SceneRenderer(
            self._window,
            self._shader_dir,
            self._build_scene(),
            self._virtual_camera,
            preview_settings=self._ui_settings,
            content=self._content,
            project_root=self._root,
        )
        self._window.pump_events()
        self._renderer.initialize()
        if self._album_options.enabled:
            if self._album_options.is_v3:
                from src.ui_album.v3_ui import V3UI

                if self._is_scripted_run():
                    # TASK_perf_textures_1119 Fix 8: V3UI.__init__ itself
                    # reads accounts.json/session.json (and, with a valid
                    # "remember me" session, WRITES session_expires_at
                    # back) the moment it's constructed — too early for
                    # _shutdown's guard above to help. Redirect BEFORE
                    # that constructor call, every scripted run, so it
                    # can never touch the real files regardless of what
                    # this machine's account/session state happens to be.
                    import tempfile

                    from src.account import store as account_store

                    # TASK_economy_1124: a run may reuse ONE temp dir (restart
                    # test) and start from a copy of a fixture accounts file.
                    # Never a dir under the repo's own assets/config.
                    keep = self._devtools.env("HCPD_ECONOMY_DIR").strip()
                    if keep and (Path(keep).resolve() / "x").is_relative_to(
                            (self._root / "assets" / "config").resolve()):
                        raise SystemExit("HCPD_ECONOMY_DIR must not be assets/config")
                    if keep:
                        tmp_dir = Path(keep)
                        tmp_dir.mkdir(parents=True, exist_ok=True)
                    else:
                        tmp_dir = Path(tempfile.mkdtemp(prefix="hcpd_scripted_run_"))
                    seed = self._devtools.env("HCPD_ECONOMY_SEED").strip()
                    target = tmp_dir / "assets" / "config" / "accounts.json"
                    if seed and not target.exists():
                        import shutil

                        target.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copyfile(seed, target)       # a COPY; the source is only read
                    self._scripted_run_tmp_dir = tmp_dir
                    account_store.accounts_path = (
                        lambda root: tmp_dir / "assets" / "config" / "accounts.json")
                    account_store.session_path = (
                        lambda root: tmp_dir / "assets" / "config" / "session.json")
                self._album = V3UI(self._root, self._shader_dir,
                                   self._album_options, window=self._window)
                signin = self._devtools.env("HCPD_ECONOMY_SIGNIN").strip()
                if signin and self._is_scripted_run():
                    self._album.dev_sign_in(signin)
            self._album.initialize()
            self._bugs = self._build_bug_painter()
            self._renderer.extra_scene_draw = self._draw_album_bugs
            if self._album_options.is_v3:
                # D2: từ đây, dời hoặc đổi cỡ cửa sổ là dời tấm kính thật.
                self._window.set_geometry_listener(self._refresh_window_plane)
            if self._album_options.ui_shots:
                out = self._root / "results" / "ui_v3_1107" / "shots"
                for old in out.glob("*.png"):
                    old.unlink()
                self._album.apply_shot(0)
            if self._album_options.shots:
                # Xoá bộ ảnh cũ TRƯỚC khi chụp: lượt 06/11 22:30 chết giữa
                # chừng mà `album_shots_check.py` vẫn báo 50/50 vì nó chấm
                # đúng những file của lượt trước.
                out = self._root / "results" / "album_ui_1106"
                for old in out.glob("*.png"):
                    old.unlink()
                # Đặt màn đầu của kế hoạch TRƯỚC khung vẽ đầu tiên, nếu không
                # ảnh thứ nhất mang tên màn này nhưng là nội dung màn khác.
                self._album.apply_shot(0)
            logger.info("UI %s: man dau %s, theme %s",
                        self._album_options.ui, self._album.state.screen,
                        self._album.state.theme)
        self._window.pump_events()
        if not self._no_webcam:
            self._camera.start()
        self._window.pump_events()
        if self._album_options.is_v3:
            # Không có F4 thì không ai xem ảnh landmark: đừng dựng nó.
            self._tracking.set_preview_enabled(False)
        self._tracking.start()
        self._window.pump_events()

        logger.info(
            "HCPD running — F1 debug, F2 filter, F3 prediction, F4 preview, "
            "F6 open media, F7 mist A/B, F8 atmospheric A/B, F9 disocclusion fill A/B, "
            "F10 relief A/B, G do mat nghi + dat vat, T tracking off, "
            "+/- zoom offset, [ ] plane Z (can gat chinh), , . far layer Z, ; ' overscan e_max"
        )
        # TASK_account_persist_1121: chi log khi demo chay MOT MINH — trong
        # album (v3) G/T/F4/F6 da khong con tac dung (xem V3_KEYS), nen dong
        # nay sai neu in ca luc --demo --album cung bat.
        if self._demo and not self._album_options.enabled:
            logger.info(
                "DEMO MODE (PB-055): chi G, T, F6, keo-tha, F4, Esc co tac dung; "
                "doc content.demo.json, khong ghi file config"
            )

        # FR-DEPTH-03: lần nạp lúc khởi động xảy ra trong ContentPlaneRenderer
        # .__init__, trước khi có toast — lấy lại thông báo ở đây.
        startup_notice = self._renderer.take_content_notice()
        if startup_notice is not None:
            self._renderer.show_toast(startup_notice)

        last_input = time.perf_counter()
        exit_reason = "unknown"
        try:
            while not self._window.should_close:
                frame_start = self._profiler.begin_frame()
                stage_profile.begin_frame()
                now = time.perf_counter()
                dt = now - last_input
                last_input = now

                input_state = self._window.poll_input()
                if input_state.quit_app:
                    exit_reason = "quit_app (ESC pressed)"
                    self._abandon_game(now, "exit")
                    break
                # Đồ nghề sandbox của lõi HCPD: HUD gỡ lỗi, đổi bộ lọc, dự
                # đoán, ô webcam PiP, mở file, và bốn cần gạt A/B. Giao diện
                # v3 là bản cho người chơi nên **cả khối** này nằm ngoài luồng
                # của nó — một cổng ở đây, thay vì rải chín phép thử. Phím đã
                # bị lọc ở ``V3_KEYS`` rồi; chặn thêm ở đây để sau này có ai
                # đặt cờ bằng đường khác thì cũng không lọt.
                if not self._album_options.is_v3:
                    if input_state.toggle_debug:
                        self._renderer.debug_overlay.toggle()
                    if input_state.cycle_filter:
                        new_type = self._filters.cycle_smoother()
                        logger.info("Filter switched to: %s", new_type)
                    if input_state.toggle_prediction:
                        enabled = self._filters.toggle_prediction()
                        logger.info("Prediction: %s", "ON" if enabled else "OFF")
                    if input_state.toggle_webcam_preview:
                        self._renderer.webcam_preview.toggle()
                        logger.info(
                            "Webcam preview: %s",
                            "ON" if self._renderer.webcam_preview.visible else "OFF",
                        )
                    if input_state.open_content_file:
                        picked = pick_media_file()
                        if picked is not None:
                            self._try_load_media(picked, now)
                    if input_state.toggle_mist:
                        state = self._renderer.toggle_mist()
                        if state is not None:
                            msg = f"Mist {'ON' if state else 'OFF'}"
                            logger.info(msg)
                            self._renderer.show_toast(msg)
                    if input_state.toggle_atmospheric:
                        state = self._renderer.toggle_atmospheric()
                        if state is not None:
                            msg = f"Atmospheric {'ON' if state else 'OFF'}"
                            logger.info(msg)
                            self._renderer.show_toast(msg)
                    if input_state.toggle_relief:
                        state = self._renderer.toggle_relief()
                        if state is not None:
                            msg = f"Relief {'ON' if state else 'OFF'}"
                            logger.info(msg)
                            self._renderer.show_toast(msg)
                    if input_state.toggle_disocclusion:
                        state = self._renderer.toggle_disocclusion()
                        if state is not None:
                            msg = f"Disocclusion fill {'ON' if state else 'OFF'}"
                            logger.info(msg)
                            self._renderer.show_toast(msg)

                # Quét chiều sâu (Phase 2). Khoảng cách mắt lấy LIVE từ camera
                # ảo, không phải hằng 600 mm: tỉ lệ lọt khung và tỉ số thị sai
                # đều phụ thuộc chỗ người dùng đang ngồi thật.
                depth_step = self._renderer.depth_step_m
                eye_d_mm = abs(
                    float(self._virtual_camera.eye_position_m[2])
                    - float(self._virtual_camera.screen_center_m[2])
                ) * 1000.0
                # Bảng công thức chiều sâu quy theo MÀN HÌNH vật lý. Từ D2,
                # camera.screen_width_m là cỡ cửa sổ, không phải cỡ màn.
                screen_w_mm = float(self._screen_cal["screen_width"]) * 1000.0
                screen_h_mm = float(self._screen_cal["screen_height"]) * 1000.0
                if input_state.plane_closer or input_state.plane_farther:
                    step = depth_step if input_state.plane_closer else -depth_step
                    value = self._renderer.nudge_plane_z(
                        step, eye_d_mm, screen_w_mm, screen_h_mm
                    )
                    if value is not None:
                        self._renderer.show_toast(f"Plane Z {value:+.2f} m")
                if input_state.far_layer_closer or input_state.far_layer_farther:
                    step = depth_step if input_state.far_layer_closer else -depth_step
                    value = self._renderer.nudge_layer_z_far(
                        step, eye_d_mm, screen_w_mm, screen_h_mm
                    )
                    if value is not None:
                        self._renderer.show_toast(f"Far layer Z {value:+.2f} m")

                if input_state.overscan_down or input_state.overscan_up:
                    step = self._renderer.overscan_step_mm
                    step = -step if input_state.overscan_down else step
                    value = self._renderer.nudge_overscan(
                        step, eye_d_mm, screen_w_mm, screen_h_mm
                    )
                    if value is not None:
                        self._renderer.show_toast(f"Overscan e_max {value:.0f} mm")

                if input_state.zoom_in:
                    self._virtual_camera.adjust_zoom(self._virtual_camera.zoom_step_mm)
                    logger.info(
                        "Zoom offset: %+.1f mm (d_effective = d_tracked + offset)",
                        self._virtual_camera.zoom_offset_mm,
                    )
                if input_state.zoom_out:
                    self._virtual_camera.adjust_zoom(-self._virtual_camera.zoom_step_mm)
                    logger.info(
                        "Zoom offset: %+.1f mm (d_effective = d_tracked + offset)",
                        self._virtual_camera.zoom_offset_mm,
                    )

                dropped = self._window.pop_dropped_path()
                while dropped:
                    self._try_load_media(Path(dropped), now)
                    dropped = self._window.pop_dropped_path()

                if self._album is not None:
                    with stage_profile.stage("tick_album_logic"):
                        self._tick_album(input_state, now)
                # G đã bị bỏ khỏi giao diện v3 (xem V3_KEYS); chặn luôn ở đây
                # để nhánh này không còn đường chạy nào trong luồng v3.
                if input_state.game_toggle and not self._album_options.is_v3:
                    self._begin_rest_measure(now)
                if input_state.toggle_tracking:
                    self._toggle_tracking()
                self._sync_prop_game(now)
                self._tick_layer_wait(now)

                self._tracking.apply_input(
                    input_state.move_x,
                    input_state.move_y,
                    input_state.move_z,
                    dt,
                )

                snapshot = self._pose_bus.latest()
                pose = snapshot.pose
                pose_age_s = snapshot.age_ms(now) * 0.001

                with stage_profile.stage("filters"):
                    if self._render_extrapolation and self._filters.stage_info.prediction_enabled:
                        pose = self._filters.extrapolate_for_display(pose, pose_age_s)

                with stage_profile.stage("kooima_camera"):
                    eye_offset_mm = np.asarray(
                        self._screen_cal["eye_offset"],
                        dtype=np.float64,
                    ) * 1000.0
                    self._virtual_camera.set_eye_mm(pose.translation_mm + eye_offset_mm)
                    control_eye = self._control_eye()
                    if control_eye is not None:
                        # Điều kiện đối chứng: mắt đứng im ở tư thế nghỉ ĐO ĐƯỢC.
                        self._virtual_camera.set_eye_mm(control_eye)
                live_eye = self._virtual_camera.eye_position_mm
                self._tick_rest_measure(now, live_eye, self._tracking.face_detected)
                self._tick_prop_game(now, live_eye)

                pitch, yaw, roll = self._tracking.head_angles_deg()
                render_start = time.perf_counter()
                preview = (None if self._album_options.is_v3
                           else self._tracking.preview_snapshot())
                scene = lambda: self._renderer.render(        # noqa: E731
                    self._build_metrics(pose, pitch, yaw, roll),
                    frame=self._camera.peek_frame(),
                    tracking_preview=preview,
                    dt_s=dt,
                )
                # TASK_game_feel_1118 P1: nhớ lại closure này — lúc cửa sổ
                # bị kéo cạnh, ``_on_refresh`` vẽ lại bằng ĐÚNG scene này
                # (pose/khung webcam đã chụp ở trên), không tự đọc gì mới.
                self._last_scene = scene
                self._drawing = True
                try:
                    with stage_profile.stage("draw_total"):
                        self._draw_frame_only(scene)
                finally:
                    self._drawing = False
                if self._album is not None:
                    self._devtools.tick_post_render(now)
                self._profiler.render_latency_ms = (
                    time.perf_counter() - render_start
                ) * 1000.0
                self._profiler.capture_latency_ms = self._camera.capture_latency_ms
                self._profiler.tracking_latency_ms = self._tracking.tracking_latency_ms
                self._profiler.pose_age_ms = snapshot.age_ms()
                if stage_profile.ENABLED:
                    # Sample the EXISTING stage logger's own webcam/tracking
                    # numbers into the same history, instead of re-timing
                    # them — they're already measured on their own worker
                    # threads (src/camera/async_capture.py,
                    # src/engine/tracking_worker.py), not the main thread.
                    stage_profile.record_ms("webcam_capture",
                                           self._profiler.capture_latency_ms)
                    stage_profile.record_ms("pose_tracking",
                                           self._profiler.tracking_latency_ms)
                if stage_profile.ENABLED:
                    # "ui_painter" is draw_total MINUS the sub-stages nested
                    # inside it (bugs/layer_render/mist/fx/swap are all
                    # timed INSIDE the same _draw_frame_only call) — the
                    # remainder is the 2D HUD/screen painter itself plus
                    # V3UI.draw()'s own per-frame state-building. Peek
                    # THIS frame's totals before flushing so it lands in
                    # the same frame's history, not the next one's.
                    this_frame = stage_profile.current_frame()
                    nested = sum(this_frame.get(name, 0.0) for name in
                                ("bugs", "layer_render", "mist", "fx",
                                 "swap_vsync"))
                    stage_profile.record_ms(
                        "ui_painter", max(0.0, this_frame.get("draw_total", 0.0) - nested))
                stage_profile.end_frame()

                profile = self._profiler.end_frame(frame_start)
                quality = self._adaptive.update(profile.fps)
                self._profiler.quality_percent = quality.quality_percent
                self._tracking.set_tracking_scale(quality.tracking_scale)
                self._filters.set_measured_latency_ms(profile.compensation_ms)
            else:
                exit_reason = "window close requested"
                self._abandon_game(time.perf_counter(), "exit")
        except BaseException as exc:
            exit_reason = f"exception: {exc!r}"
            logger.exception("Render loop terminated by exception")
            raise
        finally:
            logger.info("Render loop exiting — reason: %s", exit_reason)
            self._shutdown()
        return 0

    def _build_metrics(self, pose, pitch: float, yaw: float, roll: float) -> DebugMetrics:
        profile = self._profiler.profile
        return DebugMetrics(
            fps=profile.fps,
            frame_time_ms=profile.frame_time_ms,
            capture_latency_ms=profile.capture_latency_ms,
            tracking_latency_ms=profile.tracking_latency_ms,
            render_latency_ms=profile.render_latency_ms,
            pose_age_ms=profile.pose_age_ms,
            pipeline_latency_ms=profile.pipeline_latency_ms,
            total_latency_ms=profile.total_latency_ms,
            compensation_ms=profile.compensation_ms,
            quality_percent=profile.quality_percent,
            eye_position_mm=pose.translation_mm,
            camera_position_m=self._virtual_camera.eye_position_m,
            head_pitch_deg=pitch,
            head_yaw_deg=yaw,
            head_roll_deg=roll,
            filter_type=self._filters.stage_info.smoother,
            prediction_enabled=self._filters.stage_info.prediction_enabled,
            projection_matrix=self._virtual_camera.projection_matrix(),
            face_detected=self._tracking.face_detected,
            layer_grade_lines=(
                self._renderer.layer_grade_lines if self._renderer else []
            ),
        )

    def _is_scripted_run(self) -> bool:
        """TASK_perf_textures_1119 Fix 8: true for ``--ui-shots`` or any
        bench/evidence hook env var (``HOOK_ENV_VARS``) — none of those may
        ever write the user's real ``assets/config/*.json``. Found via a
        real violation: ``_shutdown`` used to save ``user_prefs.json`` on
        EVERY exit regardless of mode, clobbering the real
        window_rect/display with whatever a `--ui-shots` run's hidden
        window happened to be.

        TASK_account_persist_1121: ``--demo`` used to count too, which made
        EVERY real game launch (``run_game.bat`` ran ``--demo --album``)
        look scripted — accounts/session/prefs were redirected to a fresh
        temp dir every time and never persisted. Demo and album are real
        interactive modes now; only the shot sequencers and hooks redirect."""
        return is_scripted_run(shots=self._album_options.shots,
                               ui_shots=self._album_options.ui_shots)

    def _shutdown(self) -> None:
        if self._album_options.is_v3 and not self._is_scripted_run():
            # D1: nhớ chế độ và khung cửa sổ cho lần chạy sau.
            from src.config import prefs as _prefs

            _prefs.save(self._root, self._window.display_mode,
                        self._window.remembered_rect())
            # TASK_accounts_1119 Phase 5: save on exit too, not just after
            # every match result — a no-op for guest/no account signed in.
            self._album.save_progress()
        self._tracking.stop()
        self._camera.stop()
        if self._renderer is not None:
            self._renderer.release()
        self._window.destroy()
        tmp_dir = getattr(self, "_scripted_run_tmp_dir", None)
        if tmp_dir is not None:
            import shutil

            shutil.rmtree(tmp_dir, ignore_errors=True)
