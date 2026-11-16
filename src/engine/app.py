"""Main HCPD application orchestrator."""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.content.formats import detect_content_type, is_supported_media
from src.content.hidden_props import layer_planes
from src.content.persistence import media_path_for_config, save_content_settings
from src.content.prop_game import (
    PROP_COUNT,
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
from src.content import shared_album
from src.content.shared_album_worker import SharedAlbumWorker, PENDING as SHARE_PENDING
from src.config.schema import AppConfig, ContentSettings, UiSettings, _merge_dataclass
from src.gui.file_dialog import pick_media_file
from src.engine.adaptive_quality import AdaptiveQualityController
from src.engine.camera_worker import CameraWorker
from src.engine.pose_bus import PoseBus
from src.engine.profiler import Profiler
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


class HcpdApplication:
    """
    Head-coupled perspective display application.

    Three threads:
    1. CameraWorker — async webcam (internal only, never displayed)
    2. TrackingWorker — pose estimation + filtering
    3. Main thread — OpenGL rendering

    The only visible window is the virtual 3D room.
    """

    #: Mặc định ở mức class để bản dựng bằng ``__new__`` (test) vẫn là sandbox.
    _demo = False
    #: TASK_album_ui_1106: cùng lý do — bản dựng bằng ``__new__`` không có
    #: giao diện album, và không được vì thế mà hỏng.
    _album_options = AlbumOptions()
    _album = None
    #: TASK_pvp_lan_1116 P2: cùng lý do — bản dựng bằng ``__new__`` vẫn phải
    #: mở webcam thật như hôm nay, trừ khi ``__init__`` nói khác.
    _no_webcam = False

    def __init__(self, config: AppConfig, project_root: Path, *,
                 demo: bool = False, album: AlbumOptions | None = None,
                 no_webcam: bool = False) -> None:
        self._config = config
        self._root = project_root
        # TASK_album_ui_1106: giao diện album. ``None`` hoặc ``enabled=False``
        # = không đổi một hành vi nào so với hôm nay.
        self._album_options = album or AlbumOptions()
        self._album = None
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
        self._v3_live = os.environ.get("HCPD_V3_LIVE") == "1"
        #: TASK_shared_album_1111: upload/list/download chạy ở luồng riêng —
        #: ``urllib.request`` có thể chặn tới 15 s (shared_album.TIMEOUT_S).
        self._share_worker = SharedAlbumWorker()
        #: "upload" hoặc "random" — ``_tick_shared_album`` đọc kết quả khác
        #: nhau (URL/lỗi so với Path/lỗi) nên phải nhớ việc nào đang chạy.
        self._share_job_kind: str | None = None

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
        đứng yên không đổi một khoá nào. Album thắng cả cờ demo: hai cờ
        thường đi cùng nhau (``run_game.bat`` gọi ``--demo --album``), và
        ván chơi luôn cần hồ sơ game dù demo có bật hay không.
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

    def _try_load_media(self, path: Path, now: float) -> None:
        """
        Load image/video via picker or drag-drop; persist on success.

        ``now`` là đầu khung bấm F6 / thả file — trước khi hộp chọn file chặn
        vòng lặp — để số giây của ván bị bỏ dở không gồm thời gian chọn file.
        """
        if not path.exists():
            self._renderer.show_toast(f"File not found: {path.name}", is_error=True)
            return
        if not is_supported_media(path):
            self._renderer.show_toast("Unsupported file type", is_error=True)
            return

        ok, msg = self._renderer.load_content_file(path)
        if not ok:
            self._renderer.show_toast(msg, is_error=True)
            return

        kind = detect_content_type(path) or "video"
        self._content.content_path = media_path_for_config(path, self._root)
        self._content.content_type = kind
        self._content.mode = "video"
        if not self._demo:                # PB-055: demo không ghi file config
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
        album.mouse = album.design_mouse(*self._window.cursor_pos())
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
        if input_state.album_centre:
            self._album_action("rest_eye", now)
        if input_state.screenshot:
            self._save_album_shot()
        if not self._v3_live:
            album.state.face_seen = bool(self._tracking.face_detected)
            album.state.tracker_simulated = not bool(
                self._tracking.tracker_available)
        if self._album_options.is_v3:
            self._tick_v3_input(input_state, now)
        album.tick()
        self._tick_album_automatch(now)
        self._tick_album_match(now)
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

    def _tick_v3_live_shots(self) -> None:
        """
        ``HCPD_V3_LIVE=1``: chụp bộ ảnh U3 bằng **đường giá trị sống**.

        Mỗi preset vẽ đủ 3 khung rồi mới đọc pixel — cùng lý do với
        ``_album_shots_step``: đọc ngay khung đầu sau khi đổi màn cho ra ảnh
        của màn trước.
        """
        from OpenGL import GL

        from src.ui_album import v3_live

        if self._window.is_iconified():
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
                self._save_album_shot(name)
        index += 1
        if index >= len(v3_live.ORDER):
            logger.info("UI v3 live: xong %d anh", len(v3_live.ORDER))
            self._window.request_close()
            return
        self._live_index = index
        self._album.game = v3_live.apply(v3_live.ORDER[index],
                                         self._album.state, self._window)

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
        path = self._root / "results" / "ui_v3_1107" / "live" / f"{stem}.png"
        logger.info(
            "UI v3 live: cua so dang o che do Window %s — chup tay bang "
            "Win+Shift+S roi luu de len %s", self._window.window_rect(), path)

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
        from src.ui_album.viewport_filter import VIEWPORT_RECT

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
        if input_state.item_slot:
            state.slot_selected = int(input_state.item_slot)
        if input_state.share_to_album and state.screen == album.flow.PHOTOS:
            self._album_action(album.activate("btn_Share to album"), now)
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

    def _photo_rgb_for_camo(self):
        """
        Ảnh gốc (RGB, CPU) để lấy màu nguỵ trang — KHÔNG đọc lại GL.

        Đọc thẳng từ đĩa đúng file ảnh đang mở; lỗi đọc (file mất, định dạng
        lạ) trả ``None`` và ``assign_bug_camo`` bị bỏ qua — con bọ khi đó vẽ
        không nguỵ trang, không phải lỗi chết ván.
        """
        from src.content.cv_io import imread_bgr

        try:
            bgr = imread_bgr(Path(self._content.content_path))
        except (OSError, ValueError):
            return None
        if bgr is None:
            return None
        return bgr[..., ::-1]

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
        from src.ui_album.game_bridge import AimContext

        mask = self._bugs.sprite_mask(self._album.state.level) \
            if self._bugs is not None else None
        if mask is None:
            return None
        eye = self._virtual_camera.eye_position_mm
        glass, centre = self._glass_mm()
        radius = float(self._content.game_prop_radius_mm)
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
        from src.ui_album.game_bridge import fire

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
        shot = fire(match, self._album.mouse, aim)
        if shot is None:
            return True          # hết băng hoặc đang nạp: nuốt cú bấm
        hit, crit = shot
        if hit >= 0:
            self._renderer.set_prop_found(hit)
            if self._bugs is not None:
                self._bugs.mark_found(
                    hit, [v / 1000.0 for v in self._game.props[hit]["pos_mm"]])
            eye = aim.eye_mm
            logger.info("Prop BAN TRUNG #%d%s — eye (%.0f, %.0f, %.0f) mm",
                        hit, " CHI MANG" if crit else "", eye[0], eye[1], eye[2])
        return True

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
            state.player = state.password = state.password2 = ""
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

    def _tick_album_automatch(self, now: float) -> None:
        """
        Móc đo ``HCPD_ALBUM_AUTOMATCH=1``: tự mở một ván khi lớp đã xong.

        Để **tự kiểm được bằng mắt** mà không phải bấm qua cả luồng — người
        dùng 07/11: "không thấy bọ ở đâu cả, không tin bạn có thể tự test".
        """
        F = self._album.flow

        if os.environ.get("HCPD_ALBUM_AUTOMATCH") not in ("1", "2"):
            return
        if self._album.game is not None or self._renderer.waiting_for_depth:
            return
        if self._album.state.screen != F.PLAY:
            return
        self._album.state.weapon = 0
        if os.environ.get("HCPD_ALBUM_AUTOMATCH") == "2":
            # Bọ đặt TAY ở chỗ biết trước: để kiểm riêng phần VẼ sprite, tách
            # khỏi phần tìm chỗ giấu (place_props có ảnh trả về 0 chỗ).
            from src.content.prop_game import GameState
            from src.ui_album.game_bridge import start_match

            radius = float(self._content.game_prop_radius_mm)
            props = [{"pos_mm": (x, 0.0, -180.0), "radius_mm": radius,
                      "reveal_axis": "x", "reveal_450_mm": 0.0,
                      "reveal_330_mm": 0.0} for x in (-70.0, 0.0, 70.0)]
            self._game = GameState(image="debug", props=props,
                                   tracking_on=True, started_at_s=now)
            self._game_planes = layer_planes(self._renderer.layer_records())
            # Không gán số hiệu bộ lớp thì _sync_prop_game bỏ ván ngay khung sau.
            self._game_layers_revision = self._renderer.layers_revision
            self._renderer.set_props([])
            if self._bugs is not None:
                self._bugs.reset()
            self._album.game = start_match(1, self._game, "Sung", [])
            logger.info("DEBUG MATCH: dat tay %d bo", len(props))
            return
        self._start_album_match(now)

    def _tick_album_match(self, now: float) -> None:
        """Nhịp của màn tách lớp và của ván đang chơi."""
        F = self._album.flow

        if self._album_options.shots or self._album_options.ui_shots:
            return          # lượt chụp: kế hoạch màn là của _album_shots_step
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
            state.go(F.WIN if ended == "win" else F.LOSE)
            self._album.game = None

    def _start_album_match(self, now: float) -> None:
        """Mở một ván theo cấp đang chọn: số bọ lấy từ bảng cấp."""
        F = self._album.flow
        from src.ui_album.game_bridge import start_match

        from src.ui_album.viewport_filter import filter_props

        state = self._album.state
        count = F.LEVEL_TABLE[state.level][1]
        # Xin NHIỀU hơn rồi lọc: màn chơi chỉ hở ô viewport, con bọ chiếu ra
        # ngoài ô thì không bao giờ thấy được nên ván không thể thắng (F3.3).
        self._start_prop_game(count=count * 3)
        if self._game is not None and self._game.props:
            rest = self._rest_eye or (
                float(self._content.overscan_design_eye_mm[0]),
                float(self._content.overscan_design_eye_mm[1]), 450.0)
            glass, centre = self._glass_mm()
            kept, outside = filter_props(self._game.props, rest, count, glass,
                                         self._viewport_rect(), centre)
            logger.info("Album bo: xin %d · dat %d · ngoai o %d",
                        count, len(kept), outside)
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

            seed_key = Path(self._content.content_path).name or "unknown"
            assign_bug_sizes(self._game.props, seed_key)
            photo_rgb = self._photo_rgb_for_camo()
            if photo_rgb is not None:
                assign_bug_camo(self._game.props, photo_rgb, rest, glass,
                               centre)
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
            from src.ui_album.game_bridge import DEFAULT_SPEC, WEAPONS_SPEC

            spec = WEAPONS_SPEC.get(weapon, DEFAULT_SPEC)
            match.ammo_enabled = True
            match.mag_size = spec.mag
            match.rounds_left = spec.mag
            match.fire_interval_s = spec.fire_s
            match.reload_s = spec.reload_s
            logger.info("Sung %s: bang %d vien, nhip %.2f s, nap %.1f s",
                        weapon, spec.mag, spec.fire_s, spec.reload_s)
        self._album.game = match
        state.go(F.PLAY)

    def _album_shots_step(self) -> None:
        """
        ``--album-shots``: mỗi màn vẽ đủ 3 khung rồi mới đọc pixel.

        Ba điều đã làm hỏng lượt chụp đầu (18/50 ảnh đen, 6 nhóm trùng md5):
        đọc ngay khung đầu sau khi đổi màn, đọc theo cỡ cửa sổ đã rơi về
        1280×720 lúc cửa sổ bị thu nhỏ, và đồng hồ mô phỏng tự nhảy màn.
        """
        from OpenGL import GL

        if self._window.is_iconified():
            logger.warning("Album shots: cua so dang thu nho — cho")
            return
        self._shot_hold = getattr(self, "_shot_hold", 0) + 1
        if self._shot_hold < 3:
            return
        self._shot_hold = 0
        index = getattr(self, "_shot_index", 0)
        plan = self._album.shot_plan()
        entry = plan[index]
        # Bộ v3 chỉ có MỘT theme, nên kế hoạch của nó là danh sách tên màn
        # phẳng; bộ cũ là cặp (màn, theme).
        stem = entry if isinstance(entry, str) else f"{entry[0]}_{entry[1]}"
        GL.glFinish()
        self._save_album_shot(stem)
        if index + 1 >= len(plan):
            logger.info("UI shots: xong %d anh", len(plan))
            self._window.request_close()
            return
        self._shot_index = index + 1
        self._album.apply_shot(index + 1)

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

    def _start_prop_game(self, count: int = PROP_COUNT) -> None:
        """
        Đặt vật ẩn từ hình học lớp THẬT của renderer, tại mắt nghỉ đo được.

        ``count`` mặc định ``PROP_COUNT`` = y như trước; giao diện album
        truyền số bọ của cấp đang chơi (TASK_album_ui_1106).
        """
        planes = layer_planes(self._renderer.layer_records())
        if len(planes) < 2:
            self._renderer.show_toast("Chua co lop — nap anh truoc", is_error=True)
            return
        radius_mm = float(self._content.game_prop_radius_mm)
        name = Path(self._content.content_path).name or "unknown"
        rest = self._rest_eye or (
            float(self._content.overscan_design_eye_mm[0]),
            float(self._content.overscan_design_eye_mm[1]), 450.0,
        )
        t0 = time.perf_counter()
        probe = os.environ.get("HCPD_PROP_PROBE", "")
        if probe == "1":
            # Chẩn đoán thứ tự vẽ (C2): bỏ qua tìm chỗ, đặt ở độ sâu biết trước.
            self._game = GameState(image=name, props=probe_props(planes),
                                   tracking_on=True, started_at_s=t0)
        else:
            self._game = start_game(
                planes, radius_mm, rest[:2], z_rest_mm(rest[2]), t0, name,
                tracking_on=not self._tracking_off, count=count,
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
        if self._album_options.is_v3:
            flow.say(self._album.state,
                     "The photo layers changed — set up the match again.")
        self._album.state.go(flow.LOADOUT)
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
                if self._demo:            # PB-055: một toast mỗi khi lớp dựng xong
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
        radius_mm = float(self._content.game_prop_radius_mm)
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
        self._tick_watch_window(now)
        if not was_done and self._game.finished_at_s is not None:
            self._write_game_csv(now, "finished")

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

                self._album = V3UI(self._root, self._shader_dir,
                                   self._album_options)
            else:
                from src.ui_album.ui import AlbumUI

                self._album = AlbumUI(self._root, self._shader_dir,
                                      self._album_options)
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
        if self._demo:
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

                if self._render_extrapolation and self._filters.stage_info.prediction_enabled:
                    pose = self._filters.extrapolate_for_display(pose, pose_age_s)

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
                if self._album is not None:
                    if not self._album.has_viewport():
                        # Màn không có ô viewport: vẫn chạy ĐÚNG đường vẽ của
                        # bản demo (render đầy đủ) nhưng cắt xuống 1 pixel nên
                        # không thấy gì và gần như không tốn gì. Gọi riêng
                        # video_plane.update() ngoài render() đã làm app văng
                        # giữa lúc dựng lớp (log 23:01, không có traceback).
                        self._album.draw_hidden_scene(scene)
                    self._album_draw_guarded(scene)
                    if self._album_options.shots or self._album_options.ui_shots:
                        self._album_shots_step()
                    elif os.environ.get("HCPD_V3_LIVE") == "1":
                        self._tick_v3_live_shots()
                else:
                    scene()
                self._window.swap_buffers()
                self._profiler.render_latency_ms = (
                    time.perf_counter() - render_start
                ) * 1000.0
                self._profiler.capture_latency_ms = self._camera.capture_latency_ms
                self._profiler.tracking_latency_ms = self._tracking.tracking_latency_ms
                self._profiler.pose_age_ms = snapshot.age_ms()

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

    def _shutdown(self) -> None:
        if self._album_options.is_v3:
            # D1: nhớ chế độ và khung cửa sổ cho lần chạy sau.
            from src.config import prefs as _prefs

            _prefs.save(self._root, self._window.display_mode,
                        self._window.remembered_rect())
        self._tracking.stop()
        self._camera.stop()
        if self._renderer is not None:
            self._renderer.release()
        self._window.destroy()
