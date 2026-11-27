"""GLFW window and OpenGL context management."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import glfw
import numpy as np
from OpenGL import GL

from src.config.prefs import ASPECT, MIN_WINDOW, default_rect
from src.config.schema import RendererSettings

logger = logging.getLogger(__name__)


def viewport_size(
    framebuffer_size: tuple[int, int],
    fallback: tuple[int, int],
) -> tuple[int, int]:
    """
    Kích thước thật để vẽ: framebuffer nếu dùng được, nếu không thì ``fallback``.

    Đây từng là chỗ hỏng. ``width``/``height`` trả thẳng
    ``settings.window_width/height`` — tức hằng số trong ``default.json`` —
    nên ``glViewport`` ở ``renderer.py`` chạy theo con số cấu hình chứ không
    theo vùng vẽ thật. Đo 2026-10-11: client area 1304x744 nhưng cảnh vẽ
    1280x720 neo góc dưới-trái, chừa dải đen 24 px ở trên và bên phải.

    ``fallback`` dùng cho trường hợp GLFW trả ``(0, 0)`` — xảy ra khi cửa sổ bị
    thu nhỏ xuống taskbar. Lấy 0 làm bề rộng viewport thì chia cho 0 ở mọi
    overlay, nên giữ lại số cũ cho tới khi cửa sổ hiện lại.
    """
    width, height = framebuffer_size
    if width > 0 and height > 0:
        return int(width), int(height)
    return int(fallback[0]), int(fallback[1])


def letterbox_rect(
    framebuffer_size: tuple[int, int],
    aspect: tuple[int, int] = ASPECT,
) -> tuple[int, int, int, int]:
    """
    TASK_game_feel_fix_1118 Fix 9: the largest ``aspect``-ratio box that
    fits centred inside ``framebuffer_size``, as ``(x, y, w, h)`` pixels
    (origin at the bottom-left, matching GL viewport convention).

    Pure/no-GL so the resize-drag blit path's geometry is unit-testable —
    ``(0, 0, 0, 0)`` for a degenerate (zero-area) framebuffer, same "don't
    divide by zero" posture as :func:`viewport_size` above.
    """
    width, height = int(framebuffer_size[0]), int(framebuffer_size[1])
    if width <= 0 or height <= 0:
        return (0, 0, 0, 0)
    aw, ah = aspect
    box_w, box_h = width, round(width * ah / aw)
    if box_h > height:
        box_h = height
        box_w = round(height * aw / ah)
    box_w, box_h = max(1, int(box_w)), max(1, int(box_h))
    return ((width - box_w) // 2, (height - box_h) // 2, box_w, box_h)


def fullscreen_target(
    enabled: bool,
    monitor,
    video_mode,
    fallback_size: tuple[int, int],
) -> tuple[object | None, tuple[int, int], dict[int, int]]:
    """
    Chọn màn hình, kích thước và window hint cho lần tạo cửa sổ.

    Trả ``(monitor, (width, height), hints)``. ``monitor`` là ``None`` khi chạy
    cửa sổ — đúng thứ ``glfw.create_window`` cần ở đối số thứ tư.

    Khi bật toàn màn hình, dùng **đúng chế độ hình đang chạy** của màn chính:
    đặt ``RED/GREEN/BLUE_BITS`` và ``REFRESH_RATE`` khớp ``video_mode`` rồi tạo
    cửa sổ đúng kích thước đó. Đây là công thức "borderless fullscreen" của
    GLFW — vì chế độ yêu cầu trùng chế độ hiện tại nên trình điều khiển **không
    đổi độ phân giải**, chỉ bỏ viền và phủ kín cả taskbar.

    ``enabled`` nhưng không có màn hoặc không đọc được ``video_mode`` thì rơi
    về cửa sổ với ``fallback_size``; caller có trách nhiệm log cảnh báo.

    ĐỪNG tắt ``AUTO_ICONIFY`` (PB-051, đã thử 16/10): run13 tắt nó thì cửa sổ
    mất focus vẫn nằm trên cùng, phím đi vào cửa sổ phía sau, Alt-Tab chỉ đổi
    các cửa sổ phía sau — không phím nào ăn, kể cả sau khi có lớp. Để mặc định
    thì mất focus là thu nhỏ, Alt-Tab về được (run12).
    """
    if not enabled or monitor is None or video_mode is None:
        return None, (int(fallback_size[0]), int(fallback_size[1])), {}

    hints = {
        glfw.RED_BITS: int(video_mode.bits.red),
        glfw.GREEN_BITS: int(video_mode.bits.green),
        glfw.BLUE_BITS: int(video_mode.bits.blue),
        glfw.REFRESH_RATE: int(video_mode.refresh_rate),
    }
    size = (int(video_mode.size.width), int(video_mode.size.height))
    return monitor, size, hints


@dataclass(frozen=True)
class InputState:
    """Per-frame keyboard input snapshot."""

    move_x: float = 0.0
    move_y: float = 0.0
    move_z: float = 0.0
    toggle_debug: bool = False
    cycle_filter: bool = False
    toggle_prediction: bool = False
    toggle_webcam_preview: bool = False
    open_content_file: bool = False
    toggle_atmospheric: bool = False
    toggle_mist: bool = False
    toggle_relief: bool = False
    toggle_disocclusion: bool = False
    zoom_in: bool = False
    zoom_out: bool = False
    #: Virtual room, ``--demo`` only: B toggles.
    room_toggle: bool = False
    # Quét chiều sâu (Phase 2). plane_z là cần gạt CHÍNH, z_far là phụ —
    # xem ContentPlaneRenderer.nudge_plane_z để biết vì sao.
    plane_closer: bool = False
    plane_farther: bool = False
    far_layer_closer: bool = False
    far_layer_farther: bool = False
    overscan_down: bool = False
    overscan_up: bool = False
    # Phím sandbox của trò tìm vật ẩn (PB-015 C2): G mở/đặt lại ván,
    # T tắt tracking để có điều kiện đối chứng.
    game_toggle: bool = False
    toggle_tracking: bool = False
    # TASK_album_ui_1106: F12 chụp màn, C căn lại theo chỗ ngồi.
    screenshot: bool = False
    album_centre: bool = False
    # Ô nhập của màn đăng nhập: xoá lùi, sang ô kế, xác nhận.
    backspace: bool = False
    focus_next: bool = False
    submit: bool = False
    quit_app: bool = False
    # TASK_ui_v3_1107. Esc ở giao diện v3 là "lùi một bước", nên nó phải là
    # **một nhịp bấm** chứ không phải trạng thái phím như ``quit_app``: giữ
    # Esc mà mỗi khung lùi một màn thì một cú bấm nhảy về tận màn đăng nhập.
    escape: bool = False
    display_toggle: bool = False
    focus_up: bool = False
    focus_down: bool = False
    replay: bool = False
    #: Phím 1–3 chọn ô vật phẩm trên HUD; 0 = không bấm gì.
    item_slot: int = 0
    #: TASK_shared_album_1111: U = "Share to album" ở màn chọn ảnh.
    share_to_album: bool = False
    #: TASK_hud_fix_1119 Fix 2: H = ẩn/hiện HUD (không lưu, chỉ trong phiên).
    hud_hide_toggle: bool = False
    #: TASK_accounts_1119: Ctrl+V dán vào ô đang chọn (kể cả ô mật khẩu).
    paste_requested: bool = False
    #: TASK_small_fixes_1120: di chuyển/xoá con trỏ trong ô nhập.
    caret_left: bool = False
    caret_right: bool = False
    caret_home: bool = False
    caret_end: bool = False
    delete_forward: bool = False


#: Phím có tác dụng ở chế độ demo (PB-055; quyết định 11/10: demo không có phím
#: A/B và F1). Esc và kéo-thả đi đường riêng nên vẫn chạy; WASD/QE bị tắt.
_DEMO_BASE_KEYS = frozenset({glfw.KEY_G, glfw.KEY_T, glfw.KEY_F6, glfw.KEY_F4})

#: Virtual room keys: only ``--demo`` (not ``--album``) has them.
ROOM_KEYS = frozenset({glfw.KEY_B})
DEMO_KEYS = _DEMO_BASE_KEYS | ROOM_KEYS

#: TASK_album_ui_1106: giao diện album dùng chuột là chính; bàn phím thêm F12
#: (chụp màn), F9 (diễn cảnh mất kết nối) và C (căn lại theo chỗ ngồi).
ALBUM_KEYS = _DEMO_BASE_KEYS | frozenset({glfw.KEY_F9, glfw.KEY_F12, glfw.KEY_C,
                                    glfw.KEY_BACKSPACE, glfw.KEY_ENTER,
                                    glfw.KEY_KP_ENTER, glfw.KEY_TAB})

#: TASK_ui_v3_1107: giao diện v3 thêm F11 (đổi chế độ hiển thị), Esc (lùi một
#: bước), ↑/↓ (vòng focus của màn chính), R (chơi lại) và 1–3 (ô vật phẩm).
#:
#: Phím giao diện v3 nhận — khai THẲNG, không trừ dần từ ``ALBUM_KEYS``.
#:
#: Cách cũ (``ALBUM_KEYS - {T, F4, G}``) đã sót: ``F6`` (mở file) và ``F9``
#: (bật/tắt lấp khe) vẫn lọt vào vì chúng nằm sẵn trong ``ALBUM_KEYS``, và
#: người dùng gặp lại chúng trong lúc chơi. Danh sách cho phép thì đọc một
#: cái là biết v3 ăn đúng những phím nào, và thêm phím vào bộ cũ sau này
#: không âm thầm mở cửa cho v3 nữa.
#:
#: Mọi đồ nghề sandbox của lõi HCPD đều KHÔNG có mặt: ``F1``–``F3`` (HUD gỡ
#: lỗi, đổi bộ lọc, dự đoán), ``F4`` (ô webcam PiP), ``F6`` (mở file — v3 đã
#: có nút "Choose from computer"), ``F7``–``F10`` (A/B sương mù, khí quyển,
#: lấp khe, nổi khối), ``G`` (đo mắt nghỉ + rải lại vật ẩn — v3 dùng nút
#: "Recalibrate" và phím ``C``), ``T`` (tắt tracking), và các phím nhích
#: ``+ - [ ] , . ; '``. Đây là bản cho người chơi, không phải bàn thí nghiệm.
V3_KEYS = frozenset({
    glfw.KEY_F11,                      # đổi toàn màn hình ↔ cửa sổ
    glfw.KEY_F12,                      # chụp màn
    glfw.KEY_ESCAPE,                   # lùi một bước
    glfw.KEY_ENTER, glfw.KEY_KP_ENTER,  # nút chính của màn
    glfw.KEY_TAB, glfw.KEY_BACKSPACE,  # ô nhập
    glfw.KEY_C,                        # căn lại chỗ ngồi
    glfw.KEY_UP, glfw.KEY_DOWN,        # vòng focus của màn chính
    glfw.KEY_R,                        # chơi lại ở màn kết quả
    glfw.KEY_1, glfw.KEY_2, glfw.KEY_3,  # ô vật phẩm
    glfw.KEY_U,                        # TASK_shared_album_1111: Share to album
    glfw.KEY_H,                        # TASK_game_feel_fix_1118 Fix 10.4: HUD gọn
    glfw.KEY_V,                        # TASK_accounts_1119: Ctrl+V dan vao o nhap
    # TASK_small_fixes_1120: di chuyen/xoa con tro trong o nhap.
    glfw.KEY_LEFT, glfw.KEY_RIGHT, glfw.KEY_HOME, glfw.KEY_END,
    glfw.KEY_DELETE,
})


#: TASK_perf_textures_1119 Fix 8: a second PRESS callback invocation
#: within this many seconds of the first is treated as the SAME physical
#: click, not a second one — see ``GlfwWindow._on_mouse_button``'s
#: docstring for why this exists.
DUPLICATE_PRESS_GUARD_S = 0.05


class GlfwWindow:
    """Application window with modern OpenGL context."""

    #: Mặc định ở mức class để bản dựng bằng ``__new__`` (test) vẫn là sandbox.
    _demo = False
    _album = False
    _ui_v3 = False

    def __init__(self, settings: RendererSettings, *, demo: bool = False,
                 album: bool = False, ui_v3: bool = False,
                 prefs: dict | None = None) -> None:
        self._settings = settings
        self._demo = bool(demo)
        self._album = bool(album)
        #: TASK_ui_v3_1107: giao diện v3 nhận thêm phím, và Esc đổi nghĩa.
        self._ui_v3 = bool(ui_v3)
        prefs = prefs or {}
        #: D1: "fullscreen" | "window", nhớ qua ``assets/config/user_prefs.json``.
        self._display = str(prefs.get("display") or "fullscreen")
        self._window_rect = (list(prefs.get("window_rect"))
                             if prefs.get("window_rect") else None)
        #: D2: ai cần biết khi cửa sổ dời chỗ / đổi cỡ (mặt phẳng Kooima).
        self._geometry_listener = None
        #: TASK_game_feel_1118 P1: ai vẽ lại NGAY khi GLFW báo cửa sổ cần
        #: vẽ lại (đang kéo cạnh — modal loop, main loop không chạy tới).
        self._refresh_handler = None
        self._logged_dpi = False
        self._window = None
        #: TASK_game_feel_fix_1118 Fix 9: True trong lúc ``glfw.poll_events()``
        #: đang chạy — khi kéo cạnh cửa sổ (Windows modal loop) đây có thể
        #: kéo dài suốt cả lượt kéo, với callback vẽ-lại nổ ra NHIỀU lần
        #: trước khi ``poll_events()`` trả về. Chỉ để đo/log; đường vẽ-lại
        #: (``draw_last_frame_letterboxed``) không rẽ nhánh theo cờ này —
        #: nó rẻ đủ để luôn an toàn, bất kể cờ.
        self._poll_events_active = False
        #: Fix 9: bản chụp khung vừa vẽ xong, để vẽ-lại RẺ (chỉ blit, không
        #: vẽ lại toàn cảnh) trong lúc kéo cạnh. Cấp lại khi cỡ đổi.
        self._last_frame_fbo = None
        self._last_frame_tex = None
        self._last_frame_alloc_size: tuple[int, int] | None = None
        self._last_frame_valid_size: tuple[int, int] | None = None
        #: Fix 9 (post-commit correction): the MSAA-safe textured-quad
        #: draw used by ``draw_last_frame_letterboxed`` instead of a
        #: (illegal, into a multisampled FBO 0) blit — lazily built on
        #: first use by ``_ensure_blit_shader``.
        self._blit_program = None
        self._blit_vao = None
        self._blit_vbo = None
        #: TASK_ui_sakuga_1117 Phase 5: con trỏ OS theo hình — tạo một lần
        #: mỗi hình, nhớ lại tránh gọi ``glfw.set_cursor`` lại mỗi khung khi
        #: hình không đổi.
        self._cursors: dict = {}
        self._cursor_shape: str | None = None
        # Kích thước vùng vẽ thật. Khởi tạo bằng số cấu hình vì cửa sổ chưa tồn
        # tại; create() thay ngay bằng số GLFW báo về.
        self._fb_size = (settings.window_width, settings.window_height)
        self._input = InputState()
        self._debug_requested = False
        self._filter_cycle_requested = False
        self._prediction_toggle_requested = False
        self._preview_toggle_requested = False
        self._open_file_requested = False
        self._atmospheric_toggle_requested = False
        self._mist_toggle_requested = False
        self._relief_toggle_requested = False
        self._disocclusion_toggle_requested = False
        self._zoom_in_requested = False
        self._zoom_out_requested = False
        self._room_toggle_requested = False
        self._plane_closer_requested = False
        self._plane_farther_requested = False
        self._far_closer_requested = False
        self._far_farther_requested = False
        self._overscan_down_requested = False
        self._overscan_up_requested = False
        self._game_toggle_requested = False
        self._tracking_toggle_requested = False
        self._screenshot_requested = False
        self._album_centre_requested = False
        self._backspace_requested = False
        self._focus_next_requested = False
        self._submit_requested = False
        self._escape_requested = False
        self._display_toggle_requested = False
        self._focus_up_requested = False
        self._focus_down_requested = False
        self._replay_requested = False
        self._item_slot_requested = 0
        self._share_to_album_requested = False
        self._hud_hide_requested = False
        self._paste_requested = False
        self._caret_left_requested = False
        self._caret_right_requested = False
        self._caret_home_requested = False
        self._caret_end_requested = False
        self._delete_requested = False
        self._dropped_paths: list[str] = []
        #: TASK_album_ui_1106: chuột và chữ, chỉ giao diện album đọc tới.
        self._clicks: list[tuple[float, float]] = []
        self._typed = ""
        #: TASK_perf_textures_1119 Fix 8: mốc PRESS gần nhất, cho
        #: ``_on_mouse_button``'s duplicate-press guard.
        self._last_press_t: float | None = None

    @property
    def width(self) -> int:
        return self._fb_size[0]

    @property
    def height(self) -> int:
        return self._fb_size[1]

    def _set_framebuffer_size(self, width: int, height: int) -> tuple[int, int]:
        """Ghi lại kích thước vùng vẽ. Tách riêng để test được, không đụng GL."""
        self._fb_size = viewport_size(
            (width, height),
            (self._settings.window_width, self._settings.window_height),
        )
        return self._fb_size

    def _on_framebuffer_size(self, _window, width: int, height: int) -> None:
        """Callback của GLFW khi vùng vẽ đổi kích thước."""
        vw, vh = self._set_framebuffer_size(width, height)
        # renderer.render() cũng gọi glViewport mỗi khung theo width/height nên
        # dòng này là thừa về lâu dài; giữ để khung đang vẽ dở lúc resize không
        # dùng viewport cũ.
        GL.glViewport(0, 0, vw, vh)
        logger.info("Framebuffer resized to %dx%d — viewport %dx%d", width, height, vw, vh)
        # TASK_game_feel_1118 P1: hình học Kooima (D2) phải theo kịp TRƯỚC
        # khi vẽ lại — gọi ở đây phòng khi GLFW báo framebuffer-size trước
        # window-size (``_on_window_size`` tự nó cũng gọi, idempotent).
        self._notify_geometry()
        if self._refresh_handler is not None:
            self._refresh_handler()

    @property
    def should_close(self) -> bool:
        return bool(glfw.window_should_close(self._window))

    def create(self) -> None:
        """Initialize GLFW and create the OpenGL context."""
        if not glfw.init():
            raise RuntimeError("Failed to initialize GLFW")

        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.SAMPLES, self._settings.msaa_samples)

        # D1: ở giao diện v3, chế độ hiển thị do ``user_prefs.json`` quyết —
        # người dùng chọn lần trước thì lần này mở lại đúng thế. Giao diện cũ
        # giữ nguyên ``renderer.fullscreen`` của config.
        wanted = ((self._display == "fullscreen") if self._ui_v3
                  else self._settings.fullscreen)
        monitor = glfw.get_primary_monitor() if wanted else None
        video_mode = glfw.get_video_mode(monitor) if monitor is not None else None
        monitor, (win_w, win_h), hints = fullscreen_target(
            wanted,
            monitor,
            video_mode,
            (self._settings.window_width, self._settings.window_height),
        )
        if wanted and monitor is None:
            logger.warning(
                "fullscreen requested but no usable primary monitor — "
                "falling back to a %dx%d window",
                win_w, win_h,
            )
        for hint, value in hints.items():
            glfw.window_hint(hint, value)
        logger.info(
            "Window mode: %s %dx%d",
            "FULLSCREEN (primary monitor, current video mode)" if monitor else "windowed",
            win_w, win_h,
        )

        self._window = glfw.create_window(
            win_w,
            win_h,
            self._settings.window_title,
            monitor,
            None,
        )
        if self._window is None:
            glfw.terminate()
            raise RuntimeError("Failed to create GLFW window")

        glfw.make_context_current(self._window)
        glfw.swap_interval(1 if self._settings.vsync else 0)
        glfw.set_key_callback(self._window, self._on_key)
        glfw.set_drop_callback(self._window, self._on_drop)
        glfw.set_framebuffer_size_callback(self._window, self._on_framebuffer_size)
        if self._ui_v3:
            self._display = "fullscreen" if monitor else "window"
            if monitor is None:
                self._apply_window_style()
                # D1.4: mở lại đúng khung cửa sổ lần trước. Không khôi phục
                # thì cửa sổ nhảy về chỗ hệ điều hành chọn, và người dùng
                # phải kê lại mỗi lần chạy.
                mon_pos, mon_size, _hz = self.monitor_geometry()
                rect = self._window_rect or default_rect(mon_pos, mon_size)
                glfw.set_window_size(self._window, int(rect[2]), int(rect[3]))
                glfw.set_window_pos(self._window, int(rect[0]), int(rect[1]))
                self._window_rect = list(rect)
            # D2: mặt phẳng Kooima là hình chữ nhật VẬT LÝ của cửa sổ, nên nó
            # phải đổi theo cả chỗ đứng lẫn kích thước.
            glfw.set_window_pos_callback(self._window, self._on_window_pos)
            glfw.set_window_size_callback(self._window, self._on_window_size)
        # TASK_album_ui_1106: chuột và chữ gõ vào. Hai callback này chỉ GOM sự
        # kiện; không ai đọc thì không đổi hành vi nào của bản hôm nay.
        glfw.set_mouse_button_callback(self._window, self._on_mouse_button)
        glfw.set_char_callback(self._window, self._on_char)

        requested = (win_w, win_h)
        fb = glfw.get_framebuffer_size(self._window)
        vw, vh = self._set_framebuffer_size(fb[0], fb[1])
        # Cặp số này là bằng chứng của PB-038: trước bản sửa, viewport luôn lấy
        # `requested` bất kể framebuffer thật rộng bao nhiêu.
        logger.info(
            "Framebuffer %dx%d — viewport %dx%d (requested %dx%d)",
            fb[0], fb[1], vw, vh, requested[0], requested[1],
        )

        GL.glEnable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_CULL_FACE)
        GL.glCullFace(GL.GL_BACK)
        if self._settings.msaa_samples > 0:
            GL.glEnable(GL.GL_MULTISAMPLE)

    @property
    def poll_events_active(self) -> bool:
        """TASK_game_feel_fix_1118 Fix 9: True while a ``glfw.poll_events()``
        call from this window is on the stack — covers both a long modal
        resize drag and a short programmatic one (``set_display_mode``/
        ``resize_window``). Diagnostic only: the refresh handler's blit
        path is cheap and correct either way, so nothing branches on this
        — it exists so resize profiling can tell "happened while blocked
        in poll_events" apart from any other refresh."""
        return self._poll_events_active

    def _poll_events(self) -> None:
        self._poll_events_active = True
        try:
            glfw.poll_events()
        finally:
            self._poll_events_active = False

    def pump_events(self) -> None:
        """
        Lấy message Windows giữa các bước khởi động (PB-051), chưa đọc phím.

        Từ ``create()`` tới khung đầu là dựng renderer, nạp ảnh, mở tracking.
        Luồng này quá ~5 s không lấy message thì dwm thay cửa sổ bằng bản ma
        "Not Responding" và app mất focus (run12, 16/10). Cờ phím callback đặt
        ở đây vẫn được ``poll_input`` đọc ở khung đầu.
        """
        self._poll_events()

    def poll_input(self) -> InputState:
        """Poll events and return movement input for dev/testing."""
        self._poll_events()
        move_x = move_y = move_z = 0.0

        # Demo (PB-055): không dịch pose bằng tay — bỏ qua WASD/QE.
        # TASK_account_persist_1121: album cũng bỏ qua, dù demo có bật hay
        # không — ván chơi không được để WASD/QE giả lập đầu dịch chỗ.
        if not self._demo and not self._album:
            if glfw.get_key(self._window, glfw.KEY_A) == glfw.PRESS:
                move_x -= 1.0
            if glfw.get_key(self._window, glfw.KEY_D) == glfw.PRESS:
                move_x += 1.0
            if glfw.get_key(self._window, glfw.KEY_W) == glfw.PRESS:
                move_y += 1.0
            if glfw.get_key(self._window, glfw.KEY_S) == glfw.PRESS:
                move_y -= 1.0
            if glfw.get_key(self._window, glfw.KEY_Q) == glfw.PRESS:
                move_z += 1.0
            if glfw.get_key(self._window, glfw.KEY_E) == glfw.PRESS:
                move_z -= 1.0

        result = InputState(
            move_x=move_x,
            move_y=move_y,
            move_z=move_z,
            toggle_debug=self._debug_requested,
            cycle_filter=self._filter_cycle_requested,
            toggle_prediction=self._prediction_toggle_requested,
            toggle_webcam_preview=self._preview_toggle_requested,
            open_content_file=self._open_file_requested,
            toggle_atmospheric=self._atmospheric_toggle_requested,
            toggle_mist=self._mist_toggle_requested,
            toggle_relief=self._relief_toggle_requested,
            toggle_disocclusion=self._disocclusion_toggle_requested,
            zoom_in=self._zoom_in_requested,
            zoom_out=self._zoom_out_requested,
            room_toggle=self._room_toggle_requested,
            plane_closer=self._plane_closer_requested,
            plane_farther=self._plane_farther_requested,
            far_layer_closer=self._far_closer_requested,
            far_layer_farther=self._far_farther_requested,
            overscan_down=self._overscan_down_requested,
            overscan_up=self._overscan_up_requested,
            game_toggle=self._game_toggle_requested,
            toggle_tracking=self._tracking_toggle_requested,
            screenshot=self._screenshot_requested,
            album_centre=self._album_centre_requested,
            backspace=self._backspace_requested,
            focus_next=self._focus_next_requested,
            submit=self._submit_requested,
            escape=self._escape_requested,
            display_toggle=self._display_toggle_requested,
            focus_up=self._focus_up_requested,
            focus_down=self._focus_down_requested,
            replay=self._replay_requested,
            item_slot=self._item_slot_requested,
            share_to_album=self._share_to_album_requested,
            hud_hide_toggle=self._hud_hide_requested,
            paste_requested=self._paste_requested,
            caret_left=self._caret_left_requested,
            caret_right=self._caret_right_requested,
            caret_home=self._caret_home_requested,
            caret_end=self._caret_end_requested,
            delete_forward=self._delete_requested,
            # Giao diện v3: Esc là "lùi một bước", và app tự quyết khi nào cú
            # lùi ấy nghĩa là thoát (``v3_routes.on_back``).
            quit_app=(False if self._ui_v3 else
                      glfw.get_key(self._window, glfw.KEY_ESCAPE) == glfw.PRESS),
        )
        self._debug_requested = False
        self._filter_cycle_requested = False
        self._prediction_toggle_requested = False
        self._preview_toggle_requested = False
        self._open_file_requested = False
        self._atmospheric_toggle_requested = False
        self._mist_toggle_requested = False
        self._relief_toggle_requested = False
        self._disocclusion_toggle_requested = False
        self._zoom_in_requested = False
        self._zoom_out_requested = False
        self._room_toggle_requested = False
        self._plane_closer_requested = False
        self._plane_farther_requested = False
        self._far_closer_requested = False
        self._far_farther_requested = False
        self._overscan_down_requested = False
        self._overscan_up_requested = False
        self._game_toggle_requested = False
        self._tracking_toggle_requested = False
        self._screenshot_requested = False
        self._album_centre_requested = False
        self._backspace_requested = False
        self._focus_next_requested = False
        self._submit_requested = False
        self._escape_requested = False
        self._display_toggle_requested = False
        self._focus_up_requested = False
        self._focus_down_requested = False
        self._replay_requested = False
        self._item_slot_requested = 0
        self._share_to_album_requested = False
        self._hud_hide_requested = False
        self._paste_requested = False
        self._caret_left_requested = False
        self._caret_right_requested = False
        self._caret_home_requested = False
        self._caret_end_requested = False
        self._delete_requested = False
        return result

    # ---- D1: chế độ hiển thị -------------------------------------------
    @property
    def display_mode(self) -> str:
        return self._display

    def _apply_window_style(self) -> None:
        """
        Cửa sổ thường: có viền, kéo được, khoá 16:9, không nhỏ hơn 960×540.

        Khoá tỉ lệ ở tầng GLFW chứ không tự sửa lại cỡ trong callback: trình
        quản lý cửa sổ của Windows kẹp ngay lúc người dùng kéo, nên không có
        khung nào vẽ ở tỉ lệ sai rồi nhảy về.
        """
        glfw.set_window_attrib(self._window, glfw.DECORATED, glfw.TRUE)
        glfw.set_window_attrib(self._window, glfw.RESIZABLE, glfw.TRUE)
        glfw.set_window_aspect_ratio(self._window, ASPECT[0], ASPECT[1])
        glfw.set_window_size_limits(self._window, MIN_WINDOW[0], MIN_WINDOW[1],
                                    glfw.DONT_CARE, glfw.DONT_CARE)

    def monitor_geometry(self, monitor=None):
        """``((mx, my), (W, H), refresh)`` của một màn hình."""
        monitor = monitor or glfw.get_primary_monitor()
        if monitor is None:
            return ((0, 0), (self.width, self.height), glfw.DONT_CARE)
        mode = glfw.get_video_mode(monitor)
        pos = glfw.get_monitor_pos(monitor)
        if mode is None:
            return ((int(pos[0]), int(pos[1])), (self.width, self.height),
                    glfw.DONT_CARE)
        return ((int(pos[0]), int(pos[1])),
                (int(mode.size.width), int(mode.size.height)),
                int(mode.refresh_rate))

    def window_rect(self) -> list[int]:
        """``[x, y, w, h]`` vùng khách của cửa sổ, hỏi GLFW ngay lúc này."""
        if self._window is None:
            return [0, 0, self.width, self.height]
        pos = glfw.get_window_pos(self._window)
        size = glfw.get_window_size(self._window)
        return [int(pos[0]), int(pos[1]), int(size[0]), int(size[1])]

    def remembered_rect(self) -> list[int] | None:
        """Khung cửa sổ để ghi vào prefs: khung hiện tại nếu đang ở chế độ cửa sổ."""
        if self._display == "window" and self._window is not None:
            return self.window_rect()
        return self._window_rect

    def set_display_mode(self, mode: str) -> bool:
        """
        Đổi chế độ hiển thị **ngay trong lúc chạy**, không dựng lại context.

        ``glfw.set_window_monitor`` giữ nguyên cửa sổ và context, nên mọi
        texture đã nạp vẫn còn — chỉ framebuffer đổi cỡ, và callback
        ``framebuffer_size`` lo phần đó. Trả ``True`` nếu đã đổi.
        """
        mode = "fullscreen" if str(mode) == "fullscreen" else "window"
        if self._window is None or mode == self._display:
            return False
        # Fix 9: flag covers set_window_monitor too, not just the explicit
        # poll_events() below — GLFW can deliver the framebuffer-size
        # callback synchronously from inside set_window_monitor itself.
        self._poll_events_active = True
        try:
            if mode == "window":
                rect = self._window_rect
                if not rect:
                    pos, size, _refresh = self.monitor_geometry()
                    rect = default_rect(pos, size)
                self._apply_window_style()
                glfw.set_window_monitor(self._window, None, int(rect[0]),
                                        int(rect[1]), int(rect[2]), int(rect[3]),
                                        glfw.DONT_CARE)
            else:
                # Ghi lại khung ĐANG dùng trước khi phủ kín màn: không nhớ thì lần
                # quay về cửa sổ sẽ nhảy ra giữa màn, mất chỗ người dùng đã kê.
                self._window_rect = self.window_rect()
                monitor = glfw.get_primary_monitor()
                pos, size, refresh = self.monitor_geometry(monitor)
                glfw.set_window_monitor(self._window, monitor, int(pos[0]),
                                        int(pos[1]), int(size[0]), int(size[1]),
                                        refresh)
            self._display = mode
            glfw.swap_interval(1 if self._settings.vsync else 0)
            glfw.poll_events()
        finally:
            self._poll_events_active = False
        fb = glfw.get_framebuffer_size(self._window)
        self._set_framebuffer_size(fb[0], fb[1])
        logger.info("Display mode: %s — cua so %s, framebuffer %dx%d",
                    mode, self.window_rect(), fb[0], fb[1])
        self._notify_geometry()
        return True

    def resize_window(self, width: int, height: int) -> None:
        """Đặt cỡ vùng khách rồi canh giữa màn (chỉ ở chế độ cửa sổ)."""
        if self._window is None or self._display != "window":
            return
        pos, size, _refresh = self.monitor_geometry()
        width = min(int(width), int(size[0]))
        height = min(int(height), int(size[1]))
        self._poll_events_active = True
        try:
            glfw.set_window_size(self._window, width, height)
            glfw.set_window_pos(self._window, pos[0] + (size[0] - width) // 2,
                                pos[1] + (size[1] - height) // 2)
            glfw.poll_events()
        finally:
            self._poll_events_active = False

    def toggle_display(self) -> str:
        """F11: đổi qua lại hai chế độ. Trả chế độ sau khi đổi."""
        self.set_display_mode("window" if self._display == "fullscreen"
                              else "fullscreen")
        return self._display

    def set_clipboard(self, text: str) -> None:
        if self._window is not None:
            glfw.set_clipboard_string(self._window, str(text))

    def get_clipboard(self) -> str:
        """TASK_accounts_1119: Ctrl+V paste source — GLFW returns bytes
        (not text), and a non-text clipboard (an image, say) must not
        crash a paste, just yield nothing."""
        if self._window is None:
            return ""
        try:
            raw = glfw.get_clipboard_string(self._window)
        except Exception:                                    # noqa: BLE001
            return ""
        if raw is None:
            return ""
        return raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)

    def is_mouse_down(self) -> bool:
        """Chuột trái đang giữ — màn cài đặt cần để KÉO thanh âm lượng."""
        if self._window is None:
            return False
        return glfw.get_mouse_button(self._window,
                                     glfw.MOUSE_BUTTON_LEFT) == glfw.PRESS

    def is_right_mouse_down(self) -> bool:
        """TASK_game_feel_1118 P6: chuột phải đang giữ — bật ống ngắm của
        Precision gun (chỉ khẩu đó)."""
        if self._window is None:
            return False
        return glfw.get_mouse_button(self._window,
                                     glfw.MOUSE_BUTTON_RIGHT) == glfw.PRESS

    def set_cursor_pos(self, x: float, y: float) -> None:
        """TASK_game_feel_1118 P6: hạ ống ngắm xong thì dời con trỏ HĐH về
        đúng tâm ngắm (toạ độ PIXEL CỬA SỔ — gọi qua phép ánh xạ của
        painter, không phải toạ độ thiết kế) để tâm ngắm không nhảy."""
        if self._window is not None:
            glfw.set_cursor_pos(self._window, float(x), float(y))

    #: Tên hình (``v3_hover.cursor_for``) -> hằng GLFW tương ứng.
    _CURSOR_GLFW = {"hand": "HAND_CURSOR", "ibeam": "IBEAM_CURSOR",
                    "arrow": "ARROW_CURSOR"}

    def set_cursor_shape(self, kind: str) -> None:
        """Đổi hình con trỏ OS — "arrow"/"hand"/"ibeam"
        (TASK_ui_sakuga_1117 Phase 5). Không làm gì nếu hình không đổi."""
        if self._window is None or kind == self._cursor_shape:
            return
        cursor = self._cursors.get(kind)
        if cursor is None:
            constant = getattr(glfw, self._CURSOR_GLFW.get(kind, "ARROW_CURSOR"))
            cursor = glfw.create_standard_cursor(constant)
            self._cursors[kind] = cursor
        glfw.set_cursor(self._window, cursor)
        self._cursor_shape = kind

    # ---- P1: vẽ lại khi cửa sổ cần (đang kéo cạnh) -----------------------
    def set_refresh_handler(self, fn) -> None:
        """``fn()`` vẽ lại cửa sổ NGAY LẬP TỨC — gọi từ GLFW
        (``glfw.set_window_refresh_callback``, báo khi cửa sổ cần vẽ lại —
        kéo cạnh chẳng hạn) và từ cuối ``_on_framebuffer_size`` (sau khi
        viewport + hình học Kooima D2 đã cập nhật), để giao diện bắt kịp
        kích thước thật trong suốt lúc kéo, không chỉ sau khi thả chuột."""
        self._refresh_handler = fn
        if self._window is not None:
            glfw.set_window_refresh_callback(self._window, lambda _w: fn())

    # ---- D2: hình chữ nhật vật lý của cửa sổ ---------------------------
    def set_geometry_listener(self, listener) -> None:
        """Gọi ``listener()`` mỗi khi cửa sổ dời chỗ hoặc đổi cỡ."""
        self._geometry_listener = listener
        self._notify_geometry()

    def _notify_geometry(self) -> None:
        if self._geometry_listener is not None:
            self._geometry_listener()

    def _on_window_pos(self, _window, x: int, y: int) -> None:
        _ = (x, y)
        self._notify_geometry()

    def _on_window_size(self, _window, width: int, height: int) -> None:
        _ = (width, height)
        self._notify_geometry()

    def framebuffer_ratio(self) -> tuple[float, float]:
        """
        ``framebuffer / window`` — hệ số DPI scaling, thường là 1,0.

        Khác 1 thì mọi phép tính tính bằng pixel cửa sổ phải nhân lại; in
        **một lần** để lượt chạy tự nói nó đang ở hệ số nào (D2).
        """
        if self._window is None:
            return (1.0, 1.0)
        win = glfw.get_window_size(self._window)
        fb = glfw.get_framebuffer_size(self._window)
        ratio = (fb[0] / max(1, win[0]), fb[1] / max(1, win[1]))
        if not self._logged_dpi:
            self._logged_dpi = True
            logger.info("Framebuffer %dx%d vs window %dx%d — ty le %.3f x %.3f",
                        fb[0], fb[1], win[0], win[1], ratio[0], ratio[1])
        return ratio

    def window_centre_monitor(self):
        """Màn hình chứa **tâm** cửa sổ, và nó có phải màn chính không."""
        if self._window is None:
            return (None, True)
        rect = self.window_rect()
        centre = (rect[0] + rect[2] * 0.5, rect[1] + rect[3] * 0.5)
        primary = glfw.get_primary_monitor()
        for monitor in glfw.get_monitors() or ():
            pos, size, _refresh = self.monitor_geometry(monitor)
            if (pos[0] <= centre[0] < pos[0] + size[0]
                    and pos[1] <= centre[1] < pos[1] + size[1]):
                return (monitor, monitor == primary)
        return (primary, True)

    def cursor_pos(self) -> tuple[float, float]:
        """Vị trí con trỏ trong pixel cửa sổ (TASK_album_ui_1106)."""
        if self._window is None:
            return (0.0, 0.0)
        x, y = glfw.get_cursor_pos(self._window)
        return (float(x), float(y))

    def pop_clicks(self) -> list:
        """Các cú bấm chuột trái kể từ lần hỏi trước."""
        clicks, self._clicks = self._clicks, []
        return clicks

    def pop_typed(self) -> str:
        """Chuỗi ký tự đã gõ kể từ lần hỏi trước."""
        typed, self._typed = self._typed, ""
        return typed

    def _on_mouse_button(self, window, button, action, mods) -> None:
        """GLFW's left-button PRESS callback → one queued click.

        TASK_perf_textures_1119 Fix 8: the user's log showed every shot
        ("Ban hut tai ...") followed AT ONCE by a second, rejected
        ``fire()`` call ("Bam co dan: chua toi nhip") for 7 of 8 shots —
        one physical click producing two queued entries in ``_clicks``
        (this is the ONLY place anything appends to it, and
        ``_album_shoot`` is called exactly once per popped click, so a
        double queue IS a double fire). No human can release and press
        again within ``DUPLICATE_PRESS_GUARD_S`` (50 ms); a PRESS that
        close behind the last one is the same physical click delivered
        twice, not a second one.
        """
        _ = (window, mods)
        if button != glfw.MOUSE_BUTTON_LEFT or action != glfw.PRESS:
            return
        now = time.perf_counter()
        if (self._last_press_t is not None
                and now - self._last_press_t < DUPLICATE_PRESS_GUARD_S):
            return
        self._last_press_t = now
        self._clicks.append(self.cursor_pos())

    def _on_char(self, window, codepoint) -> None:
        _ = window
        self._typed += chr(int(codepoint))

    def pop_dropped_path(self) -> str | None:
        """Return the next drag-and-dropped file path, if any."""
        if not self._dropped_paths:
            return None
        return self._dropped_paths.pop(0)

    def framebuffer_size(self) -> tuple[int, int]:
        """Cỡ framebuffer **hỏi GLFW ngay lúc này**, không qua bộ nhớ đệm.

        ``width``/``height`` rơi về số trong config khi cửa sổ bị thu nhỏ
        (``viewport_size``), nên đọc pixel theo chúng sẽ ra khung 1280×720 đen
        — đúng 18 ảnh hỏng của lượt chụp đầu.
        """
        if self._window is None:
            return (0, 0)
        return tuple(int(v) for v in glfw.get_framebuffer_size(self._window))

    def is_iconified(self) -> bool:
        """Cửa sổ đang thu nhỏ: đọc framebuffer lúc này chỉ ra màu đen."""
        if self._window is None:
            return False
        return bool(glfw.get_window_attrib(self._window, glfw.ICONIFIED))

    def restore_and_focus(self) -> None:
        """
        Kéo cửa sổ trở lại sau khi một hộp thoại khác chiếm màn hình.

        Hộp thoại chọn file (tkinter) làm cửa sổ toàn màn hình bị **thu nhỏ**:
        log 07/11 00:54 có đúng dòng ``Framebuffer resized to 0x0`` ngay lúc
        mở hộp thoại, và sau đó Alt-Tab cũng không đưa cửa sổ về được (bẫy
        AUTO_ICONIFY đã ghi ở ``fullscreen_target``). Không gọi hàm này thì
        app vẫn chạy nhưng người dùng chỉ thấy desktop.
        """
        if self._window is None:
            return
        if glfw.get_window_attrib(self._window, glfw.ICONIFIED):
            glfw.restore_window(self._window)
        glfw.show_window(self._window)
        glfw.focus_window(self._window)

    def request_close(self) -> None:
        """Yêu cầu đóng cửa sổ (--ui-shots dùng khi chụp xong)."""
        if self._window is not None:
            glfw.set_window_should_close(self._window, True)

    def swap_buffers(self) -> None:
        """Present the back buffer."""
        glfw.swap_buffers(self._window)

    # ---- Fix 9: last-frame cache for cheap resize-drag redraws -----------
    #: TASK_game_feel_fix_1118 Fix 9 (post-commit correction): the default
    #: framebuffer is MULTISAMPLED whenever ``msaa_samples > 0`` (the
    #: project default is 4) — ``glBlitFramebuffer`` into a multisampled
    #: DRAW framebuffer is illegal regardless of rect sizes (GL spec:
    #: "If SAMPLE_BUFFERS for the draw framebuffer is greater than zero,
    #: an INVALID_OPERATION error is generated"), and reproduced for real
    #: the first time this ran against a live window (GLError 1282 from
    #: glBlitFramebuffer, caught while gathering Fix 9's own evidence).
    #: ``capture_last_frame``'s blit is the OTHER direction (multisampled
    #: READ -> single-sample draw, at identical size) and stays legal;
    #: only the letterboxed draw-back-into-FBO-0 half needs to stop using
    #: a blit and rasterize a plain textured quad instead — always legal,
    #: MSAA or not.
    _BLIT_VERTEX_SRC = """#version 330 core
    layout (location = 0) in vec2 aPos;
    layout (location = 1) in vec2 aUV;
    out vec2 vUV;
    void main() {
        gl_Position = vec4(aPos, 0.0, 1.0);
        vUV = aUV;
    }
    """
    _BLIT_FRAGMENT_SRC = """#version 330 core
    in vec2 vUV;
    out vec4 fragColor;
    uniform sampler2D uTex;
    void main() {
        fragColor = texture(uTex, vUV);
    }
    """

    def _ensure_blit_shader(self) -> None:
        if self._blit_program is not None:
            return

        def compile_one(source: str, kind) -> int:
            shader = GL.glCreateShader(kind)
            GL.glShaderSource(shader, source)
            GL.glCompileShader(shader)
            if GL.glGetShaderiv(shader, GL.GL_COMPILE_STATUS) != GL.GL_TRUE:
                info = GL.glGetShaderInfoLog(shader).decode("utf-8")
                raise RuntimeError(f"Window blit shader compile failed: {info}")
            return shader

        vertex = compile_one(self._BLIT_VERTEX_SRC, GL.GL_VERTEX_SHADER)
        fragment = compile_one(self._BLIT_FRAGMENT_SRC, GL.GL_FRAGMENT_SHADER)
        program = GL.glCreateProgram()
        GL.glAttachShader(program, vertex)
        GL.glAttachShader(program, fragment)
        GL.glLinkProgram(program)
        GL.glDeleteShader(vertex)
        GL.glDeleteShader(fragment)
        if GL.glGetProgramiv(program, GL.GL_LINK_STATUS) != GL.GL_TRUE:
            info = GL.glGetProgramInfoLog(program).decode("utf-8")
            raise RuntimeError(f"Window blit shader link failed: {info}")
        self._blit_program = program
        self._blit_vao = GL.glGenVertexArrays(1)
        self._blit_vbo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._blit_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._blit_vbo)
        # 6 verts (2 tris) x 4 floats (pos.xy, uv.xy) — filled in per draw.
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 6 * 4 * 4, None, GL.GL_DYNAMIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16,
                                 GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)

    def _draw_textured_rect(self, x: int, y: int, w: int, h: int,
                            fb_width: int, fb_height: int) -> None:
        """Rasterizes ``_last_frame_tex`` into the CURRENTLY BOUND
        framebuffer at pixel rect ``(x, y, w, h)`` (bottom-left origin,
        matching ``letterbox_rect``) — the MSAA-safe replacement for a
        blit into FBO 0."""
        self._ensure_blit_shader()
        x0, x1 = 2.0 * x / fb_width - 1.0, 2.0 * (x + w) / fb_width - 1.0
        y0, y1 = 2.0 * y / fb_height - 1.0, 2.0 * (y + h) / fb_height - 1.0
        data = np.array([x0, y0, 0.0, 0.0, x1, y0, 1.0, 0.0, x1, y1, 1.0, 1.0,
                         x0, y0, 0.0, 0.0, x1, y1, 1.0, 1.0, x0, y1, 0.0, 1.0],
                        dtype=np.float32)
        GL.glUseProgram(self._blit_program)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._last_frame_tex)
        GL.glUniform1i(GL.glGetUniformLocation(self._blit_program, "uTex"), 0)
        GL.glBindVertexArray(self._blit_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._blit_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, data.nbytes, data)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 6)
        GL.glBindVertexArray(0)
        GL.glUseProgram(0)

    def _release_blit_shader(self) -> None:
        if self._blit_program is not None:
            GL.glDeleteProgram(self._blit_program)
            self._blit_program = None
        if self._blit_vao is not None:
            GL.glDeleteVertexArrays(1, [self._blit_vao])
            self._blit_vao = None
        if self._blit_vbo is not None:
            GL.glDeleteBuffers(1, [self._blit_vbo])
            self._blit_vbo = None

    def _ensure_last_frame_storage(self, width: int, height: int) -> None:
        if self._last_frame_alloc_size == (width, height) and self._last_frame_fbo is not None:
            return
        self._release_last_frame_storage()
        self._last_frame_fbo = GL.glGenFramebuffers(1)
        self._last_frame_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._last_frame_tex)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, width, height, 0,
                        GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glBindTexture(GL.GL_TEXTURE_2D, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self._last_frame_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0,
                                  GL.GL_TEXTURE_2D, self._last_frame_tex, 0)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self._last_frame_alloc_size = (width, height)

    def _release_last_frame_storage(self) -> None:
        if self._last_frame_fbo is not None:
            GL.glDeleteFramebuffers(1, [self._last_frame_fbo])
            self._last_frame_fbo = None
        if self._last_frame_tex is not None:
            GL.glDeleteTextures(1, [self._last_frame_tex])
            self._last_frame_tex = None
        self._last_frame_alloc_size = None
        self._last_frame_valid_size = None

    def capture_last_frame(self) -> None:
        """Call once, right before ``swap_buffers()``, after every NORMAL
        (full-render) frame — copies the just-drawn back buffer into an
        FBO-backed texture, reallocated only when the size actually
        changes. ``draw_last_frame_letterboxed`` blits this back out
        during a resize drag instead of re-running the full draw."""
        width, height = self._fb_size
        if width <= 0 or height <= 0:
            return
        self._ensure_last_frame_storage(width, height)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, 0)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._last_frame_fbo)
        GL.glBlitFramebuffer(0, 0, width, height, 0, 0, width, height,
                             GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
        self._last_frame_valid_size = (width, height)

    #: TASK_perf_textures_1119 Fix 9: GL enable-bits ``draw_last_frame_letterboxed``
    #: must own, save, disable and restore — none of these are reset
    #: between a normal frame and a resize mid-drag, and this path draws
    #: with NONE of the state the normal frame path set up for itself.
    _RESIZE_REDRAW_GL_STATES = (GL.GL_DEPTH_TEST, GL.GL_CULL_FACE,
                                GL.GL_SCISSOR_TEST, GL.GL_STENCIL_TEST,
                                GL.GL_BLEND)

    def draw_last_frame_letterboxed(self, bg_colour: tuple = (0.05, 0.06, 0.08)) -> bool:
        """The entire resize-drag redraw: clear to ``bg_colour``, draw the
        cached last frame letterboxed (16:9) into the CURRENT framebuffer
        size, swap. No scene/UI redraw — this is what keeps a drag smooth.
        Returns ``False`` (no-op) if nothing has been captured yet, e.g. a
        resize before the very first frame.

        A textured-quad draw, not a blit: FBO 0 is multisampled whenever
        ``msaa_samples > 0`` (the project default), and
        ``glBlitFramebuffer`` into a multisampled draw framebuffer is
        illegal regardless of rect sizes — see ``_draw_textured_rect``'s
        docstring.

        TASK_perf_textures_1119 Fix 9: the window showed only the flat
        clear colour during a drag — ``window.py`` enables
        ``GL_DEPTH_TEST``/``GL_CULL_FACE`` globally (normal-frame setup),
        and this quad draws at z=0 against a depth buffer GLFW just
        reallocated at the NEW size for the resize (undefined/uncleared
        contents) — the clear ran, but the quad silently failed the
        depth test (or got scissored/blended/stencilled out by whatever
        the interrupted normal frame had set). Saves, disables, draws,
        restores — every GL enable this function doesn't own itself is
        now exactly as the caller left it, same as before this call."""
        if self._last_frame_fbo is None or self._last_frame_valid_size is None:
            return False
        width, height = self._fb_size
        if width <= 0 or height <= 0:
            return False
        was_enabled = {state: GL.glIsEnabled(state)
                      for state in self._RESIZE_REDRAW_GL_STATES}
        for state in self._RESIZE_REDRAW_GL_STATES:
            GL.glDisable(state)
        try:
            GL.glColorMask(True, True, True, True)
            GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, 0)
            GL.glViewport(0, 0, width, height)
            GL.glClearColor(bg_colour[0], bg_colour[1], bg_colour[2], 1.0)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
            dst_x, dst_y, dst_w, dst_h = letterbox_rect((width, height))
            self._draw_textured_rect(dst_x, dst_y, dst_w, dst_h, width, height)
        finally:
            for state, enabled in was_enabled.items():
                (GL.glEnable if enabled else GL.glDisable)(state)
        self.swap_buffers()
        return True

    def destroy(self) -> None:
        """Destroy window and terminate GLFW."""
        self._release_blit_shader()
        self._release_last_frame_storage()
        if self._window is not None:
            glfw.destroy_window(self._window)
            self._window = None
        glfw.terminate()

    def _on_key(self, window, key, scancode, action, mods) -> None:
        _ = (window, scancode)
        if action != glfw.PRESS:
            return
        if self._ui_v3 and key not in V3_KEYS:
            return                       # v3: chuột là chính
        if self._album and not self._ui_v3 and key not in ALBUM_KEYS:
            return                       # album: chuột là chính
        if self._demo and not self._album and key not in DEMO_KEYS:
            return                       # PB-055: demo không có phím A/B, F1
        if key == glfw.KEY_F1:
            self._debug_requested = True
        elif key == glfw.KEY_F2:
            self._filter_cycle_requested = True
        elif key == glfw.KEY_F3:
            self._prediction_toggle_requested = True
        elif key == glfw.KEY_F4:
            self._preview_toggle_requested = True
        elif key == glfw.KEY_F6:
            self._open_file_requested = True
        elif key == glfw.KEY_F7:
            self._mist_toggle_requested = True
        elif key == glfw.KEY_F8:
            self._atmospheric_toggle_requested = True
        elif key == glfw.KEY_F9:
            self._disocclusion_toggle_requested = True
        elif key == glfw.KEY_F10:
            self._relief_toggle_requested = True
        elif self._demo and not self._album and key in ROOM_KEYS:
            # Virtual room (demo only).
            self._room_toggle_requested = True
        elif key == glfw.KEY_G:
            self._game_toggle_requested = True
        elif key == glfw.KEY_T:
            self._tracking_toggle_requested = True
        elif key in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            self._zoom_in_requested = True
        elif key in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            self._zoom_out_requested = True
        elif key == glfw.KEY_LEFT_BRACKET:
            self._plane_closer_requested = True
        elif key == glfw.KEY_RIGHT_BRACKET:
            self._plane_farther_requested = True
        elif key == glfw.KEY_COMMA:
            self._far_closer_requested = True
        elif key == glfw.KEY_PERIOD:
            self._far_farther_requested = True
        elif key == glfw.KEY_SEMICOLON:
            self._overscan_down_requested = True
        elif key == glfw.KEY_APOSTROPHE:
            self._overscan_up_requested = True
        elif key == glfw.KEY_F12:
            self._screenshot_requested = True
        elif key == glfw.KEY_C:
            self._album_centre_requested = True
        elif key == glfw.KEY_BACKSPACE:
            self._backspace_requested = True
        elif key == glfw.KEY_TAB:
            self._focus_next_requested = True
        elif key in (glfw.KEY_ENTER, glfw.KEY_KP_ENTER):
            self._submit_requested = True
        elif key == glfw.KEY_ESCAPE:
            self._escape_requested = True
        elif key == glfw.KEY_F11:
            self._display_toggle_requested = True
        elif key == glfw.KEY_UP:
            self._focus_up_requested = True
        elif key == glfw.KEY_DOWN:
            self._focus_down_requested = True
        elif key == glfw.KEY_R:
            self._replay_requested = True
        elif key in (glfw.KEY_1, glfw.KEY_2, glfw.KEY_3):
            self._item_slot_requested = key - glfw.KEY_1 + 1
        elif key == glfw.KEY_U:
            self._share_to_album_requested = True
        elif key == glfw.KEY_H:
            self._hud_hide_requested = True
        elif key == glfw.KEY_V and (mods & glfw.MOD_CONTROL):
            self._paste_requested = True
        elif key == glfw.KEY_LEFT:
            self._caret_left_requested = True
        elif key == glfw.KEY_RIGHT:
            self._caret_right_requested = True
        elif key == glfw.KEY_HOME:
            self._caret_home_requested = True
        elif key == glfw.KEY_END:
            self._caret_end_requested = True
        elif key == glfw.KEY_DELETE:
            self._delete_requested = True

    def _on_drop(self, window, paths) -> None:
        _ = window
        for raw in paths:
            if isinstance(raw, bytes):
                self._dropped_paths.append(raw.decode("utf-8", errors="replace"))
            else:
                self._dropped_paths.append(str(raw))
