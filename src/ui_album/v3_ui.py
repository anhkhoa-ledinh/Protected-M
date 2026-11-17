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

from src.config import prefs as _prefs
from src.fx.library import FxLibrary
from src.fx.player import FxPlayer
from src.fx.spec import load_spec
from src.ui_album import v3_routes, v3_values
from src.ui_album import v3_flow as F
from src.ui_album.v3 import V3Manifest, V3Painter, draw_screen, hit_test, shot_state

logger = logging.getLogger(__name__)


class V3UI:
    """Giao diện v3 gắn vào một cửa sổ GLFW đã có context."""

    def __init__(self, root: Path, shader_dir: Path, options) -> None:
        self.manifest = V3Manifest(root)
        self.painter = V3Painter(self.manifest, shader_dir)
        self.state = F.V3Flow()
        self.options = options
        self.game = None
        self.mouse = (0.0, 0.0)
        self.pending = ""
        #: Khung webcam đã lên GPU: ô ``camera`` của màn cài đặt / thử nghiêng.
        self._camera_entry = None
        self._camera_token = None
        if options.guest:
            self.state.player = "guest"
            self.state.go(F.MENU)
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

    def initialize(self) -> None:
        self.painter.initialize()
        if self._fx_enabled and not self._fx_library.check_capability():
            self._fx_enabled = False

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
        action = v3_routes.on_click(self.state, self.hover(),
                                    time.perf_counter(), self.mouse)
        self.pending = action
        return action

    def activate(self, name: str) -> str:
        """Kích hoạt một hotspot CỤ THỂ (phím tắt U/R…), bỏ qua vị trí chuột."""
        action = v3_routes.on_click(self.state, name, time.perf_counter(),
                                    self.mouse)
        self.pending = action
        return action

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
        return v3_routes.on_click(self.state, name, time.perf_counter(),
                                  self.mouse)

    def back(self) -> str:
        """Esc = lùi một bước."""
        return v3_routes.on_back(self.state, time.perf_counter())

    def move_focus(self, delta: int) -> None:
        """↑/↓ trên màn chính: dịch vòng focus giữa bốn nút."""
        F.move_menu_focus(self.state, delta)

    # ---- nhịp ------------------------------------------------------------
    def tick(self) -> None:
        # Khi chụp, KHÔNG chạy đồng hồ mô phỏng: nó đẩy "đang tìm trận" sang
        # "đã tìm thấy" ngay, nên hai ảnh ra y hệt nhau (bẫy của lượt album).
        if self.options.ui_shots:
            return
        F.update(self.state, time.perf_counter())
        if self._fx_enabled:
            now = time.perf_counter()
            dt = 0.0 if self._fx_last_t is None else max(0.0, now - self._fx_last_t)
            self._fx_last_t = now
            self._fx_player.step(dt)

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
                                        self.game, self.mouse)
            state.hover = self.hover()
            state.camera_tex = self._camera_entry
            if scene_draw is not None:
                state.viewport_draw = lambda rect: self._scene(rect, scene_draw)
        if self._fx_enabled:
            if name != self._fx_screen:
                self._fx_library.enter_screen(name, self._fx_player)
                self._fx_screen = name
            self._fx_library.pump()
            state.fx_on = True
            state.fx_draw = lambda: self._fx_library.draw(self.painter, self._fx_player)
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

    @property
    def fx_enabled(self) -> bool:
        return self._fx_enabled

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
