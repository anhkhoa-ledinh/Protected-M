"""
Cầu nối giữa app và giao diện album: một đối tượng, ba việc.

``handle_input`` (chuột/phím) → ``tick`` (nhịp mô phỏng) → ``draw`` (vẽ màn
đang mở). App chỉ cần gọi ba hàm đó; mọi thứ khác nằm trong gói này.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path

from OpenGL import GL

from src.ui_album import flow as F
from src.ui_album import routes, values
from src.ui_album.draw import AlbumPainter
from src.ui_album.manifest import AlbumManifest
from src.ui_album.screen import ScreenState, draw_screen, hit_test

logger = logging.getLogger(__name__)


class AlbumUI:
    """Giao diện album gắn vào một cửa sổ GLFW đã có context."""

    def __init__(self, root: Path, shader_dir: Path, options) -> None:
        self.manifest = AlbumManifest(root)
        self.painter = AlbumPainter(self.manifest, shader_dir)
        self.state = F.AlbumFlow()
        self.options = options
        self.game = None
        self.scene_draw = None
        self.mouse = (0.0, 0.0)
        self.pending = ""
        if options.guest:
            self.state.player = "khach"
            self.state.go(F.COVER)
        # Móc đo: mở thẳng một màn để đếm FPS trên đúng màn đó (vd. màn chơi,
        # nơi có cả cảnh 3D). Không đặt biến thì không đổi gì.
        forced = os.environ.get("HCPD_ALBUM_SCREEN", "").strip()
        if forced and forced in self.manifest.screens:
            self.state.go(forced)

    def initialize(self) -> None:
        self.painter.initialize()

    #: Module trạng thái của giao diện này. ``app.py`` đọc hằng tên màn qua
    #: đây, nên một bộ hàm phục vụ được cả giao diện album và giao diện v3
    #: (TASK_ui_v3_1107 U0: "delete no old code").
    flow = F

    @property
    def bug_theme(self) -> str:
        return self.state.theme

    # ---- nhập liệu -------------------------------------------------------
    def design_mouse(self, x: float, y: float) -> tuple[float, float]:
        return self.painter.box.unmap(x, y)

    def hover(self) -> str | None:
        return hit_test(self.manifest, self.state.screen, self.state,
                        *self.mouse)

    def click(self) -> str:
        """Bấm chuột trái tại vị trí hiện tại; trả hành động cho app."""
        action = routes.on_click(self.state, self.hover(), time.perf_counter())
        self.pending = action
        return action

    def key(self, char: str) -> None:
        """Một ký tự gõ vào ô đang chọn."""
        F.type_char(self.state, char)

    def backspace(self) -> None:
        """Xoá lùi một ký tự ở ô đang chọn."""
        F.backspace(self.state)

    def focus_next(self) -> None:
        """Tab: sang ô nhập kế tiếp của màn đang mở."""
        fields = {h["n"] for h in
                  self.manifest.screens[self.state.screen].get("hotspots", [])}
        F.focus_next(self.state, fields)

    def submit(self) -> str:
        """Enter = bấm nút chính của màn đang mở."""
        buttons = [h["n"] for h in
                   self.manifest.screens[self.state.screen].get("hotspots", [])
                   if h["n"].startswith("nut_")]
        if not buttons:
            return ""
        return routes.on_click(self.state, buttons[0], time.perf_counter())

    # ---- nhịp ------------------------------------------------------------
    def tick(self) -> None:
        # Khi chụp, KHÔNG chạy đồng hồ mô phỏng: nó đẩy màn "đang tìm trận"
        # sang "đã tìm thấy" ngay lập tức, nên hai ảnh ra y hệt nhau.
        if self.options.shots:
            return
        F.update(self.state, time.perf_counter())

    # ---- vẽ --------------------------------------------------------------
    def draw(self, window_size, scene_draw=None) -> None:
        """Vẽ màn đang mở. ``scene_draw(rect_px)`` vẽ cảnh 3D vào ô viewport."""
        name = self.state.screen
        design = self.manifest.size_of(name)
        self.painter.set_frame(window_size, design)
        state = ScreenState(
            theme=self.state.theme, photo=self.state.photo,
            values=self._values(),
            hidden=values.hidden_for(self.state, self.game),
            selected=self.state.selected_hotspots(),
            hover=None if self.options.shots else self.hover(),
            disabled=self.state.disabled_hotspots(),
            crosshair=self.mouse if name in (F.PLAY, F.PVP_HUNT) else None,
            shot=self._shot(),
            message=(self.state.message
                     if time.perf_counter() < self.state.message_until_s
                     else ""))
        if scene_draw is not None:
            state.viewport_draw = lambda rect: self._scene(rect, scene_draw)
        self.painter.begin()
        draw_screen(self.painter, self.manifest, name, state)
        self.painter.end()

    def _shot(self):
        """``(x, y, tuổi, trúng?)`` của phát bắn gần nhất; ``None`` nếu chưa bắn."""
        match = self.game
        if match is None or not match.shot_at_s:
            return None
        hit = match.missed_until_s < time.perf_counter()
        return (match.shot_point[0], match.shot_point[1],
                time.perf_counter() - match.shot_at_s, hit)

    def _values(self) -> dict:
        """Giá trị ô động; lượt chụp ảnh thêm một ván giả (xem SHOT_MATCH)."""
        live = values.values_for(self.state, self.game)
        if self.options.shots:
            live.update(values.shot_values(self.state))
        return live

    def _scene(self, rect, scene_draw) -> None:
        """
        Cảnh 3D chỉ hiện **trong ô viewport**, bằng ``glScissor``.

        Phép chiếu Kooima giữ nguyên cho cả cửa sổ (đúng vật lý), ô viewport
        là một **khung cắt** của khung nhìn thật — không co giãn gì.
        """
        self.painter.end()
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(*self.painter.viewport_px(rect))
        scene_draw()
        GL.glDisable(GL.GL_SCISSOR_TEST)
        self.painter.begin()

    def draw_hidden_scene(self, scene_draw) -> None:
        """
        Chạy trọn ``renderer.render`` nhưng cắt còn **1 pixel**.

        Vì sao không gọi thẳng ``video_plane.update``: nhịp tách lớp phải đi
        đúng đường mà bản demo đã chạy hàng chục lượt. Gọi tắt một nhánh của
        nó thì bước dựng lớp chạy ở trạng thái GL khác, và app đã văng ngay
        giữa lúc dựng lớp (23:01, không có traceback Python).
        """
        GL.glEnable(GL.GL_SCISSOR_TEST)
        GL.glScissor(0, 0, 1, 1)
        try:
            scene_draw()
        finally:
            GL.glDisable(GL.GL_SCISSOR_TEST)

    def has_viewport(self) -> bool:
        """Màn đang mở có ô cảnh 3D không (chỉ màn chơi thì có)."""
        screen = self.manifest.screens[self.state.screen]
        return any(photo.get("kind") == "viewport"
                   for photo in screen.get("photos", []))

    # ---- chụp cả bộ màn (P4) --------------------------------------------
    def shot_plan(self) -> list:
        """Danh sách ``(màn, theme)`` cho ``--album-shots``: 25 màn × 2 theme."""
        return [(name, theme)
                for name in self.manifest.screen_names()
                for theme in ("ngay", "dem")]

    def apply_shot(self, index: int) -> tuple[str, str] | None:
        """Đặt màn/theme thứ ``index`` của kế hoạch; ``None`` khi hết."""
        plan = self.shot_plan()
        if index >= len(plan):
            return None
        name, theme = plan[index]
        self.state.screen, self.state.theme = name, theme
        return name, theme

    def release(self) -> None:
        self.painter.release()
