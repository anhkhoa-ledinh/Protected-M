r"""
PB-074 Phase 0b — dựng cảnh **qua đúng đường GL của app**, cửa sổ ẩn.

Không giả lập OpenGL như ``measure_props.py``: ở đây mở `GlfwWindow` thật
(chỉ thêm hint ``VISIBLE = False``), dựng `SceneRenderer` thật, chờ tách lớp
xong, đặt mắt, vẽ, rồi đọc khung bằng ``glReadPixels``. Vậy mọi thứ ảnh hưởng
tới ảnh — overscan, relief, feather, viền sáng, lấp khe, sương — đều là code
thật của app.
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import glfw
import numpy as np
from OpenGL import GL

from src.config.schema import AppConfig
from src.renderer.camera import VirtualCamera
from src.renderer.debug_overlay import DebugMetrics
from src.renderer.renderer import SceneRenderer
from src.renderer.window import GlfwWindow
from src.scene.video_scene import build_video_window_scene

ROOT = Path(__file__).resolve().parents[1]
READY_TIMEOUT_S = 240.0


def _metrics(camera, eye_mm):
    """``DebugMetrics`` tối thiểu; HUD tắt nên chỉ vài trường được dùng."""
    return DebugMetrics(
        fps=0.0, frame_time_ms=0.0, eye_position_mm=np.asarray(eye_mm, float),
        camera_position_m=camera.eye_position_m, head_pitch_deg=0.0,
        head_yaw_deg=0.0, head_roll_deg=0.0, filter_type="none",
        prediction_enabled=False, projection_matrix=camera.projection_matrix())


class OffscreenApp:
    """Một phiên GL ẩn cho MỘT ảnh; đóng xong mới mở ảnh khác."""

    def __init__(self, config: AppConfig, content, screen_cal, root=ROOT):
        self._content = content
        self._root = root
        renderer_cfg = replace(config.renderer, fullscreen=False, vsync=False)
        glfw.init()
        glfw.window_hint(glfw.VISIBLE, False)   # phải đặt SAU glfw.init()
        self._window = GlfwWindow(renderer_cfg)
        self._window.create()
        self._camera = VirtualCamera(
            screen_width_m=screen_cal["screen_width"],
            screen_height_m=screen_cal["screen_height"],
            screen_center_m=screen_cal["screen_center"], cache_projection=False)
        self._renderer = SceneRenderer(
            self._window, root / config.renderer.shader_path,
            build_video_window_scene(content), self._camera,
            content=content, project_root=root)
        self._renderer.initialize()
        self._window.pump_events()

    @property
    def size(self):
        return self._window.width, self._window.height

    def draw(self, eye_mm):
        """Vẽ một khung ở vị trí mắt ``eye_mm`` và trả ảnh RGB (uint8)."""
        self._camera.set_eye_mm(np.asarray(eye_mm, dtype=np.float64))
        self._renderer.render(_metrics(self._camera, eye_mm), dt_s=1.0 / 60.0)
        width, height = self.size
        # Đọc BACK buffer TRƯỚC khi swap: cửa sổ ẩn nên front buffer không được
        # bảo đảm giữ gì cả.
        GL.glFinish()
        GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
        GL.glReadBuffer(GL.GL_BACK)
        raw = GL.glReadPixels(0, 0, width, height, GL.GL_RGB,
                              GL.GL_UNSIGNED_BYTE)
        self._window.swap_buffers()
        self._window.pump_events()
        img = np.frombuffer(raw, dtype=np.uint8).reshape(height, width, 3)
        return np.ascontiguousarray(img[::-1])      # GL gốc ở góc dưới-trái

    def wait_ready(self, eye_mm, timeout_s=READY_TIMEOUT_S):
        """
        Vẽ liên tục tới khi bộ lớp đã dựng xong; trả số giây đã chờ.

        Depth chạy ở luồng riêng (``DepthWorker``) và lớp chỉ xuất hiện khi
        ``update()`` lấy được kết quả, nên phải vẽ thật chứ không ngủ suông.
        """
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < timeout_s:
            self.draw(eye_mm)
            if len(self._renderer.layer_records()) > 0:
                return time.perf_counter() - t0
            time.sleep(0.05)
        return None

    def layer_count(self):
        return len(self._renderer.layer_records())

    def close(self):
        self._renderer.release()
        self._window.destroy()
        glfw.terminate()
