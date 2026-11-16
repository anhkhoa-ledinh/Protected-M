"""Debug overlay (F1 toggle) with ImGui or GPU texture fallback."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from OpenGL import GL

from src.renderer.shader import ShaderProgram

#: Hai tam giác của quad, cùng thứ tự với ``toast_overlay`` và ``webcam_preview``.
HUD_QUAD_INDICES = np.array([0, 1, 2, 0, 2, 3], dtype=np.uint32)

#: Lề của HUD so với mép trên-trái của viewport, tính bằng pixel.
HUD_MARGIN_PX = 8.0

#: Giãn cách giữa hai lần in khối HUD ra console, giây. [GIẢ ĐỊNH]
HUD_CONSOLE_INTERVAL_S = 0.5


def hud_quad_vertices(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """
    Bốn đỉnh quad HUD trong NDC, mỗi đỉnh bốn số ``x, y, u, v``.

    ``x0 < x1`` và ``y0 < y1``; ``y0`` là cạnh **dưới**.

    Thứ tự là dưới-trái, dưới-phải, trên-phải, trên-trái — **ngược chiều kim
    đồng hồ**, giống hệt ``toast_overlay`` và ``webcam_preview``. Bắt buộc phải
    vậy: ``window.py`` bật ``GL_CULL_FACE`` với ``glCullFace(GL_BACK)`` cho cả
    phiên và ``_draw_fallback`` không tắt culling, nên quad quấn theo chiều kim
    đồng hồ bị loại trước khi raster hoá — mất sạch, không lỗi GL nào. Trước
    2026-10-11 nó quấn CW và HUD F1 chưa bao giờ hiện.

    V đảo ngược: ``glTexImage2D`` coi hàng 0 của mảng — đỉnh canvas chữ — là
    ``v = 0``, nên cạnh trên của quad phải nhận ``v = 0`` thì chữ mới đứng.

    Xem ``tests/test_debug_overlay.py``.
    """
    return np.array(
        [
            x0, y0, 0.0, 1.0,
            x1, y0, 1.0, 1.0,
            x1, y1, 1.0, 0.0,
            x0, y1, 0.0, 0.0,
        ],
        dtype=np.float32,
    )


@dataclass
class DebugMetrics:
    """Runtime metrics shown in debug overlay."""

    fps: float
    frame_time_ms: float
    eye_position_mm: np.ndarray
    camera_position_m: np.ndarray
    head_pitch_deg: float
    head_yaw_deg: float
    head_roll_deg: float
    filter_type: str
    prediction_enabled: bool
    projection_matrix: np.ndarray
    capture_latency_ms: float = 0.0
    tracking_latency_ms: float = 0.0
    render_latency_ms: float = 0.0
    pose_age_ms: float = 0.0
    pipeline_latency_ms: float = 0.0
    total_latency_ms: float = 0.0
    compensation_ms: float = 0.0
    quality_percent: int = 100
    face_detected: bool = False
    layer_grade_lines: list[str] = field(default_factory=list)


class DebugOverlay:
    """Debug HUD — ImGui when available, otherwise textured quad HUD."""

    def __init__(self, window, shader_dir: Path) -> None:
        self._window = window
        self._visible = False
        self._imgui_ready = False
        self._impl = None
        self._renderer = None
        self._hud_shader: ShaderProgram | None = None
        self._hud_vao = 0
        self._hud_vbo = 0
        self._texture = 0
        self._fallback_ready = False
        self._shader_dir = shader_dir
        self._last_echo_at = 0.0

    @property
    def visible(self) -> bool:
        return self._visible

    def toggle(self) -> None:
        """Show or hide debug overlay."""
        self._visible = not self._visible

    def initialize(self) -> None:
        """Initialize ImGui or fallback HUD."""
        try:
            import imgui
            from imgui.integrations.glfw import GlfwRenderer

            imgui.create_context()
            self._impl = GlfwRenderer(self._window)
            self._renderer = imgui
            self._imgui_ready = True
        except (ImportError, OSError):
            self._init_fallback_hud()

    def _init_fallback_hud(self) -> None:
        self._hud_shader = ShaderProgram(
            self._shader_dir / "hud.vert",
            self._shader_dir / "hud.frag",
        )
        self._texture = GL.glGenTextures(1)
        indices = HUD_QUAD_INDICES
        self._hud_vao = GL.glGenVertexArrays(1)
        # DYNAMIC: đỉnh được tính lại mỗi khung từ kích thước texture chữ, để
        # quad bằng đúng canvas thay vì nén nó xuống một ô cố định.
        self._hud_vbo = GL.glGenBuffers(1)
        ebo = GL.glGenBuffers(1)
        GL.glBindVertexArray(self._hud_vao)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._hud_vbo)
        GL.glBufferData(GL.GL_ARRAY_BUFFER, 64, None, GL.GL_DYNAMIC_DRAW)
        GL.glBindBuffer(GL.GL_ELEMENT_ARRAY_BUFFER, ebo)
        GL.glBufferData(GL.GL_ELEMENT_ARRAY_BUFFER, indices.nbytes, indices, GL.GL_STATIC_DRAW)
        GL.glEnableVertexAttribArray(0)
        GL.glVertexAttribPointer(0, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, None)
        GL.glEnableVertexAttribArray(1)
        GL.glVertexAttribPointer(1, 2, GL.GL_FLOAT, GL.GL_FALSE, 16, GL.ctypes.c_void_p(8))
        GL.glBindVertexArray(0)
        self._fallback_ready = True

    def begin_frame(self) -> None:
        if not self._visible or not self._imgui_ready:
            return
        self._impl.process_inputs()
        self._renderer.new_frame()

    def draw(
        self,
        metrics: DebugMetrics,
        window_width: int = 1280,
        window_height: int = 720,
    ) -> None:
        if not self._visible:
            return
        if self._imgui_ready:
            self._draw_imgui(metrics)
        elif self._fallback_ready:
            self._draw_fallback(metrics, window_width, window_height)

    def _echo_to_console(self, lines: list[str]) -> None:
        """
        In khối HUD ra console, giãn cách ``HUD_CONSOLE_INTERVAL_S``.

        Để đọc số **trong khi vẫn nhìn cảnh**: overlay nằm đè lên cảnh, nên
        không thể vừa đọc ``Eye mm`` vừa quan sát thị sai. Chỉ chạy khi F1 đang
        bật, nên tắt F1 là console im.
        """
        now = time.perf_counter()
        if now - self._last_echo_at < HUD_CONSOLE_INTERVAL_S:
            return
        self._last_echo_at = now
        print("\n".join(["", "-" * 46, *lines, "-" * 46]), flush=True)

    def shutdown(self) -> None:
        if self._impl is not None:
            self._impl.shutdown()
        if self._texture:
            GL.glDeleteTextures(1, [self._texture])

    def _draw_imgui(self, metrics: DebugMetrics) -> None:
        imgui = self._renderer
        imgui.begin("HCPD Debug", True)
        imgui.text(f"FPS: {metrics.fps:.1f}")
        imgui.text(f"Frame: {metrics.frame_time_ms:.2f} ms")
        imgui.text(f"Filter: {metrics.filter_type}")
        for line in metrics.layer_grade_lines:
            imgui.text(line)
        imgui.end()
        imgui.render()
        self._impl.render(imgui.get_draw_data())

    def _draw_fallback(
        self,
        metrics: DebugMetrics,
        window_width: int,
        window_height: int,
    ) -> None:
        lines = self._format_lines(metrics)
        self._echo_to_console(lines)
        canvas_h = max(240, 24 + len(lines) * 20)
        canvas = np.zeros((canvas_h, 500, 3), dtype=np.uint8)
        canvas[:] = (20, 24, 30)
        y = 22
        for line in lines:
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (120, 255, 180), 1, cv2.LINE_AA)
            y += 20

        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGB, rgb.shape[1], rgb.shape[0], 0,
            GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb,
        )

        # Quad bằng ĐÚNG kích thước canvas chữ, neo mép trên-trái. Trước đây nó
        # là ô NDC cố định 25% x 20% viewport, nén canvas 500x~344 xuống còn
        # 320x144 px -> chữ cao ~4 px, không đọc nổi. Cùng cách toast_overlay
        # đang dùng.
        tex_w, tex_h = float(rgb.shape[1]), float(rgb.shape[0])
        x0 = -1.0 + 2.0 * HUD_MARGIN_PX / window_width
        x1 = x0 + 2.0 * tex_w / window_width
        y1 = 1.0 - 2.0 * HUD_MARGIN_PX / window_height
        y0 = y1 - 2.0 * tex_h / window_height
        vertices = hud_quad_vertices(x0, y0, x1, y1)
        GL.glBindBuffer(GL.GL_ARRAY_BUFFER, self._hud_vbo)
        GL.glBufferSubData(GL.GL_ARRAY_BUFFER, 0, vertices.nbytes, vertices)

        GL.glDisable(GL.GL_DEPTH_TEST)
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        assert self._hud_shader is not None
        self._hud_shader.use()
        tex_loc = GL.glGetUniformLocation(self._hud_shader.program_id, "uTexture")
        GL.glUniform1i(tex_loc, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        GL.glBindVertexArray(self._hud_vao)
        GL.glDrawElements(GL.GL_TRIANGLES, 6, GL.GL_UNSIGNED_INT, None)
        GL.glBindVertexArray(0)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_DEPTH_TEST)

    @staticmethod
    def _format_lines(metrics: DebugMetrics) -> list[str]:
        eye = metrics.eye_position_mm
        lines = [
            "HCPD DEBUG (F1)",
            f"FPS {metrics.fps:.1f}  Frame {metrics.frame_time_ms:.1f}ms  Q{metrics.quality_percent}%",
            f"Cap {metrics.capture_latency_ms:.1f}  Track {metrics.tracking_latency_ms:.1f}  Render {metrics.render_latency_ms:.1f}ms",
            f"Pose age {metrics.pose_age_ms:.1f}ms  Pipeline {metrics.pipeline_latency_ms:.1f}ms",
            f"Compensation {metrics.compensation_ms:.1f}ms  Total {metrics.total_latency_ms:.1f}ms",
            f"Eye mm {eye[0]:.0f} {eye[1]:.0f} {eye[2]:.0f}",
            f"Pose P{metrics.head_pitch_deg:.0f} Y{metrics.head_yaw_deg:.0f} R{metrics.head_roll_deg:.0f}",
            f"Filter {metrics.filter_type}  Pred {'ON' if metrics.prediction_enabled else 'OFF'}",
            f"Face {'DETECTED' if metrics.face_detected else 'LOST'}",
            f"P0 [{metrics.projection_matrix[0,0]:.3f} {metrics.projection_matrix[0,1]:.3f} "
            f"{metrics.projection_matrix[0,2]:.3f} {metrics.projection_matrix[0,3]:.3f}]",
            f"P1 [{metrics.projection_matrix[1,0]:.3f} {metrics.projection_matrix[1,1]:.3f} "
            f"{metrics.projection_matrix[1,2]:.3f} {metrics.projection_matrix[1,3]:.3f}]",
        ]
        if metrics.layer_grade_lines:
            lines.append("--- layers ---")
            lines.extend(metrics.layer_grade_lines)
        return lines

    # NOTE: this used to `return` the list literal directly, leaving the
    # layer_grade_lines block below as unreachable code — so the per-layer
    # status never appeared in the fallback HUD (only in the ImGui path).
