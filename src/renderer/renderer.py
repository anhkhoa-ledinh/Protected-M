"""Main OpenGL scene renderer."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
from OpenGL import GL

from src.common.stage_profile import stage
from src.math.homogeneous import to_column_major_flat
from src.renderer.camera import VirtualCamera
from src.renderer.debug_geometry import DebugGeometryRenderer
from src.renderer.debug_overlay import DebugMetrics, DebugOverlay
from src.renderer.shader import ShaderProgram
from src.renderer.webcam_preview import (
    TrackingPreviewSnapshot,
    WebcamPreviewOverlay,
    WebcamPreviewSettings,
)
from src.renderer.window import GlfwWindow
from src.content.prop_game import COLOR_FOUND
from src.scene.node import Material, SceneNode
from src.scene.primitives import create_mesh, create_sphere
from src.camera.capture import FramePacket
from src.config.schema import ContentSettings
from src.renderer.content_plane import ContentPlaneRenderer
from src.renderer.toast_overlay import ToastOverlay
from src.scene.window_box import build_scene_for


logger = logging.getLogger(__name__)


class SceneRenderer:
    """Phong-shaded scene renderer with head-coupled camera."""

    def __init__(
        self,
        window: GlfwWindow,
        shader_dir: Path,
        scene_root: SceneNode,
        camera: VirtualCamera,
        preview_settings: WebcamPreviewSettings | None = None,
        content: ContentSettings | None = None,
        project_root: Path | None = None,
    ) -> None:
        self._window = window
        self._camera = camera
        self._scene_root = scene_root
        self._content = content
        self._shader_dir = shader_dir
        self._project_root = project_root
        # D2 (TASK_ui_v3_1107): màn hình VẬT LÝ, chốt lúc dựng. Từ khi mặt
        # phẳng Kooima bám theo cửa sổ, ``camera.screen_width_m`` không còn là
        # cỡ màn hình nữa mà là cỡ **cửa sổ**. Hình học cảnh và bảng công thức
        # chiều sâu thì vẫn phải quy theo màn hình: đọc lại từ camera sau khi
        # đã thu nhỏ sẽ dựng tấm ảnh bằng đúng cỡ cửa sổ, và từ đó dời cửa sổ
        # là ảnh lệch đi — đúng cái người dùng báo ở USER NEEDED 2.
        self._screen_m = (float(camera.screen_width_m),
                          float(camera.screen_height_m))
        self._shader = ShaderProgram(
            shader_dir / "phong.vert",
            shader_dir / "phong.frag",
        )
        self._line_shader = ShaderProgram(
            shader_dir / "line.vert",
            shader_dir / "line.frag",
        )
        self._debug_geometry = DebugGeometryRenderer(self._line_shader)
        self._debug = DebugOverlay(window._window, shader_dir)
        self._preview = WebcamPreviewOverlay(
            shader_dir, preview_settings or WebcamPreviewSettings()
        )
        self._video_plane: ContentPlaneRenderer | None = None
        if (
            content is not None
            and content.enabled
            and content.mode == "video"
            and project_root is not None
        ):
            self._video_plane = ContentPlaneRenderer(
                shader_dir, content, project_root,
                screen_width_mm=self._screen_m[0] * 1000.0,
                screen_height_mm=self._screen_m[1] * 1000.0,
            )
        self._toast = ToastOverlay(shader_dir)
        # Gom vật ẩn dưới một node riêng để thay cả lượt mà không chạm scene cũ.
        self._prop_root = SceneNode(name="hidden_props")
        #: TASK_album_fix_1106 F3: hàm vẽ thêm vào CẢNH (bọ của giao diện
        #: album), gọi sau pass cảnh và TRƯỚC các lớp — đúng chỗ của quả cầu
        #: cũ, nên cơ chế lớp-che-bọ không đổi.
        self.extra_scene_draw = None
        #: drawn after the photo layers, depth test off (the thermal marks)
        self.post_layer_draw = None
        # Chẩn đoán C2, tắt mặc định: HCPD_PROP_PROBE=1 in state depth thật.
        self._probe = os.environ.get("HCPD_PROP_PROBE") == "1"
        self._probe_logged = 0
        self._ambient = 0.22 if self._video_plane else 0.12
        self._dir_light = np.array([-0.4, -1.0, -0.2], dtype=np.float64)
        self._point_light = np.array([1.5, 2.5, -1.0], dtype=np.float64)

    def initialize(self) -> None:
        """Prepare renderer resources."""
        self._debug.initialize()
        self._preview.initialize()
        self._toast.initialize()
        if self._video_plane is not None:
            self._video_plane.initialize()

    def update_content(self, dt_s: float) -> None:
        """
        Chạy **nhịp** của content plane mà không vẽ gì.

        Giao diện album chỉ gọi ``render`` khi màn đang mở có ô viewport, nên
        ở màn tách lớp content plane đứng im: depth tính xong mà không ai dựng
        lớp, thanh tiến trình kẹt ở 95 % mãi (người dùng gặp 06/11 22:41).
        """
        if self._video_plane is not None:
            self._video_plane.update(dt_s)

    def show_toast(self, message: str, *, is_error: bool = False) -> None:
        self._toast.show(message, is_error=is_error)

    def take_content_notice(self) -> str | None:
        """Thông báo chờ của content plane (FR-DEPTH-03); None khi không có."""
        if self._video_plane is None:
            return None
        return self._video_plane.take_pending_notice()

    def layer_records(self):
        """Hình học các lớp **đúng như đã vẽ**; rỗng khi chưa có content plane."""
        if self._video_plane is None:
            return ()
        return self._video_plane.layer_records()

    @property
    def layers_revision(self) -> int:
        """Số lần bộ lớp được dựng lại (xem ``ContentPlaneRenderer``)."""
        if self._video_plane is None:
            return 0
        return self._video_plane.layers_revision

    @property
    def layer_hole_stats(self):
        """Phần lỗ của từng lớp (xem ``ContentPlaneRenderer``)."""
        if self._video_plane is None:
            return ()
        return self._video_plane.layer_hole_stats

    @property
    def waiting_for_depth(self) -> bool:
        """Ảnh hiện tại chưa tách lớp xong (PB-051)."""
        return self._video_plane is not None and self._video_plane.waiting_for_depth

    def refresh_layers(self) -> bool:
        """Dựng lại các lớp sau khi ``ContentSettings`` đổi trong bộ nhớ."""
        if self._video_plane is None:
            return False
        return self._video_plane.refresh_layers()

    def set_props(self, items) -> None:
        """
        Dựng lại các node vật ẩn dưới scene root. ``items`` là
        ``(x_m, y_m, z_m, radius_m, color)``.

        Vật là ``SceneNode`` thường nên đi theo đường vẽ Phong có depth write,
        chạy TRƯỚC các lớp. Lớp vẽ sau với depth test vẫn bật và depth write
        tắt, nên tấm nền ở xa trượt depth test (không phủ vật) còn lớp gần hơn
        blend lên trên (pixel đục che vật). Đó là toàn bộ cơ chế ẩn/hiện.
        """
        for node in self._prop_root.children:
            if node.mesh is not None:
                node.mesh.release()
        self._prop_root.children.clear()
        for index, (x_m, y_m, z_m, radius_m, color) in enumerate(items):
            self._prop_root.children.append(
                SceneNode(
                    name=f"prop_{index}",
                    mesh=create_mesh(create_sphere(float(radius_m))),
                    material=Material(color=tuple(color), shininess=48.0),
                    translation=np.array([x_m, y_m, z_m], dtype=np.float64),
                )
            )
        if self._prop_root not in self._scene_root.children:
            self._scene_root.children.append(self._prop_root)

    def _log_depth_state(self, where: str) -> None:
        """Chẩn đoán C2: state depth THẬT lúc chạy, không suy từ code."""
        logger.info(
            "GL depth [%s]: test=%s func=%d mask=%s",
            where,
            bool(GL.glIsEnabled(GL.GL_DEPTH_TEST)),
            int(GL.glGetIntegerv(GL.GL_DEPTH_FUNC)),
            bool(GL.glGetBooleanv(GL.GL_DEPTH_WRITEMASK)),
        )

    def _log_probe_once(self) -> None:
        """Một lần: số bit depth của framebuffer và các node vật đang vẽ."""
        bits = GL.glGetFramebufferAttachmentParameteriv(
            GL.GL_DRAW_FRAMEBUFFER, GL.GL_DEPTH,
            GL.GL_FRAMEBUFFER_ATTACHMENT_DEPTH_SIZE,
        )
        nodes = [
            f"{n.name}@({n.translation[0]:+.3f},{n.translation[1]:+.3f},"
            f"{n.translation[2]:+.3f}) mesh={n.mesh is not None}"
            for n in self._prop_root.children
        ]
        drawn = [n.name for n, _ in self._scene_root.traverse() if n.mesh is not None]
        logger.info("GL depth bits=%s | prop nodes: %s", bits, "; ".join(nodes))
        logger.info("Scene meshes ve theo thu tu: %s", ", ".join(drawn))

    def set_prop_found(self, index: int) -> None:
        """Đổi màu một vật đã tìm thấy. Không dựng lại mesh."""
        if 0 <= index < len(self._prop_root.children):
            self._prop_root.children[index].material.color = COLOR_FOUND

    @property
    def layer_grade_lines(self) -> list[str]:
        if self._video_plane is None:
            return []
        return self._video_plane.layer_grade_lines

    def toggle_atmospheric(self) -> bool | None:
        """Flip atmospheric grading on the content plane; ``None`` if unavailable."""
        if self._video_plane is None:
            return None
        return self._video_plane.toggle_atmospheric()

    def toggle_mist(self) -> bool | None:
        """Flip atmospheric mist quad on the content plane; ``None`` if unavailable."""
        if self._video_plane is None:
            return None
        return self._video_plane.toggle_mist()

    def toggle_relief(self) -> bool | None:
        """Flip intra-layer relief meshes on the content plane; ``None`` if unavailable."""
        if self._video_plane is None:
            return None
        return self._video_plane.toggle_relief()

    def toggle_disocclusion(self) -> bool | None:
        """Flip the C1 disocclusion fill; ``None`` if unavailable."""
        if self._video_plane is None:
            return None
        return self._video_plane.toggle_disocclusion()

    def nudge_plane_z(
        self,
        delta_m: float,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
    ) -> float | None:
        """Dịch content plane theo Z (cần gạt chính). ``None`` nếu chưa sẵn sàng."""
        if self._video_plane is None:
            return None
        return self._video_plane.nudge_plane_z(
            delta_m, eye_distance_mm, screen_width_mm, screen_height_mm
        )

    def nudge_layer_z_far(
        self,
        delta_m: float,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
    ) -> float | None:
        """Dịch lớp xa nhất (cần gạt phụ). ``None`` nếu chưa sẵn sàng."""
        if self._video_plane is None:
            return None
        return self._video_plane.nudge_layer_z_far(
            delta_m, eye_distance_mm, screen_width_mm, screen_height_mm
        )

    def nudge_overscan(
        self,
        delta_mm: float,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
    ) -> float | None:
        """Chỉnh biên overscan e_max. ``None`` nếu chưa sẵn sàng."""
        if self._video_plane is None:
            return None
        return self._video_plane.nudge_overscan(
            delta_mm, eye_distance_mm, screen_width_mm, screen_height_mm
        )

    @property
    def overscan_step_mm(self) -> float:
        """Bước chỉnh e_max, mm."""
        from src.renderer.content_plane import ContentPlaneRenderer

        return ContentPlaneRenderer.OVERSCAN_STEP_MM

    @property
    def depth_step_m(self) -> float:
        """Bước quét chiều sâu, mét."""
        from src.renderer.content_plane import ContentPlaneRenderer

        return ContentPlaneRenderer.DEPTH_STEP_M

    def load_content_file(self, path, *, label: str | None = None) -> tuple[bool, str]:
        """Hot-swap the content plane media without restarting. ``path`` is a
        ``Path`` or an already checked ``media_gate.CleanPhoto``."""
        if self._project_root is None or self._content is None:
            return False, "Content system not initialized"
        self._ensure_content_plane()
        assert self._video_plane is not None
        return self._video_plane.load_file(path, label=label)

    @property
    def current_clean_photo(self):
        """``media_gate.CleanPhoto`` on screen now (``None``: a video or none)."""
        plane = self._video_plane
        return None if plane is None else plane.clean_photo

    def _ensure_content_plane(self) -> None:
        if self._video_plane is not None:
            return
        if self._content is None or self._project_root is None:
            return
        self._content.mode = "video"
        self._scene_root = build_scene_for(
            self._content, self._screen_m[0], self._screen_m[1])
        self._video_plane = ContentPlaneRenderer(
            self._shader_dir, self._content, self._project_root,
            screen_width_mm=self._screen_m[0] * 1000.0,
            screen_height_mm=self._screen_m[1] * 1000.0,
        )
        self._video_plane.initialize()
        self._ambient = 0.22

    @property
    def debug_overlay(self) -> DebugOverlay:
        return self._debug

    @property
    def webcam_preview(self) -> WebcamPreviewOverlay:
        return self._preview

    def render(
        self,
        metrics: DebugMetrics,
        frame: FramePacket | None = None,
        tracking_preview: TrackingPreviewSnapshot | None = None,
        dt_s: float = 1.0 / 60.0,
    ) -> None:
        """Draw one frame of the virtual room."""
        width, height = self._window.width, self._window.height
        GL.glViewport(0, 0, width, height)
        if self._content is not None:
            cc = self._content.void_clear_color
            GL.glClearColor(cc[0], cc[1], cc[2], 1.0)
        else:
            GL.glClearColor(0.05, 0.06, 0.08, 1.0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT | GL.GL_DEPTH_BUFFER_BIT)
        probe = self._probe and self._prop_root.children and self._probe_logged < 2
        if probe:
            self._probe_logged += 1
            self._log_probe_once()
            self._log_depth_state("sau glClear")

        view_flat, proj_flat = self._camera.view_projection_flat()
        # TASK_game_feel_1118 P6: the "steady eye" while scoped (a no-op
        # otherwise) — ``uViewPos`` feeds specular lighting, which would
        # otherwise use the raw tracked eye while the vertex transform
        # above (view_flat/proj_flat) already uses the steadied one.
        eye = self._camera.effective_eye_position_m

        self._shader.use()
        self._shader.set_mat4("uView", view_flat)
        self._shader.set_mat4("uProjection", proj_flat)
        self._shader.set_vec3("uViewPos", float(eye[0]), float(eye[1]), float(eye[2]))
        self._shader.set_vec3(
            "uDirLightDir",
            float(self._dir_light[0]),
            float(self._dir_light[1]),
            float(self._dir_light[2]),
        )
        self._shader.set_vec3("uDirLightColor", 0.9, 0.9, 0.85)
        self._shader.set_vec3(
            "uPointLightPos",
            float(self._point_light[0]),
            float(self._point_light[1]),
            float(self._point_light[2]),
        )
        self._shader.set_vec3("uPointLightColor", 1.0, 0.85, 0.65)
        self._shader.set_float("uAmbient", self._ambient)

        for node, world in self._scene_root.traverse():
            if node.mesh is None:
                continue
            self._shader.set_mat4("uModel", to_column_major_flat(world))
            color = node.material.color
            self._shader.set_vec3("uBaseColor", color[0], color[1], color[2])
            self._shader.set_float("uShininess", node.material.shininess)
            node.mesh.draw()

        if probe:
            self._log_depth_state("sau scene pass, truoc layer")
        if self.extra_scene_draw is not None:
            with stage("bugs"):
                self.extra_scene_draw(view_flat, proj_flat)
            self._shader.use()
        if self._video_plane is not None:
            self._video_plane.update(dt_s)
            # PB-076: khung hoãn vừa được đặt trong ``update``; toast phải lên
            # ngay khung này vì ``self._toast.draw`` ở cuối hàm, rồi khung sau
            # mới dựng lớp (và đứng hình).
            build_notice = self._video_plane.take_build_notice()
            if build_notice is not None:
                self._toast.show(build_notice)
            with stage("layer_render"):
                self._video_plane.draw(view_flat, proj_flat)
        if probe:
            self._log_depth_state("sau layer")
        if self.post_layer_draw is not None:
            with stage("bugs_heat"):
                self.post_layer_draw(view_flat, proj_flat)
            self._shader.use()

        self._debug.begin_frame()
        if self._debug.visible:
            self._debug_geometry.upload_segments(self._camera.debug_frustum_lines())
            self._debug_geometry.draw(view_flat, proj_flat)
        # Cùng nguồn kích thước với glViewport ở trên, để quad HUD khớp viewport
        # thật sự đang dùng (xem PB-038 về việc nguồn đó lấy từ config).
        self._debug.draw(metrics, width, height)

        preview = tracking_preview or TrackingPreviewSnapshot()
        self._preview.draw(
            frame, preview, self._window.width, self._window.height
        )
        self._toast.draw(self._window.width, self._window.height)

    def release(self) -> None:
        """Free GPU resources."""
        for node in self._prop_root.children:
            if node.mesh is not None:
                node.mesh.release()
        self._prop_root.children.clear()
        self._toast.release()
        if self._video_plane is not None:
            self._video_plane.release()
        self._preview.release()
        self._debug_geometry.release()
        self._line_shader.release()
        self._debug.shutdown()
        self._shader.release()
