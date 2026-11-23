"""Unified image/video content plane renderer with optional depth displacement."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np
import numpy.typing as npt
from OpenGL import GL

from src.config.schema import ContentSettings
from src.common.stage_profile import stage
from src.content.bake import BakeWorker
from src.content.depth_worker import DepthWorker
from src.content.formats import detect_content_type
from src.content.image_source import ImageSource
from src.content.adaptive_k import AdaptiveKReport
from src.content.debug_marks import paint_border, pad_colour
from src.scene.window_box import box_layer_scale
from src.content.layer_shadow import (resample_to, shadow_geometry,
                                      shadow_rgba)
from src.content.layer_clustering import LayerGradeReport, assign_layer_z, inter_layer_gap_m
from src.math.homogeneous import to_column_major_flat, translation_matrix
from src.renderer.shader import ShaderProgram
from src.scene.textured_mesh import TexturedMesh
from src.renderer.video_player import VideoPlayer
from src.common.textures import bleed_transparent
from src.scene.primitives import (
    RELIEF_CLAMP_GAP_FRACTION,
    create_displaced_vertical_plane_uv,
    create_layer_relief_vertical_plane_uv,
    create_vertical_plane_uv,
)

logger = logging.getLogger(__name__)

#: Z của mặt phẳng màn hình, mm. ``screen.json`` ship ``screen_center = [0,0,0]``
#: và Kooima đặt màn ở z = 0, nên hằng số này đúng hôm nay. Nếu
#: ``screen_center[2]`` khác 0 thì phải truyền giá trị thật vào
#: ``ContentPlaneRenderer`` thay vì dùng hằng ở đây.
SCREEN_Z_MM = 0.0

#: Lời nhắn nằm trên màn suốt lúc dựng lớp (PB-076). ASCII thuần: toast vẽ bằng
#: ``FONT_HERSHEY_SIMPLEX`` của OpenCV, ký tự ngoài ASCII thành ô vuông.
LAYER_BUILD_TOAST = "Building 3D layers... (large photos can take ~30 s)"

#: Kiểu viền cho dải đệm ở chế độ ``pad``.
#:
#: ``BORDER_REFLECT_101`` soi gương nội dung ra ngoài. Trên ảnh thật nó tạo cấu
#: trúc giả rất dễ nhận: người dùng thử t21 ngày 2026-10-13 thấy cành cây ra
#: ĐỐI XỨNG GƯƠNG, "kì cục và không chân thực". ``BORDER_REPLICATE`` kéo dài
#: điểm mép thành vệt thẳng — vẫn là nội dung bịa, nhưng không dựng lên hình
#: thể mới nên mắt ít bắt hơn. Dải này chỉ lộ ra khi mắt dịch khỏi tâm, đúng
#: lúc người xem đang chú ý, nên chọn kiểu ít gây chú ý là đáng.
PAD_BORDER_MODE = cv2.BORDER_REPLICATE

#: PB-073: nâng mặt bóng 1 mm về phía người xem để nó nằm TRÊN mặt nhận.
SHADOW_LIFT_M = 0.001


def front_layer_half_width_limit(t_mm, screen_w_mm, e_max_mm, design_d_mm):
    """
    Nửa bề rộng tối đa của một lớp nằm TRƯỚC kính (PB-010, FR-LAYER-04).

    Lớp sau kính cần quad **rộng ra** để mắt dịch không nhìn lọt ra ngoài; lớp
    trước kính thì ngược lại — nó phải **hẹp lại**, nếu không mép của nó chạm
    viền màn và bị khung cắt, tức vi phạm cửa sổ. Hai ràng buộc ngược dấu nhau
    nên không thể dùng chung một công thức.

        limit = (W/2)·(Dd + t)/Dd − e_max·|t|/Dd

    với ``t < 0``. Ví dụ t = −100, Dd = 600, e_max = 150 → 118.3 mm.
    Trả ``inf`` khi ``t ≥ 0`` (không ràng buộc).
    """
    if t_mm >= 0.0:
        return float("inf")
    d = max(float(design_d_mm), 1.0)
    return screen_w_mm * 0.5 * (d + t_mm) / d - float(e_max_mm) * abs(t_mm) / d


def overscan_need_mm(screen_side_mm, t_mm, e_max_mm, design_d_mm):
    """
    Nửa kích thước quad TỐI THIỂU theo một trục, để mắt dịch ``e_max`` không
    nhìn lọt ra ngoài mép lớp ở độ sâu ``t``.

        need = (S/2)·(Dd + t)/Dd + e_max·t/Dd

    Cả hai chế độ overscan phải thoả đúng con số này: ``scale`` thoả bằng cách
    phóng quad, ``pad`` bằng cách co nội dung rồi đệm thêm viền.
    """
    d = max(float(design_d_mm), 1.0)
    return screen_side_mm * 0.5 * (d + t_mm) / d + float(e_max_mm) * t_mm / d


@dataclass(frozen=True)
class LayerGeometry:
    """Hình học quad của một lớp, đúng như đường vẽ dùng."""

    t_mm: float                 # độ sâu có dấu so với mặt kính
    scale: float                # hệ số overscan k
    quad_w_m: float
    quad_h_m: float
    centre_x_m: float           # tâm quad sau khi dời theo mắt thiết kế
    centre_y_m: float


#: FR-DEPTH-03: ở chế độ lớp, video chỉ được tách lớp MỘT lần từ khung đầu —
#: ảnh vẫn chạy nhưng hình học lớp đứng yên. Phải nói rõ cho người dùng biết
#: thay vì để họ tưởng app treo hoặc tưởng lớp đang chạy theo video.
STATIC_VIDEO_NOTICE = "Video: depth from the first frame only"


def static_video_notice(kind: str, layer_clustering_enabled: bool) -> str | None:
    """
    Trả câu thông báo khi video chạy ở chế độ lớp; không thì ``None``.

    Tách thành hàm thuần để test được mà không cần context OpenGL.
    """
    if kind == "video" and layer_clustering_enabled:
        return STATIC_VIDEO_NOTICE
    return None


def layer_geometry(overscan_scale, content, plane_z_m, layer_z_m,
                   base_w_m, base_h_m, screen_w_mm, screen_h_mm):
    """
    MỘT nguồn sự thật cho hình học quad: ``_apply_bake`` gọi hàm này, và mọi
    thứ cần biết lớp nằm đâu (đặt vật ẩn, script đo) cũng gọi đúng hàm này.

    ``MEMORY.md`` mục 7: script đo B2 và renderer từng lệch nhau qua ``abs(t)``
    suốt một ngày mà không ai thấy, vì mỗi bên giữ một bản công thức.

    ``overscan_scale`` truyền vào thay vì gọi thẳng, để chỗ gọi quyết định dùng
    hàm nào — ``_apply_bake`` đưa method của chính nó.
    """
    t_mm = SCREEN_Z_MM - (float(plane_z_m) + float(layer_z_m)) * 1000.0
    ox, oy = design_eye_offset_mm(
        content.overscan_design_eye_mm, t_mm,
        content.overscan_design_distance_mm,
    )
    if str(getattr(content, "surround", "void")).lower().strip() == "box":
        # PB-034: quad phủ kín tiết diện hộp, **vẫn dời tâm theo mắt thiết kế**.
        # Bản đầu ghim tâm về 0 và ảnh rách theo chiều dọc: trên demo mắt thiết
        # kế cao 100 mm nên lớp xa phải dời −77,8 mm, tức 40 % chiều cao ảnh —
        # xem ``box\t25_box_x-100.png`` lượt 30/10 18:34 so với
        # ``before\t25_demo_x-100.png``.
        k = box_layer_scale(t_mm, base_w_m, base_h_m, screen_w_mm, screen_h_mm,
                            content.overscan_design_distance_mm, (ox, oy))
    else:
        k = overscan_scale(t_mm, base_w_m, base_h_m, screen_w_mm, screen_h_mm)
    return LayerGeometry(
        t_mm=t_mm, scale=k,
        quad_w_m=base_w_m * k, quad_h_m=base_h_m * k,
        centre_x_m=ox / 1000.0, centre_y_m=oy / 1000.0,
    )


def design_eye_offset_mm(design_eye_mm, t_mm, design_d_mm):
    """
    Dời tâm quad của một lớp để nó khớp vị trí ở **mắt thiết kế**, không phải
    ở tâm màn.

        offset = −(x0, y0) · t / Dd

    Nhìn từ mắt ở ``(x0, y0, Dd)``, một lớp ở độ sâu ``t`` phải dịch đúng chừng
    này mới chiếu về giữa màn. ``t`` có dấu như L2a, nên lớp trước kính dời
    ngược chiều. Hệ số overscan ``k`` không đổi — đây chỉ là tịnh tiến.
    """
    d = max(float(design_d_mm), 1.0)
    return (
        -float(design_eye_mm[0]) * float(t_mm) / d,
        -float(design_eye_mm[1]) * float(t_mm) / d,
    )


def pad_content_scale(t_mm, t_plane_mm, design_d_mm):
    """
    Hệ số co nội dung ở chế độ ``pad`` để lớp KHỚP VỊ TRÍ với mặt phẳng.

        s = (Dd + t) / (Dd + t_plane)

    Nhìn từ mắt thiết kế, một vật ở độ sâu ``t`` hiện nhỏ hơn cùng vật ở độ sâu
    ``t_plane`` theo đúng tỉ số này. Nhân nội dung với ``s`` thì lỗ của lớp xa
    lại nằm đúng sau vật che nó — thứ mà chế độ ``scale`` làm lệch đi M lần.
    """
    d = max(float(design_d_mm), 1.0)
    return (d + float(t_mm)) / (d + float(t_plane_mm))


def pad_margin_mm(t_mm, e_max_mm, design_d_mm):
    """
    Bề rộng dải đệm mỗi bên, mm: ``m = e_max · t / Dd`` khi ``t > 0``.

    Đúng bằng phần mà mắt dịch ``e_max`` nhìn lọt ra ngoài mép lớp ở độ sâu
    ``t``. Lớp ở hoặc trước kính (``t ≤ 0``) không cần đệm.
    """
    if float(t_mm) <= 0.0:
        return 0.0
    return float(e_max_mm) * float(t_mm) / max(float(design_d_mm), 1.0)


def relief_front_offset_m(z_world_m):
    """
    Lùi lớp lại bao nhiêu (mét) để không đỉnh nào còn nằm trước kính.

    ``offset = max(z_world, 0)`` trên toàn bộ đỉnh. Đây là một phép TỊNH TIẾN
    nên hình dạng relief không đổi — chỉ cái nền nó ngồi lên lùi lại. Trả 0 khi
    đã không có đỉnh nào nhô ra.
    """
    top = float(np.max(np.asarray(z_world_m, dtype=np.float64)))
    return max(top, 0.0)


@dataclass
class _MistRenderable:
    """Semi-transparent atmospheric haze between the two farthest layers."""

    mesh: TexturedMesh
    color: tuple[float, float, float]
    opacity: float
    z_m: float


@dataclass
class _LayerRenderable:
    """A single depth-clustered layer ready to draw (RGBA texture + quad)."""

    texture: int
    mesh: TexturedMesh
    z_m: float
    # Các trường dưới GHI LẠI thứ đường vẽ đã tính, không tính lại. Ai cần biết
    # lớp nằm đâu (đặt vật ẩn ở C1) đọc từ đây, đừng dựng công thức thứ hai.
    alpha: npt.NDArray[np.uint8] | None = None   # đúng mảng đã nạp lên texture
    quad_w_m: float = 0.0
    quad_h_m: float = 0.0
    centre_x_m: float = 0.0
    centre_y_m: float = 0.0
    z_world_m: float = 0.0        # sau relief shift
    relief_z_min_m: float = 0.0   # world z thật của đỉnh sâu nhất / gần nhất
    relief_z_max_m: float = 0.0
    # PB-073: bóng của lớp GẦN HƠN kế tiếp hắt xuống chính lớp này. Gắn vào
    # mặt nhận chứ không thêm phần tử vào ``self._layers``, để ``layer_records``
    # và chỗ đặt vật ẩn vẫn thấy đúng bấy nhiêu lớp như trước.
    shadow_texture: int = 0
    shadow_mesh: TexturedMesh | None = None


class _MediaSource(Protocol):
    aspect_ratio: float

    def tick(self, dt_s: float) -> npt.NDArray[np.uint8] | None: ...
    def close(self) -> None: ...


class ContentPlaneRenderer:
    """
    Draw a textured quad (flat or depth-displaced) behind the monitor plane.

    Supports live media swap via ``load_file`` without restarting the app.
    """

    def __init__(
        self,
        shader_dir: Path,
        content: ContentSettings,
        root: Path,
        screen_width_mm: float = 344.0,
        screen_height_mm: float = 194.0,
    ) -> None:
        self._content = content
        # Kích thước màn hình vật lý, cần để tính overscan lúc dựng mesh.
        # Truyền từ SceneRenderer (nó có VirtualCamera) thay vì tự đọc
        # screen.json — một nguồn sự thật, và ContentPlane không phải biết
        # tới file cấu hình nào.
        self._screen_w_mm = float(screen_width_mm)
        self._screen_h_mm = float(screen_height_mm)
        self._shader_dir = shader_dir
        self._position = np.asarray(content.plane_position_m, dtype=np.float64)
        self._plane_width = content.plane_width_m
        self._plane_height = content.plane_height_m
        self._source: _MediaSource | None = None
        self._is_video = True
        self._mesh = self._build_flat_mesh(16.0 / 9.0)
        self._shader = ShaderProgram(shader_dir / "video.vert", shader_dir / "video.frag")
        self._video_uniform = -1
        self._texture = 0
        self._tex_width = 0
        self._tex_height = 0
        self._ready = False
        self._current_path: Path | None = None
        # FR-DEPTH-03: thông báo chờ hiện toast; lần nạp lúc khởi động xảy ra
        # trước khi có toast nên phải giữ lại để app lấy sau.
        self._pending_notice: str | None = None
        self._depth_worker: DepthWorker | None = None
        self._depth_mesh_key: str | None = None
        # PB-076: bộ lớp đã có depth nhưng cố ý hoãn một khung (xem
        # ``_poll_depth_mesh``), và lời nhắn cho toast của khung hoãn đó.
        self._deferred_layers: tuple[object, str] | None = None
        self._build_notice: str | None = None
        self._last_depth_rgb: npt.NDArray[np.uint8] | None = None
        self._last_depth_map: npt.NDArray[np.float32] | None = None
        self._using_displaced_mesh = False
        self._layer_shader: ShaderProgram | None = None
        self._layer_video_uniform = -1
        self._layers: list[_LayerRenderable] = []
        self._mist: _MistRenderable | None = None
        self._mist_shader: ShaderProgram | None = None
        self._mist_runtime_enabled = bool(content.mist_enabled)
        self._relief_runtime_enabled = bool(content.relief_enabled)
        self._disocclusion_runtime_enabled = bool(content.disocclusion_fill_enabled)
        self._layer_reports: list[LayerGradeReport] = []
        self._semantic_report = None  # L1: san o buoc nuong, khong con o day
        self._adaptive_k_report: AdaptiveKReport | None = None
        self._layer_hole_pct: list[float] = []

        # L1 (TASK_game_1110): ở chế độ lớp, depth + cụm + lấp khe chạy hết
        # trong một TIẾN TRÌNH con (``BakeWorker``), nên không luồng nào của
        # app đụng onnxruntime nữa. ``DepthWorker`` chỉ còn phục vụ chế độ
        # mesh-biến-dạng (không tách lớp), đường mà album không dùng.
        self._bake_worker: BakeWorker | None = None
        self._last_bake = None
        if content.depth_enabled and content.layer_clustering_enabled:
            self._bake_worker = BakeWorker(root)
        elif content.depth_enabled:
            self._depth_worker = DepthWorker(content.depth_model, content.depth_backend)

        initial = content.resolved_media_path(root)
        logger.info("Content plane initial media: %s", initial)
        ok, msg = self.load_file(initial, strict=False)
        if not ok or self._source is None:
            logger.warning(
                "Initial media load failed (%s) — using procedural fallback", msg
            )
            fallback = VideoPlayer(initial, loop=content.loop_video)
            self._source = fallback
            self._is_video = True
            self._resize_plane(fallback.aspect_ratio)

    @property
    def current_path(self) -> Path | None:
        return self._current_path

    @property
    def atmospheric_enabled(self) -> bool:
        return bool(self._content.atmospheric_enabled)

    @property
    def layer_grade_lines(self) -> list[str]:
        """One short status line per graded layer for the F1 overlay."""
        if not self._layer_reports:
            return []
        state = "ON" if self._content.atmospheric_enabled else "OFF"
        lines = [
            f"Atmospheric {state} (F8)  layers={len(self._layer_reports)}  "
            f"mist={'ON' if self._mist_runtime_enabled else 'OFF'} (F7)  "
            f"relief={'ON' if self._relief_runtime_enabled else 'OFF'} (F10)  "
            f"fill={'ON' if self._disocclusion_runtime_enabled else 'OFF'} (F9)",
        ]
        # Reports are far->near; label the extremes for readability.
        for rep in self._layer_reports:
            lines.append(
                f" L{rep.index} d={rep.depth_norm:.2f} "
                f"blend={rep.atmosphere_blend * 100:.0f}% "
                f"sat={rep.saturation_factor:.2f} con={rep.contrast_factor:.2f}"
            )
        if self._semantic_report is not None:
            rep = self._semantic_report
            lines.append(
                f"Semantic depth: segs={rep.num_segments} flat={rep.segments_flattened} "
                f"sep={rep.segment_pairs_separated}"
            )
        if self._adaptive_k_report is not None:
            rep = self._adaptive_k_report
            lines.append(
                f"Adaptive K={rep.chosen_k} [gvf] thr={rep.gvf_threshold:.2f} "
                f"flat={rep.flat_fallback} "
                f"gvf_var={{{', '.join(f'{k}:{v:.3f}' for k, v in sorted(rep.gvf_variance.items()))}}}"
            )
            if rep.rejections:
                lines.append(f" K rejections: {'; '.join(rep.rejections)}")
        return lines

    def toggle_mist(self) -> bool:
        """Flip atmospheric mist quad ON/OFF at runtime (F7 A/B)."""
        self._mist_runtime_enabled = not self._mist_runtime_enabled
        logger.info(
            "Atmospheric mist toggled -> %s",
            "ON" if self._mist_runtime_enabled else "OFF",
        )
        return self._mist_runtime_enabled

    @property
    def relief_enabled(self) -> bool:
        return self._relief_runtime_enabled

    def toggle_relief(self) -> bool:
        """Flip intra-layer relief meshes ON/OFF and rebuild layers (F10 A/B)."""
        self._relief_runtime_enabled = not self._relief_runtime_enabled
        self.refresh_layers()
        logger.info(
            "Intra-layer relief toggled -> %s",
            "ON" if self._relief_runtime_enabled else "OFF",
        )
        return self._relief_runtime_enabled

    def toggle_disocclusion(self) -> bool:
        """
        Flip the C1 background fill (F9 A/B).

        L1: lấp khe là việc của **bước nướng**, không phải của lúc vẽ, nên
        phím này nay xin nướng lại chứ không dựng lại mesh. Khoá cấu hình đổi
        → khoá cache đổi → lượt sau bấm lại là cache trúng, gần như tức thì.
        """
        self._disocclusion_runtime_enabled = not self._disocclusion_runtime_enabled
        self._content.disocclusion_fill_enabled = self._disocclusion_runtime_enabled
        self._rebake("disocclusion")
        logger.info(
            "Disocclusion fill toggled -> %s",
            "ON" if self._disocclusion_runtime_enabled else "OFF",
        )
        return self._disocclusion_runtime_enabled

    @property
    def disocclusion_enabled(self) -> bool:
        return self._disocclusion_runtime_enabled

    # --- Quét chiều sâu lúc chạy (Phase 2) -----------------------------------
    #
    # HAI cần gạt, không phải một, và chúng KHÔNG mạnh như nhau.
    #
    # `plane_position_m[2]` là cần gạt CHÍNH. `layer_z_far_m` là phụ. Lý do là
    # hệ số dịch trên màn hình `t / (D + t)` **bão hoà** khi `t` lớn: ba lớp
    # hiện nằm ở 750/925/1100 mm sau màn hình, tức trong vùng 0.556–0.647, nên
    # tỉ số thị sai xa/gần chỉ 1.165×. Đẩy lớp xa ra sau nữa là kéo sâu hơn vào
    # vùng bão hoà; kéo cả mặt phẳng về gần màn hình thì đi ngược lại.
    #
    # Đo bằng scripts/measure_parallax.py (khớp giải tích tới 3e-16):
    #   plane_z  -0.75 -> ti so 1.165x     z_far  -0.35 -> ti so 1.165x
    #   plane_z  -0.20 -> ti so 1.913x     z_far  -2.50 -> ti so 1.519x
    # plane_z đạt tỉ số cao hơn mà GIỮ NGUYÊN khoảng cách giữa hai lớp 175 mm,
    # còn z_far phải nới khoảng cách gấp bảy — chính khoảng cách đó đẻ ra hai
    # tác dụng phụ: relief bị kẹp theo gap, và lỗ disocclusion rộng ra.

    #: Bước quét, mét. 50 mm đủ nhỏ để thấy khác biệt, đủ lớn để không phải
    #: bấm hàng chục lần.
    DEPTH_STEP_M = 0.05

    #: Bước chỉnh e_max lúc chạy, mm.
    OVERSCAN_STEP_MM = 25.0

    #: Tăng mỗi lần ``_apply_bake`` dựng xong một bộ lớp mới. Người đọc ngoài
    #: (trò chơi PB-015) nhờ đó biết hình học đã đổi mà không phải so từng quad.
    #: Để ở mức class nên instance dựng bằng ``__new__`` trong script đo vẫn đọc
    #: được, không cần gán thêm thuộc tính.
    _layers_revision = 0

    def _pad_layer(self, layer, depth_map, t_mm, t_plane_mm, width, height, t_margin):
        """
        Chế độ ``pad``: co nội dung cho khớp mặt phẳng rồi đệm viền quanh nó.

        Trả ``(rgba, mask, depth, quad_w_m, quad_h_m, pad_px)``. ``rgba``,
        ``mask`` và ``depth`` được đệm CÙNG số pixel để UV của
        ``create_layer_relief_vertical_plane_uv`` còn khớp; kích thước quad suy
        ngược từ số pixel thật nên không có sai số làm lệch UV.
        """
        e_max = float(self._content.overscan_e_max_mm)
        d_mm = float(self._content.overscan_design_distance_mm)
        s = pad_content_scale(t_mm, t_plane_mm, d_mm)
        content_w_m, content_h_m = width * s, height * s
        m_mm = pad_margin_mm(t_margin, e_max, d_mm)
        tex_w = int(layer.rgba.shape[1])
        pad_px = int(round(m_mm / max(content_w_m * 1000.0, 1e-6) * tex_w))
        if pad_px <= 0:
            return layer.rgba, layer.mask, depth_map, content_w_m, content_h_m, 0
        grow = (tex_w + 2 * pad_px) / tex_w
        border = PAD_BORDER_MODE
        rgba = cv2.copyMakeBorder(layer.rgba, pad_px, pad_px, pad_px, pad_px, border)
        # PB-074 (đo): tô vành đệm để đếm được nó trên khung hình. Không đặt
        # HCPD_MARK_PAD thì ``paint_border`` trả nguyên ``rgba``.
        paint_border(rgba, pad_px, pad_colour())
        mask = cv2.copyMakeBorder(
            layer.mask.astype(np.uint8), pad_px, pad_px, pad_px, pad_px, border
        ).astype(bool)
        depth = cv2.copyMakeBorder(depth_map, pad_px, pad_px, pad_px, pad_px, border)
        return rgba, mask, depth, content_w_m * grow, content_h_m * grow, pad_px

    def layer_records(self):
        """
        Hình học các lớp **đúng như đã vẽ**, xếp từ xa tới gần.

        Trả chính các ``_LayerRenderable`` mà ``_draw_layers`` dùng, nên không
        có bản sao công thức nào ở phía người đọc. Rỗng khi chưa nạp ảnh.
        """
        return tuple(self._layers)

    @property
    def layers_revision(self) -> int:
        """Số lần bộ lớp được dựng lại; đổi nghĩa là hình học đã khác."""
        return int(self._layers_revision)

    @property
    def layer_hole_stats(self) -> tuple[float, ...]:
        """
        TASK_game_1110 L0 — phần **lỗ** của từng lớp, xếp từ xa tới gần.

        Lỗ của lớp ``r`` = số điểm bị các lớp GẦN HƠN che mà chính nó không
        có, chia cho cả khung. Đó đúng là phần phải tổng hợp lại (xem
        ``fill_layer_disocclusion``), nên nó là thước đo "ảnh này khó lấp cỡ
        nào". Tính ở bước nướng (``bake.hole_fractions``) vì chỉ ở đó mới còn
        mặt nạ; cache giữ lại con số, không giữ mặt nạ.
        """
        return tuple(self._layer_hole_pct)

    @property
    def waiting_for_depth(self) -> bool:
        """
        Đã có ảnh nhưng depth/lớp của đúng ảnh đó chưa xong (PB-051).

        Cùng điều kiện với ``_request_depth`` và ``_poll_depth_mesh``; worker
        hỏng hoặc tắt depth thì không bao giờ có lớp, nên trả ``False``.
        """
        worker = self._bake_worker or self._depth_worker
        if (worker is None or self._current_path is None
                or worker.failed or not self._content.depth_enabled):
            return False
        return self._depth_mesh_key != DepthWorker.media_key(self._current_path)

    def refresh_layers(self) -> bool:
        """
        Dựng lại bộ lớp với ``ContentSettings`` hiện tại, qua đúng đường mà các
        phím nudge/toggle vẫn dùng. ``False`` khi chưa có bộ lớp nào.

        L1: **không** nướng lại. Các phím nudge chỉ đổi z / overscan / relief —
        toàn khoá lúc vẽ — nên dựng lại mesh là đủ. Đổi một khoá thuộc
        ``bake.BAKE_KEYS`` thì phải nạp lại ảnh.
        """
        if self._last_bake is None or self._depth_mesh_key is None:
            return False
        self._apply_bake(self._last_bake, self._depth_mesh_key)
        return True

    def _warn_if_front_layer_too_wide(self, layer_idx, layer, t_mm, quad_w_m):
        """
        Lớp trước kính mà rộng quá giới hạn thì báo một dòng. KHÔNG tự sửa.

        Đo trên **hộp bao của alpha**, không phải bề rộng quad: phần trong suốt
        ở hai mép không chạm viền màn nên không vi phạm cửa sổ.
        """
        if t_mm >= 0.0:
            return
        limit = front_layer_half_width_limit(
            t_mm, self._screen_w_mm, self._content.overscan_e_max_mm,
            self._content.overscan_design_distance_mm,
        )
        cols = np.nonzero((layer.rgba[..., 3] > 0).any(axis=0))[0]
        if cols.size == 0:
            return
        w_px = layer.rgba.shape[1]
        half = max(abs(cols[0] - (w_px - 1) / 2.0), abs(cols[-1] - (w_px - 1) / 2.0))
        half_mm = half / max(w_px - 1, 1) * quad_w_m * 1000.0
        if half_mm > limit:
            logger.warning(
                "PB-010 L%d: lop truoc kinh t=%.0fmm, hop bao alpha nua rong "
                "%.1fmm > gioi han %.1fmm — mep se bi vien man cat",
                layer_idx, t_mm, half_mm, limit,
            )

    def _overscan_scale(
        self,
        t_mm: float,
        quad_w_m: float,
        quad_h_m: float,
        screen_w_mm: float,
        screen_h_mm: float,
    ) -> float:
        """
        Hệ số phóng quad cho một lớp ở độ sâu ``t_mm``.

        Trả về 1.0 khi tắt hoặc khi nội dung đã đủ rộng. Dùng MỘT hệ số chung
        cho cả X và Y (lấy trục cần nhiều hơn) để KHÔNG làm méo tỉ lệ ảnh —
        co giãn X và Y khác nhau sẽ kéo dãn khuôn mặt và đường thẳng.

        Kích thước quad phải TRUYỀN VÀO, không đọc ``self._plane_height``.
        Trường đó phụ thuộc thời điểm ``_resize_plane`` chạy với tỉ lệ ảnh
        thật hay tỉ lệ dự phòng 16:9, nên đọc nó làm hệ số dao động giữa các
        lần chạy: đo được 1.191 và 1.525 cho CÙNG một lớp t=1100 mm, vì chiều
        cao quad lúc 0.96 m (4:3 thật) lúc 0.72 m (16:9 dự phòng).
        """
        e_max = float(self._content.overscan_e_max_mm)
        if e_max <= 0.0 or t_mm <= 1e-6:
            return 1.0
        d_mm = max(float(self._content.overscan_design_distance_mm), 1.0)
        half_w_mm = abs(float(quad_w_m)) * 500.0    # m -> mm, roi /2
        half_h_mm = abs(float(quad_h_m)) * 500.0
        if half_w_mm <= 1e-6 or half_h_mm <= 1e-6:
            return 1.0

        need_w = overscan_need_mm(screen_w_mm, t_mm, e_max, d_mm)
        need_h = overscan_need_mm(screen_h_mm, t_mm, e_max, d_mm)
        return max(1.0, need_w / half_w_mm, need_h / half_h_mm)

    def nudge_plane_z(
        self,
        delta_m: float,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
    ) -> float:
        """
        Dịch cả content plane theo Z. CẦN GẠT CHÍNH.

        ``delta_m`` dương = kéo về phía màn hình (bớt âm). Không dựng lại lớp:
        vị trí mặt phẳng chỉ là ma trận model trong ``draw``, hình học lớp
        không đổi.
        """
        self._position[2] = float(self._position[2] + delta_m)
        self._log_depth_state(
            "plane_z", eye_distance_mm, screen_width_mm, screen_height_mm
        )
        return float(self._position[2])

    def nudge_layer_z_far(
        self,
        delta_m: float,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
    ) -> float:
        """
        Dịch lớp xa nhất trong toạ độ cục bộ. Cần gạt PHỤ.

        ``delta_m`` dương = kéo lớp xa về gần lớp gần (bớt âm). Việc này ĐỔI
        hình học lớp nên phải dựng lại mesh — nhưng từ bản đồ độ sâu đã cache,
        không suy luận lại (cùng đường F7/F8/F10 dùng).
        """
        self._content.layer_z_far_m = float(self._content.layer_z_far_m + delta_m)
        t0 = time.perf_counter()
        rebuilt = self.refresh_layers()
        rebuild_ms = (time.perf_counter() - t0) * 1000.0
        self._log_depth_state(
            "layer_z_far", eye_distance_mm, screen_width_mm, screen_height_mm,
            rebuild_ms=rebuild_ms if rebuilt else None,
        )
        return float(self._content.layer_z_far_m)

    def nudge_overscan(
        self,
        delta_mm: float,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
    ) -> float:
        """
        Chỉnh ``overscan_e_max_mm`` lúc chạy. Kẹp sàn ở 0 (= tắt).

        Đổi hệ số phóng là đổi hình học quad nên phải dựng lại mesh — từ bản
        đồ độ sâu đã cache, không suy luận lại.
        """
        self._content.overscan_e_max_mm = max(
            0.0, float(self._content.overscan_e_max_mm) + float(delta_mm)
        )
        t0 = time.perf_counter()
        rebuilt = self.refresh_layers()
        self._log_depth_state(
            "overscan", eye_distance_mm, screen_width_mm, screen_height_mm,
            rebuild_ms=(time.perf_counter() - t0) * 1000.0 if rebuilt else None,
        )
        return float(self._content.overscan_e_max_mm)

    def _log_depth_state(
        self,
        lever: str,
        eye_distance_mm: float,
        screen_width_mm: float,
        screen_height_mm: float,
        rebuild_ms: float | None = None,
    ) -> None:
        """Ghi trạng thái chiều sâu sau mỗi lần quét, kèm tỉ lệ nội dung lọt khung."""
        plane_z = float(self._position[2])
        z_near = float(self._content.layer_z_near_m)
        z_far = float(self._content.layer_z_far_m)
        count = max(len(self._layers), 1)
        gap_mm = inter_layer_gap_m(z_near, z_far, count) * 1000.0

        # Độ sâu sau mặt phẳng màn hình (z = 0), mm. Dương.
        t_near_mm = abs(plane_z + z_near) * 1000.0
        t_far_mm = abs(plane_z + z_far) * 1000.0
        d_mm = max(float(eye_distance_mm), 1.0)
        f_near = t_near_mm / (d_mm + t_near_mm)
        f_far = t_far_mm / (d_mm + t_far_mm)
        ratio = f_far / f_near if f_near > 1e-9 else float("inf")

        # TỈ LỆ PHỦ KHUNG: cửa sổ màn hình mở một hình nón từ mắt; ở độ sâu t
        # nó cắt ra một dải rộng screen_w * (D + t) / D. Nội dung rộng hơn dải
        # đó thì bị cắt ở mép.
        #
        # CẢNH BÁO ĐỌC SỐ: chỉ số này KHÔNG phải "càng cao càng tốt". 100% nghĩa
        # là toàn bộ ảnh lọt trong khung nhìn — tức CÓ VÙNG TRỐNG quanh nó. Một
        # cửa sổ thật không bao giờ cho thấy mép của thế giới. Chỉ số dùng để
        # quyết định là `void tu` bên dưới, không phải chỉ số này.
        screen_w_mm = max(float(screen_width_mm), 1.0)
        content_w_mm = abs(float(getattr(self, "_plane_width", 0.0))) * 1000.0
        content_h_mm = abs(float(getattr(self, "_plane_height", 0.0))) * 1000.0
        visible_mm = screen_w_mm * (d_mm + t_near_mm) / d_mm
        in_frame = min(1.0, visible_mm / content_w_mm) if content_w_mm > 1e-6 else 0.0

        # NGƯỠNG LỘ VÙNG TRỐNG — đây mới là chỉ số để quyết định.
        #
        # Ở độ sâu t, vùng nhìn thấy trên mặt phẳng nội dung trải
        #     ±(w/2)·(D+t)/D  −  e·t/D
        # với e là độ lệch đầu. Muốn không lộ trống tới ±e thì nửa bề rộng nội
        # dung phải ≥ (w/2)·(D+t)/D + e·t/D. Đảo lại cho nửa bề rộng W hiện có:
        #     e_void = (W − (w/2)·(D+t)/D) · D/t
        #
        # Lớp CÀNG XA càng trượt nhiều nên ngưỡng càng NHỎ — lớp xa lộ trống
        # trước. Tính cả hai trục và lấy trục chạm trước: với ảnh 16:9 thì
        # chiều dọc thường chặt hơn chiều ngang.
        screen_h_mm = max(float(screen_height_mm), 1.0)

        def _void_onset(half_content: float, half_screen: float, t_mm: float) -> float:
            if t_mm <= 1e-6:
                return float("inf")
            return (half_content - half_screen * (d_mm + t_mm) / d_mm) * d_mm / t_mm

        # Nửa bề rộng phải là bề rộng ĐÃ PHÓNG của chính lớp đó, không phải
        # bề rộng ảnh gốc — nếu không, chỉ số này sẽ không thấy overscan và
        # báo ngưỡng cũ mãi mãi.
        onsets = []
        for t_mm in (t_near_mm, t_far_mm):
            k = self._overscan_scale(
                t_mm, content_w_mm * 0.001, content_h_mm * 0.001,
                screen_w_mm, screen_h_mm,
            )
            e_x = _void_onset(content_w_mm * 0.5 * k, screen_w_mm * 0.5, t_mm)
            e_y = _void_onset(content_h_mm * 0.5 * k, screen_h_mm * 0.5, t_mm)
            onsets.append(min(e_x, e_y))
        e_void_mm = min(onsets)          # lop cham truoc quyet dinh

        bezel_z = float(self._content.frame_z_m)
        warn = ""
        if plane_z > bezel_z:
            warn = (
                f"  [CANH BAO: noi dung o {plane_z:+.2f} m da RA TRUOC khung bezel "
                f"{bezel_z:+.2f} m — mat an du cua so]"
            )

        logger.info(
            "Depth sweep [%s]: plane_z=%+.2f m  layer_z=%+.2f..%+.2f m  "
            "gap=%.0f mm | t=%.0f..%.0f mm sau man hinh  "
            "thi sai xa/gan=%.3fx  VOID TU +-%.0f mm  "
            "(anh phu %.0f%% khung nhin; cao KHONG phai tot)%s%s",
            lever, plane_z, z_near, z_far, gap_mm,
            t_near_mm, t_far_mm, ratio, e_void_mm, 100.0 * in_frame,
            "" if rebuild_ms is None else f"  dung lai={rebuild_ms:.0f} ms",
            warn,
        )

    @property
    def mist_enabled(self) -> bool:
        return self._mist_runtime_enabled

    def toggle_atmospheric(self) -> bool:
        """
        Flip atmospheric grading ON/OFF and re-grade cached layers in place.

        Reuses the cached depth map + RGB so no depth re-inference is needed —
        this is the A/B switch used for thesis screenshot comparisons. Returns
        the new enabled state.
        """
        self._content.atmospheric_enabled = not self._content.atmospheric_enabled
        self._rebake("atmospheric")
        logger.info(
            "Atmospheric grading toggled -> %s",
            "ON" if self._content.atmospheric_enabled else "OFF",
        )
        return bool(self._content.atmospheric_enabled)

    def initialize(self) -> None:
        """Create GPU texture after the OpenGL context is active."""
        if self._texture:
            GL.glDeleteTextures(1, [self._texture])
        self._texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        # TASK_game_feel_1118 P2: mipmap — mặt kính/ảnh luôn lớn hơn cỡ
        # hiển thị thật, không mip thì cạnh khối ảnh alias ở cửa sổ nhỏ.
        # RGB thuần (không kênh alpha) nên không cần loang màu.
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        self._video_uniform = GL.glGetUniformLocation(self._shader.program_id, "uVideo")
        if self._video_uniform < 0:
            logger.error("video.frag is missing sampler uniform uVideo")
        if self._content.layer_clustering_enabled and self._layer_shader is None:
            self._layer_shader = ShaderProgram(
                self._shader_dir / "video.vert",
                self._shader_dir / "video_rgba.frag",
            )
            self._layer_video_uniform = GL.glGetUniformLocation(
                self._layer_shader.program_id, "uVideo"
            )
            self._mist_shader = ShaderProgram(
                self._shader_dir / "video.vert",
                self._shader_dir / "mist.frag",
            )
        self._ready = True
        if self._depth_worker is not None:
            self._depth_worker.start()
        if self._bake_worker is not None:
            self._bake_worker.start()
        self._prime_and_upload("initialize")

    def _rebake(self, reason: str) -> None:
        """
        Xin nướng lại bộ lớp vì một khoá **thuộc bước nướng** vừa đổi.

        Không chặn: đi đúng đường của một lượt nạp ảnh mới, nên toast chờ và
        cờ ``waiting_for_depth`` vẫn như cũ. Không có ảnh thì không làm gì.
        """
        if self._bake_worker is None or self._current_path is None:
            self.refresh_layers()
            return
        logger.info("Nuong lai bo lop (%s): khoa cua buoc nuong vua doi", reason)
        self._depth_mesh_key = None
        self._deferred_layers = None
        self._bake_worker.submit(DepthWorker.media_key(self._current_path),
                                 self._current_path, self._content)

    def load_file(self, path: Path, *, strict: bool = True) -> tuple[bool, str]:
        """
        Replace current media with a new image or video file.

        Returns (success, user-facing message).
        """
        kind = detect_content_type(path)
        if kind is None:
            logger.error("Unsupported media type: %s", path)
            return False, "Unsupported file type"

        if not path.exists():
            # strict=False is the start-up default photo, which may be a path
            # from another machine (e.g. a Windows drive letter): the caller
            # falls back, so this is a warning there, not an error.
            (logger.error if strict else logger.warning)(
                "Media file not found: %s", path.resolve())
            return False, f"File not found: {path.name}"

        try:
            self._swap_source(path, kind)
            self._current_path = path.resolve()
            self._depth_mesh_key = None
            self._using_displaced_mesh = False
            self._mist_runtime_enabled = bool(self._content.mist_enabled)
            if self._ready:
                self._release_layers()
            self._invalidate_texture()
            if self._ready:
                if not self._prime_and_upload("load_file"):
                    raise ValueError(f"Texture upload failed for {path.name}")
            logger.info(
                "Content plane loaded: %s (%s, %dx%d)",
                path.name,
                kind,
                self._tex_width,
                self._tex_height,
            )
            notice = static_video_notice(
                kind, bool(self._content.layer_clustering_enabled)
            )
            if notice is not None:
                logger.info("%s", notice)
                self._pending_notice = notice
            return True, f"Loaded: {path.name}"
        except (FileNotFoundError, ValueError, OSError) as exc:
            logger.error("Failed to load media %s: %s", path.resolve(), exc)
            return False, str(exc)

    def take_pending_notice(self) -> str | None:
        """Lấy và xoá thông báo chờ (FR-DEPTH-03); chỉ hiện toast một lần."""
        notice, self._pending_notice = self._pending_notice, None
        return notice

    def take_build_notice(self) -> str | None:
        """
        Lấy và xoá lời nhắn của khung hoãn (PB-076); ``None`` khi không có.

        Người gọi là ``SceneRenderer.render`` ngay sau ``update``, nên toast lên
        màn ở chính khung này — khung cuối trước lúc đóng băng.
        """
        notice, self._build_notice = self._build_notice, None
        return notice

    def update(self, dt_s: float) -> None:
        if self._is_video and self._source is not None:
            rgb = self._source.tick(dt_s)
            if rgb is not None:
                self._upload_rgb(rgb)
        self._poll_depth_mesh()

    def draw(
        self, view_flat: npt.NDArray[np.float32], proj_flat: npt.NDArray[np.float32]
    ) -> None:
        if not self._ready:
            return
        if self._tex_width == 0 or self._tex_height == 0:
            if self._source is not None:
                self._prime_and_upload("draw")
            if self._tex_width == 0:
                return

        model = translation_matrix(
            float(self._position[0]),
            float(self._position[1]),
            float(self._position[2]),
        )
        model_flat = to_column_major_flat(model)

        if self._layers:
            self._draw_layers(model_flat, view_flat, proj_flat)
            return

        GL.glDisable(GL.GL_CULL_FACE)
        self._shader.use()
        self._shader.set_mat4("uModel", model_flat)
        self._shader.set_mat4("uView", view_flat)
        self._shader.set_mat4("uProjection", proj_flat)
        if self._video_uniform < 0:
            logger.error("Shader uniform uVideo not found — texture will not display")
        else:
            GL.glUniform1i(self._video_uniform, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        self._mesh.draw()
        GL.glEnable(GL.GL_CULL_FACE)

    def _draw_layers(
        self,
        model_flat: npt.NDArray[np.float32],
        view_flat: npt.NDArray[np.float32],
        proj_flat: npt.NDArray[np.float32],
    ) -> None:
        assert self._layer_shader is not None
        GL.glDisable(GL.GL_CULL_FACE)
        # Feathered layers overlap at edges: blend back-to-front (painter's) and
        # disable depth writes so soft edges don't punch holes in nearer layers.
        GL.glEnable(GL.GL_BLEND)
        GL.glBlendFunc(GL.GL_SRC_ALPHA, GL.GL_ONE_MINUS_SRC_ALPHA)
        GL.glDepthMask(GL.GL_FALSE)
        self._layer_shader.use()
        self._layer_shader.set_mat4("uModel", model_flat)
        self._layer_shader.set_mat4("uView", view_flat)
        self._layer_shader.set_mat4("uProjection", proj_flat)
        self._layer_shader.set_float(
            "uAlphaThreshold", float(self._content.layer_alpha_threshold)
        )
        if self._layer_video_uniform >= 0:
            GL.glUniform1i(self._layer_video_uniform, 0)
        GL.glActiveTexture(GL.GL_TEXTURE0)
        for idx, layer in enumerate(self._layers):
            GL.glBindTexture(GL.GL_TEXTURE_2D, layer.texture)
            layer.mesh.draw()
            # PB-073: bóng vẽ NGAY SAU mặt nhận, trước lớp gần hơn, nên nó nằm
            # trên mặt nhận mà vẫn bị lớp đổ bóng che đúng chỗ.
            if layer.shadow_mesh is not None:
                GL.glBindTexture(GL.GL_TEXTURE_2D, layer.shadow_texture)
                layer.shadow_mesh.draw()
                GL.glBindTexture(GL.GL_TEXTURE_2D, layer.texture)
            if (
                idx == 0
                and self._mist_runtime_enabled
                and self._mist is not None
                and len(self._layers) >= 2
            ):
                with stage("mist"):
                    self._draw_mist(model_flat, view_flat, proj_flat)
        GL.glDepthMask(GL.GL_TRUE)
        GL.glDisable(GL.GL_BLEND)
        GL.glEnable(GL.GL_CULL_FACE)

    def release(self) -> None:
        if self._depth_worker is not None:
            self._depth_worker.stop()
            self._depth_worker = None
        if self._bake_worker is not None:
            self._bake_worker.stop()
            self._bake_worker = None
        if self._source is not None:
            self._source.close()
            self._source = None
        self._release_layers()
        if self._texture:
            GL.glDeleteTextures(1, [self._texture])
            self._texture = 0
        self._mesh.release()
        if self._layer_shader is not None:
            self._layer_shader.release()
            self._layer_shader = None
        if self._mist_shader is not None:
            self._mist_shader.release()
            self._mist_shader = None
        self._shader.release()

    def _prime_and_upload(self, reason: str) -> bool:
        if self._source is None:
            logger.error("No media source to upload (%s)", reason)
            return False
        if self._is_video and isinstance(self._source, VideoPlayer):
            frame = self._source.prime_first_frame()
        else:
            frame = self._source.tick(0.0)
        if frame is None:
            logger.error(
                "No frame available from %s (%s)",
                self._current_path or "unknown",
                reason,
            )
            return False
        self._upload_rgb(frame)
        self._request_depth(frame)
        logger.info(
            "Texture uploaded (%s): id=%d %dx%d from %s",
            reason,
            self._texture,
            self._tex_width,
            self._tex_height,
            self._current_path.name if self._current_path else "?",
        )
        return self._tex_width > 0

    def _request_depth(self, rgb: npt.NDArray[np.uint8]) -> None:
        if not self._content.depth_enabled or self._current_path is None:
            return
        key = DepthWorker.media_key(self._current_path)
        if self._depth_mesh_key == key:
            return
        self._last_depth_rgb = rgb.copy()
        if self._bake_worker is not None:
            if not self._bake_worker.failed:
                self._bake_worker.submit(key, self._current_path, self._content)
            return
        if self._depth_worker is None or self._depth_worker.failed:
            return
        self._depth_worker.submit(key, rgb)

    def _poll_depth_mesh(self) -> None:
        if self._bake_worker is not None:
            self._poll_bake()
            return
        if (
            self._depth_worker is None
            or self._current_path is None
            or self._depth_worker.failed
        ):
            return
        key = DepthWorker.media_key(self._current_path)
        if self._depth_mesh_key == key:
            return
        depth = self._depth_worker.pop_ready(key)
        if depth is None:
            return
        self._apply_depth_mesh(depth, key)

    def _poll_bake(self) -> None:
        """
        Nhịp của lượt nướng ngoài tiến trình. Không bao giờ chặn luồng GL.

        PB-076 vẫn giữ: hoãn ĐÚNG MỘT khung trước khi dựng mesh, để khung mang
        toast được vẽ và swap trước. Lần đóng băng còn lại chỉ là phần dựng
        texture + mesh (hàng trăm ms), không còn 9–30 s tách lớp.
        """
        assert self._bake_worker is not None
        if self._current_path is None or self._bake_worker.failed:
            return
        key = DepthWorker.media_key(self._current_path)
        if self._depth_mesh_key == key:
            return
        if self._deferred_layers is not None:
            result, deferred_key = self._deferred_layers
            self._deferred_layers = None
            if deferred_key == key:
                self._apply_bake(result, deferred_key)
                return
            # Ảnh đã đổi trong đúng một khung đó: bỏ bản hoãn, đi lại từ đầu.
        self._bake_worker.poll()
        result = self._bake_worker.pop_ready(key)
        if result is None:
            return
        self._deferred_layers = (result, key)
        self._build_notice = LAYER_BUILD_TOAST
        logger.info("Layer build deferred one frame (PB-076): toast %r",
                    LAYER_BUILD_TOAST)

    def _apply_depth_mesh(self, depth_map: npt.NDArray[np.float32], key: str) -> None:
        width = self._plane_width
        height = self._plane_height if self._plane_height > 0 else width / (
            self._tex_width / max(self._tex_height, 1)
        )
        resolution = max(8, int(self._content.depth_mesh_resolution))
        mesh_data = create_displaced_vertical_plane_uv(
            width,
            height,
            depth_map,
            self._content.depth_scale_meters,
            resolution,
            resolution,
            z_base=0.0,
        )
        self._mesh.release()
        self._mesh = TexturedMesh(mesh_data)
        self._depth_mesh_key = key
        self._using_displaced_mesh = True
        logger.info(
            "Depth-displaced mesh applied: %dx%d grid, scale=%.3f m, %d triangles",
            resolution + 1,
            resolution + 1,
            self._content.depth_scale_meters,
            mesh_data.vertex_count // 3,
        )

    def _apply_bake(self, result, key: str) -> None:
        """
        Dựng texture + mesh từ một bộ lớp **đã nướng xong** (L1).

        Phần nặng (depth, cụm, lấp khe) đã chạy ở tiến trình con; ở đây chỉ còn
        việc của luồng GL: gán z, dựng quad, nạp texture. Đổi ``layer_z_far_m``
        thì chỉ hàm này chạy lại — đó là thứ làm lượt quét z của L4 chạy nổi.
        """
        if self._layer_shader is None:
            return
        depth_map = result.depth
        t_total0 = time.perf_counter()
        image_name = Path(key).name if key else "unknown"
        depth_ms = float(result.timing_ms.get("depth", 0.0))

        width = self._plane_width
        height = self._plane_height if self._plane_height > 0 else width / (
            self._tex_width / max(self._tex_height, 1)
        )
        n_clusters = int(result.k)
        k_ms = 0.0
        self._adaptive_k_report = None
        self._last_bake = result

        layers = result.layers
        count = len(layers)
        for rank, layer in enumerate(layers):
            layer.z_m = assign_layer_z(rank, count,
                                       self._content.layer_z_near_m,
                                       self._content.layer_z_far_m)
        semantic_ms = float(result.timing_ms.get("semantic_adjust", 0.0))
        disocclusion_ms = float(result.timing_ms.get("disocclusion_fill", 0.0))
        clustering_ms = float(result.timing_ms.get("clustering", 0.0))
        self._last_depth_map = depth_map
        self._layer_hole_pct = [float(layer.hole_pct) for layer in layers]
        self._semantic_report = None
        self._layer_reports = [
            layer.grade_report for layer in layers if layer.grade_report is not None
        ]
        self._release_layers()
        layer_z_values: list[float] = []
        layer_gap = inter_layer_gap_m(
            self._content.layer_z_near_m,
            self._content.layer_z_far_m,
            len(layers),
        )
        resolution = max(8, int(self._content.depth_mesh_resolution))
        use_relief = bool(self._relief_runtime_enabled)
        t_mesh0 = time.perf_counter()
        overscan_lines: list[str] = []
        pad_mode = str(self._content.overscan_mode).lower() == "pad"
        shift_front = str(self._content.relief_front_policy).lower() == "shift"
        t_plane_mm = SCREEN_Z_MM - float(self._position[2]) * 1000.0
        design_eye = tuple(float(v) for v in self._content.overscan_design_eye_mm)
        design_d = float(self._content.overscan_design_distance_mm)
        for layer_idx, layer in enumerate(layers):
            # OVERSCAN: phóng quad của TỪNG lớp theo độ sâu của chính nó. Lớp
            # xa trượt nhiều hơn nên cần rộng hơn — một hệ số chung sẽ hoặc
            # thiếu cho lớp xa, hoặc thừa cho lớp gần. UV giữ nguyên 0..1 nên
            # ảnh bị phóng to, tức mép bố cục bị cắt ở vị trí chính giữa. Đó
            # là cái giá đã biết; Phase 3 sẽ nới ảnh ra nếu thấy khó chịu.
            # t CÓ DẤU: dương = sau kính, âm = trước kính. Dạng cũ dùng
            # ``abs()`` nên lớp nằm TRƯỚC kính bị phóng to y như nằm sau —
            # preset (b) đo 2026-10-13 cho k = 2.058 ở lớp t = −100 mm, trong
            # khi ``measure_parallax.py`` (t có dấu) trả 1.000. Hai bên lệch
            # nhau ở mọi lớp t < 0. ``_overscan_scale`` vẫn trả 1.0 khi t ≤ 0.
            # width/height la kich thuoc quad THAT da tinh o tren tu tex dims,
            # khong phai self._plane_height co the con la 16:9 du phong.
            geom = layer_geometry(
                self._overscan_scale, self._content, self._position[2], layer.z_m,
                width, height, self._screen_w_mm, self._screen_h_mm,
            )
            t_mm = geom.t_mm
            if pad_mode:
                # Margin phải tính theo đỉnh SÂU NHẤT của lớp, không phải t gốc,
                # nếu không relief phía sau kính mở ra khe trống mỏng ở mép màn.
                # Dùng chặn trên giải tích ``t + clamp`` thay vì đo đỉnh thật:
                # đo thật sẽ vòng tròn (pad đổi cách lấy mẫu UV nên đổi luôn
                # relief), còn chặn này không bao giờ thiếu và khỏi dựng mesh
                # thăm dò. Rộng hơn mức tối thiểu đúng bằng clamp.
                t_margin = t_mm + (
                    RELIEF_CLAMP_GAP_FRACTION * layer_gap * 1000.0 if use_relief else 0.0
                )
                rgba_l, mask_l, depth_l, width_s, height_s, pad_px = self._pad_layer(
                    layer, depth_map, t_mm, t_plane_mm, width, height, t_margin
                )
                scale = width_s / max(width, 1e-9)
                note = f"pad {pad_px}px"
            else:
                rgba_l, mask_l, depth_l, pad_px = (
                    layer.rgba, layer.mask, depth_map, 0
                )
                scale = geom.scale
                width_s, height_s = geom.quad_w_m, geom.quad_h_m
                note = "scale"
            # Dựng texture SAU khi đã chốt rgba: ở chế độ pad là mảng đã đệm.
            # Dựng sẵn từ ``layer.rgba`` ở đầu vòng rồi dựng lại sẽ rò một
            # texture GL mỗi lớp mỗi lần nạp ảnh.
            # TASK_game_feel_fix_1118 Fix 2: ``apply_bleed=False`` — xem
            # docstring ``_create_rgba_texture``.
            texture = self._create_rgba_texture(rgba_l, apply_bleed=False)
            self._warn_if_front_layer_too_wide(layer_idx, layer, t_mm, width_s)
            overscan_lines.append(
                f"L{layer_idx} t={t_mm:.0f}mm {note} x{scale:.3f} "
                f"({width * 1000:.0f}x{height * 1000:.0f}"
                f"->{width_s * 1000:.0f}x{height_s * 1000:.0f}mm)"
            )
            shift_offset = 0.0
            if use_relief:
                mesh_data = create_layer_relief_vertical_plane_uv(
                    width_s,
                    height_s,
                    depth_l,
                    mask_l,
                    z_base=layer.z_m,
                    depth_lo=layer.depth_lo,
                    depth_hi=layer.depth_hi,
                    mean_depth=layer.mean_depth,
                    depth_norm=layer.depth_norm,
                    inter_layer_gap_m=layer_gap,
                    relief_amplitude=float(self._content.relief_amplitude),
                    relief_far_falloff=float(self._content.relief_far_falloff),
                    relief_smooth_sigma=float(self._content.relief_smooth_sigma),
                    segments_x=resolution,
                    segments_y=resolution,
                    layer_index=layer_idx + 1,
                    layer_count=len(layers),
                )
                if shift_front:
                    # Lùi CẢ lớp lại đúng bằng phần nhô ra lớn nhất. Tịnh tiến
                    # thuần nên hình dạng relief không đổi; chỉ cái nền nó ngồi
                    # lên lùi lại, và lớp "ghim" mới thật sự đứng yên.
                    verts = mesh_data.vertices.reshape(-1, TexturedMesh.STRIDE)
                    shift_offset = relief_front_offset_m(
                        float(self._position[2]) + verts[:, 2]
                    )
                    if shift_offset > 0.0:
                        verts[:, 2] -= np.float32(shift_offset)
            else:
                mesh_data = create_vertical_plane_uv(
                    width_s, height_s, z=layer.z_m
                )
            # Dời tâm quad theo mắt thiết kế. Tịnh tiến thuần trong XY, sau khi
            # mesh đã dựng, nên dùng chung được cho cả hai nhánh và không đụng
            # tới relief hay hệ số overscan.
            if geom.centre_x_m or geom.centre_y_m:
                verts = mesh_data.vertices.reshape(-1, TexturedMesh.STRIDE)
                verts[:, 0] += np.float32(geom.centre_x_m)
                verts[:, 1] += np.float32(geom.centre_y_m)
            mesh = TexturedMesh(mesh_data)
            # z thật của đỉnh sâu nhất / gần nhất, đọc thẳng từ mesh đã dựng —
            # chính xác hơn chặn trên 0.4 x gap, và không tốn gì thêm.
            vz = mesh_data.vertices.reshape(-1, TexturedMesh.STRIDE)[:, 2]
            pz = float(self._position[2])
            self._layers.append(
                _LayerRenderable(
                    texture=texture, mesh=mesh, z_m=layer.z_m,
                    alpha=np.ascontiguousarray(rgba_l[..., 3]),
                    quad_w_m=width_s, quad_h_m=height_s,
                    centre_x_m=geom.centre_x_m, centre_y_m=geom.centre_y_m,
                    z_world_m=pz + float(layer.z_m) - shift_offset,
                    relief_z_min_m=pz + float(vz.min()),
                    relief_z_max_m=pz + float(vz.max()),
                )
            )
            layer_z_values.append(layer.z_m)
        if bool(self._content.layer_shadow):
            self._build_layer_shadows()
        mesh_ms = (time.perf_counter() - t_mesh0) * 1000.0
        if overscan_lines:
            e_max = float(self._content.overscan_e_max_mm)
            logger.info(
                "Overscan e_max=%.0f mm @ D=%.0f mm: %s",
                e_max,
                float(self._content.overscan_design_distance_mm),
                "  ".join(overscan_lines) if e_max > 0.0 else "TAT",
            )
        self._release_mist()
        if len(layer_z_values) >= 2 and bool(self._content.mist_enabled):
            mist_z = (layer_z_values[0] + layer_z_values[1]) * 0.5
            self._mist = self._build_mist_renderable(
                width, height, mist_z, result.mist_color
            )
        self._depth_mesh_key = key
        self._using_displaced_mesh = True
        self._layers_revision += 1
        total_ms = (time.perf_counter() - t_total0) * 1000.0 + depth_ms
        logger.info(
            "Load timing [%s]: depth_inference=%.1fms | k_selection=%.1fms | "
            "semantic_adjust=%.1fms | clustering=%.1fms | disocclusion=%.1fms | "
            "mesh_build=%.1fms | total=%.1fms",
            image_name,
            depth_ms,
            k_ms,
            semantic_ms,
            clustering_ms,
            disocclusion_ms,
            mesh_ms,
            total_ms,
        )
        logger.info(
            "Depth layers applied: %d clusters (K=%d), atmospheric=%s, relief=%s, Z %.3f..%.3f m, "
            "coverage %s",
            len(layers),
            n_clusters,
            "ON" if self._content.atmospheric_enabled else "OFF",
            "ON" if use_relief else "OFF",
            layers[0].z_m if layers else 0.0,
            layers[-1].z_m if layers else 0.0,
            ", ".join(f"{lyr.coverage:.0%}" for lyr in layers),
        )

    def _create_rgba_texture(self, rgba: npt.NDArray[np.uint8], *,
                             apply_bleed: bool = True) -> int:
        """
        ``apply_bleed=False`` (TASK_game_feel_fix_1118 Fix 2): photo LAYER
        arrays never need it. ``extract_layer_rgba`` keeps the full
        graded-photo RGB at every pixel, even where alpha is 0 — "RGB is
        kept everywhere (not zeroed outside the cluster) so the soft edge
        blends the true image colors" (its own docstring) — and
        ``_pad_layer``'s border is ``cv2.BORDER_REPLICATE`` (copies the
        real edge pixel outward), never zero-fill. So a layer's RGB under
        a transparent pixel was never black to begin with; the bug
        ``bleed_transparent`` exists for (straight-alpha PNGs whose
        AUTHORING pipeline left RGB=0 under alpha=0) does not apply here.
        Skipping it removes a 6-iteration box-blur pass over a full-
        resolution array from the MAIN RENDER THREAD (``_apply_bake``,
        called from ``update()``'s per-frame tick path — every OTHER
        caller already does its own GL-only work, per that method's own
        docstring; this blur broke that invariant). The layer-SHADOW
        texture (``_build_layer_shadows``) is a different, small,
        procedurally generated array — not verified safe, keeps bleeding
        (the default).
        """
        # TASK_game_feel_1118 P2: loang màu TRƯỚC ``flipud`` — loang không
        # quan tâm hướng trục, và làm trên mảng gốc (liên tục trong bộ
        # nhớ) rẻ hơn làm sau khi lật.
        bled = bleed_transparent(rgba) if apply_bleed else rgba
        data = np.ascontiguousarray(np.flipud(bled), dtype=np.uint8)
        h, w = data.shape[:2]
        texture = int(GL.glGenTextures(1))
        GL.glBindTexture(GL.GL_TEXTURE_2D, texture)
        # Mipmap: lớp ảnh/bóng co lại rõ rệt theo khoảng cách camera ảo,
        # viền cắt lớp (alpha nhị phân, đã loang màu ở trên) alias mạnh
        # nhất trong cả app nếu không mip.
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER,
                           GL.GL_LINEAR_MIPMAP_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glTexImage2D(
            GL.GL_TEXTURE_2D, 0, GL.GL_RGBA, w, h, 0,
            GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, data,
        )
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)
        return texture

    def _build_layer_shadows(self) -> None:
        """
        PB-073: mỗi lớp nhận bóng của lớp GẦN HƠN kế tiếp.

        Alpha của vật đổ bóng phải đổi sang hệ toạ độ texture của mặt nhận
        trước: overscan phóng mỗi lớp một hệ số nên hai lưới không trùng nhau.
        """
        for idx in range(len(self._layers) - 1):
            receiver, caster = self._layers[idx], self._layers[idx + 1]
            if receiver.alpha is None or caster.alpha is None:
                continue
            moved = resample_to(
                caster.alpha.astype(np.float32), caster.quad_w_m,
                caster.centre_x_m, receiver.alpha.shape, receiver.quad_w_m,
                receiver.centre_x_m, src_h_m=caster.quad_h_m,
                src_cy_m=caster.centre_y_m, dst_h_m=receiver.quad_h_m,
                dst_cy_m=receiver.centre_y_m,
            )
            offset_px, sigma_px = shadow_geometry(
                receiver.alpha.shape[1], caster.z_m - receiver.z_m,
                float(self._content.shadow_offset_frac),
                float(self._content.shadow_blur_frac),
            )
            rgba = shadow_rgba(
                np.clip(moved, 0, 255).astype(np.uint8), receiver.alpha,
                offset_px=offset_px, sigma_px=sigma_px,
                angle_deg=float(self._content.shadow_angle_deg),
                opacity=float(self._content.shadow_opacity),
            )
            mesh_data = create_vertical_plane_uv(
                receiver.quad_w_m, receiver.quad_h_m,
                z=receiver.z_m + SHADOW_LIFT_M,
            )
            if receiver.centre_x_m or receiver.centre_y_m:
                verts = mesh_data.vertices.reshape(-1, TexturedMesh.STRIDE)
                verts[:, 0] += np.float32(receiver.centre_x_m)
                verts[:, 1] += np.float32(receiver.centre_y_m)
            receiver.shadow_texture = self._create_rgba_texture(rgba)
            receiver.shadow_mesh = TexturedMesh(mesh_data)
            logger.info(
                "Layer shadow L%d <- L%d: doi %.1f px, nhoe sigma %.1f px, "
                "duc %.0f%%", idx, idx + 1, offset_px, sigma_px,
                float(self._content.shadow_opacity) * 100.0,
            )

    def _release_layers(self) -> None:
        for layer in self._layers:
            if layer.texture:
                GL.glDeleteTextures(1, [layer.texture])
            layer.mesh.release()
            if layer.shadow_texture:
                GL.glDeleteTextures(1, [layer.shadow_texture])
            if layer.shadow_mesh is not None:
                layer.shadow_mesh.release()
        self._layers = []
        self._release_mist()

    def _release_mist(self) -> None:
        if self._mist is not None:
            self._mist.mesh.release()
            self._mist = None

    def _mist_quad_size(self, mist_z: float) -> tuple[float, float]:
        reference_screen_distance = max(abs(float(self._position[2])), 1e-6)
        depth_ratio = 1.0 + abs(mist_z) / reference_screen_distance
        margin = float(self._content.mist_margin)
        return self._plane_width * depth_ratio * margin, self._plane_height * depth_ratio * margin

    def _build_mist_renderable(
        self,
        plane_width: float,
        plane_height: float,
        mist_z: float,
        mist_color: tuple[float, float, float],
    ) -> _MistRenderable:
        mist_w, mist_h = self._mist_quad_size(mist_z)
        opacity = float(np.clip(self._content.mist_opacity, 0.15, 0.30))
        mesh = TexturedMesh(create_vertical_plane_uv(mist_w, mist_h, z=mist_z))
        logger.info(
            "Mist quad: z=%.3f size=%.2fx%.2f m color=(%.2f,%.2f,%.2f) opacity=%.2f",
            mist_z,
            mist_w,
            mist_h,
            mist_color[0],
            mist_color[1],
            mist_color[2],
            opacity,
        )
        return _MistRenderable(
            mesh=mesh,
            color=mist_color,
            opacity=opacity,
            z_m=mist_z,
        )

    def _draw_mist(
        self,
        model_flat: npt.NDArray[np.float32],
        view_flat: npt.NDArray[np.float32],
        proj_flat: npt.NDArray[np.float32],
    ) -> None:
        if self._mist is None or self._mist_shader is None:
            return
        self._mist_shader.use()
        self._mist_shader.set_mat4("uModel", model_flat)
        self._mist_shader.set_mat4("uView", view_flat)
        self._mist_shader.set_mat4("uProjection", proj_flat)
        color = self._mist.color
        self._mist_shader.set_vec3("mistColor", color[0], color[1], color[2])
        self._mist_shader.set_float("mistOpacity", self._mist.opacity)
        self._mist.mesh.draw()

    def _swap_source(self, path: Path, kind: str) -> None:
        if self._source is not None:
            self._source.close()
            self._source = None

        if kind == "video":
            player = VideoPlayer(path, loop=self._content.loop_video)
            if player.is_procedural:
                raise ValueError(f"Failed to open video: {path.name}")
            self._source = player
            self._is_video = True
        else:
            self._source = ImageSource(path)
            self._is_video = False

        self._current_path = path.resolve()
        self._resize_plane(self._source.aspect_ratio)

    def _resize_plane(self, aspect: float) -> None:
        width = self._content.plane_width_m
        height = self._content.plane_height_m
        if height <= 0.0:
            height = width / max(aspect, 1e-6)
        self._plane_width = width
        self._plane_height = height
        self._mesh.release()
        self._mesh = self._build_flat_mesh(width / height)
        self._depth_mesh_key = None
        self._using_displaced_mesh = False

    def _build_flat_mesh(self, aspect: float) -> TexturedMesh:
        width = self._plane_width
        height = self._plane_height if self._plane_height > 0 else width / aspect
        return TexturedMesh(create_vertical_plane_uv(width, height, z=0.0))

    def _invalidate_texture(self) -> None:
        self._tex_width = 0
        self._tex_height = 0

    def _upload_rgb(self, rgb: npt.NDArray[np.uint8]) -> None:
        if not self._ready or not self._texture:
            logger.warning("Skipping texture upload — GPU texture not initialized")
            return
        if rgb.ndim != 3 or rgb.shape[2] != 3:
            logger.error("Expected RGB uint8 image, got shape %s", rgb.shape)
            return

        rgb = np.ascontiguousarray(np.flipud(rgb), dtype=np.uint8)
        h, w = rgb.shape[:2]
        if w < 1 or h < 1:
            logger.error("Invalid texture dimensions: %dx%d", w, h)
            return

        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self._texture)
        if w != self._tex_width or h != self._tex_height:
            GL.glTexImage2D(
                GL.GL_TEXTURE_2D, 0, GL.GL_RGB, w, h, 0,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb,
            )
            self._tex_width = w
            self._tex_height = h
        else:
            GL.glTexSubImage2D(
                GL.GL_TEXTURE_2D, 0, 0, 0, w, h,
                GL.GL_RGB, GL.GL_UNSIGNED_BYTE, rgb,
            )
        # TASK_game_feel_1118 P2: bậc mip phải theo kịp nội dung mới —
        # nhánh glTexSubImage2D (video đang phát) cũng cần, không chỉ
        # nhánh cấp phát lại.
        GL.glGenerateMipmap(GL.GL_TEXTURE_2D)


# Backward-compatible alias
VideoPlaneRenderer = ContentPlaneRenderer
