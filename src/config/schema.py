"""Application configuration schema and persistence."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

#: Sentinel for "key absent", so a legitimate ``None``/``0``/``False`` in JSON
#: is not mistaken for a missing key.
_MISSING = object()

#: Legacy key fallbacks for :class:`ContentSettings`: ``field -> older key``.
#: Used only when the field's own name is absent from the payload.
CONTENT_KEY_ALIASES: dict[str, str] = {
    "rim_light_intensity": "layer_rim_intensity",
}


def _coerce_to_default_type(value: Any, default: Any, key: str) -> Any:
    """
    Coerce a JSON value to the type of the field's default.

    Every :class:`ContentSettings` field has a default, so its runtime type is
    an exact and always-available description of what the field expects — no
    ``typing.get_type_hints`` needed, which matters because this module uses
    postponed annotations (``from __future__ import annotations``) and would
    otherwise see plain strings.
    """
    # bool must be tested before int: bool is a subclass of int.
    if isinstance(default, bool):
        return bool(value)
    if isinstance(default, int):
        return int(value)
    if isinstance(default, float):
        return float(value)
    if isinstance(default, str):
        return str(value)
    if isinstance(default, tuple):
        if isinstance(value, (str, bytes)):
            raise TypeError(f"expected {len(default)} numbers, got a string")
        coerced = tuple(float(v) for v in value)
        if len(coerced) != len(default):
            raise ValueError(
                f"expected {len(default)} numbers, got {len(coerced)}"
            )
        return coerced
    raise TypeError(f"unsupported default type {type(default).__name__} for {key!r}")


@dataclass
class CameraSettings:
    """Webcam capture parameters."""

    device_id: int = 0
    width: int = 1280
    height: int = 720
    fps: int = 60
    buffer_size: int = 1
    fourcc: str = "MJPG"
    async_capture: bool = True
    queue_size: int = 2


@dataclass
class ScreenSettings:
    """Physical monitor dimensions and pixel resolution."""

    width_mm: float = 531.0
    height_mm: float = 299.0
    width_px: int = 1920
    height_px: int = 1080
    monitor_index: int = 0
    center_x_m: float = 0.0
    center_y_m: float = 0.0
    center_z_m: float = 0.0


@dataclass
class WebcamMountSettings:
    """Webcam position relative to screen center (millimeters)."""

    offset_x_mm: float = 0.0
    offset_y_mm: float = 0.0
    offset_z_mm: float = 0.0


@dataclass
class UserSettings:
    """User-specific calibration values."""

    ipd_mm: float = 63.0
    viewing_distance_mm: float = 600.0


@dataclass
class FilterSettings:
    """Temporal filtering and motion prediction."""

    type: str = "one_euro"
    min_cutoff: float = 1.0
    beta: float = 0.007
    ema_alpha: float = 0.35
    process_noise: float = 1.0
    measurement_noise: float = 4.0
    prediction_enabled: bool = True
    prediction_frames: float = 1.5
    max_prediction_ms: float = 25.0
    adaptive_prediction: bool = True
    min_velocity_mm_s: float = 5.0

    # The three knobs below are wired end to end but ship DISABLED (0.0), and
    # assets/config/default.json deliberately does not set them. This is a
    # decision, not an oversight -- do not "fix" it by turning them on.
    #
    # They were added while chasing intermittent frame jerk, on the theory that
    # a dt swinging over 0.5-50 ms had to be clamped. It did not: the swing was
    # a symptom of differentiating over the tracking-loop interval instead of
    # the camera capture interval. The real fix is in engine/tracking_worker.py,
    # which now uses the packet's own capture timestamp and skips the velocity
    # update on a repeated pose. Measured at rest, p99 pixel displacement went
    # 3.41 px -> 1.14 px, with moving latency unchanged at +0.0 ms.
    #
    # Clamping dt on top of that would suppress the readout of a cause that is
    # already gone, and would hide the next timestamp bug instead of surfacing
    # it. Keep them at 0.0 so the dt that reaches _update_velocity is the real
    # one.
    #
    # When to reconsider: if rest p99 pixel displacement climbs back above
    # ~2 px, audit the dt SOURCE first (capture timestamps, frame_index
    # dedup, camera FPS vs track_max_fps). Only if the source checks out
    # should these knobs be considered, and then with a before/after p99
    # measurement, not by feel.
    velocity_smoothing: float = 0.0
    velocity_dt_min_s: float = 0.0
    velocity_dt_max_s: float = 0.0


@dataclass
class OptimizationSettings:
    """Performance tuning for latency and throughput."""

    target_latency_ms: float = 20.0
    track_max_fps: float = 120.0
    tracking_downscale: bool = True
    fast_solve_pnp: bool = True
    cache_projection: bool = True
    render_extrapolation: bool = True
    latency_ema_alpha: float = 0.15


@dataclass
class TrackingSettings:
    """MediaPipe tracking quality."""

    model_complexity: int = 1
    min_detection_confidence: float = 0.5
    min_tracking_confidence: float = 0.5
    simulation_fallback: bool = True
    model_path: str = "assets/models/face_landmarker.task"


@dataclass
class AdaptiveQualitySettings:
    """Automatic quality scaling when FPS drops."""

    enabled: bool = True
    target_fps: float = 60.0
    min_fps: float = 45.0
    quality_levels: list[int] = field(default_factory=lambda: [100, 75, 50, 35])


@dataclass
class RendererSettings:
    """OpenGL renderer and GPU camera pipeline."""

    vsync: bool = True
    triple_buffer: bool = True
    gpu_camera_uniforms: bool = True
    shader_path: str = "assets/shaders"
    window_width: int = 1280
    window_height: int = 720
    window_title: str = "HCPD — Virtual Room"
    msaa_samples: int = 4
    # Toàn màn hình thật trên màn chính, giữ nguyên chế độ hình đang chạy
    # (không đổi độ phân giải). Mặc định tắt để không đổi hành vi hiện có.
    fullscreen: bool = False


@dataclass
class UiSettings:
    """On-screen HUD elements (webcam preview, etc.)."""

    webcam_preview_enabled: bool = True
    webcam_preview_corner: str = "bottom-right"
    webcam_preview_scale: float = 0.18
    webcam_preview_margin_px: int = 16
    webcam_preview_mirror: bool = True
    webcam_preview_landmarks: bool = True


@dataclass
class ContentSettings:
    """Scene content — video plane through the monitor window."""

    enabled: bool = True
    mode: str = "video"
    content_path: str = ""
    content_type: str = "auto"
    video_path: str = "assets/video/demo.mp4"
    loop_video: bool = True
    plane_position_m: tuple[float, float, float] = (0.0, 0.0, -0.75)
    plane_width_m: float = 1.28
    plane_height_m: float = 0.0
    frame_z_m: float = -0.58
    frame_bezel_m: float = 0.05
    frame_color: tuple[float, float, float] = (0.18, 0.17, 0.16)
    mid_layer_z_m: float = -1.5
    mid_layer_width_m: float = 0.9
    mid_layer_height_m: float = 1.4
    mid_layer_offset_x_m: float = 1.1
    mid_layer_color_a: tuple[float, float, float] = (0.10, 0.12, 0.18)
    mid_layer_color_b: tuple[float, float, float] = (0.16, 0.10, 0.14)
    backdrop_z_m: float = -2.8
    backdrop_width_m: float = 5.0
    backdrop_height_m: float = 3.2
    backdrop_color: tuple[float, float, float] = (0.06, 0.06, 0.08)
    void_clear_color: tuple[float, float, float] = (0.04, 0.04, 0.05)
    depth_enabled: bool = True
    depth_scale_meters: float = 0.15
    depth_mesh_resolution: int = 96
    depth_model: str = "depth-anything-small"
    # Engine chạy inference depth: "torch" (mặc định, giữ nguyên hành vi cũ)
    # hoặc "onnxruntime" (nạp file .onnx xuất ở PB-060). Thiếu file .onnx thì
    # DepthEstimator tự rơi về torch, không làm chết app.
    depth_backend: str = "torch"
    # Discrete depth-layer clustering (alternative to per-pixel displacement).
    layer_clustering_enabled: bool = False
    # Cách gán nhãn lớp: "kmeans" (mặc định, giữ nguyên hành vi cũ) hoặc
    # "kmeans_vote" (PB-070b/SPIKE-15: bầu theo vùng P10 sau k-means, KHÔNG
    # cụm lại). Mặc định phải là "kmeans" — người dùng chưa chọn bằng mắt.
    layer_method: str = "kmeans"
    layer_count: int = 3
    adaptive_k_enabled: bool = True
    adaptive_k_min: int = 2
    adaptive_k_max: int = 4
    k_policy: str = "gvf"
    gvf_threshold: float = 0.90
    log_silhouette_comparison: bool = False
    layer_z_near_m: float = 0.0
    layer_z_far_m: float = -0.35

    # OVERSCAN THEO TỪNG LỚP — chặn người xem nhìn thấy mép ảnh.
    #
    # Ảnh là hữu hạn. Dịch đầu đủ xa thì khung nhìn trượt ra ngoài mép nó và lộ
    # vùng trống. Một cửa sổ thật không bao giờ cho thấy mép của thế giới, nên
    # lỗi này phá ảo giác ngay lập tức.
    #
    # Ở độ sâu t, nửa bề rộng nội dung cần có để không lộ trống tới ±e_max là
    #     (w/2)·(D+t)/D + e_max·t/D
    # Lớp CÀNG XA trượt càng nhiều nên cần RỘNG hơn — vì vậy hệ số phải tính
    # THEO TỪNG LỚP, không phải một hệ số chung. Đo trên cấu hình hiện tại:
    # lớp xa cần 1.19×, lớp gần chỉ cần 1.04×.
    #
    # 0.0 = TẮT hẳn (giữ hành vi cũ, quad đúng bằng plane_width_m).
    overscan_e_max_mm: float = 150.0

    # Khoảng cách xem dùng để TÍNH hệ số lúc dựng mesh. Phải chốt một giá trị
    # vì mesh dựng một lần lúc nạp ảnh, còn người xem thì di chuyển. Ngồi GẦN
    # hơn cần overscan NHIỀU hơn (t/D lớn hơn), nên đặt giá trị này thấp hơn
    # khoảng cách ngồi thật là phía an toàn. Dòng log quét vẫn báo biên thật
    # theo khoảng cách LIVE, nên biết ngay khi không đủ.
    overscan_design_distance_mm: float = 600.0
    # Cách nới quad cho lớp xa (PB-045 L2b).
    #   "scale" — phóng to quad, UV giữ 0..1, nên ẢNH bị phóng theo và mép bố
    #             cục bị cắt. Ở mắt giữa lớp xa trông to hơn ảnh M = k·D/(D+t)
    #             lần, nên lỗ không còn nằm đúng sau vật. Hành vi cũ, mặc định.
    #   "pad"   — giữ nguyên tỉ lệ nội dung và ĐỆM thêm viền quanh ảnh. Lớp
    #             khớp vị trí với mặt phẳng nhìn từ mắt thiết kế; chỉ dải đệm
    #             là phần tổng hợp thêm.
    overscan_mode: str = "scale"
    # Relief đẩy đỉnh của lớp ghim ra TRƯỚC kính (đo t21: tới +24.6 mm, 6.8%
    # số đỉnh, cực đại nằm ở mép ảnh). Khi đó lớp "ghim" không còn đứng yên và
    # nó phá cửa sổ ở mép.
    #   "keep"  — giữ nguyên, hành vi cũ, mặc định.
    #   "shift" — lùi cả lớp lại đúng bằng phần nhô ra lớn nhất, giữ nguyên
    #             hình dạng relief.
    relief_front_policy: str = "keep"
    # Vị trí mắt NGHỈ, mm trong hệ HCPD (X phải, Y lên), dùng làm mắt thiết kế.
    # Mặc định [0, 0] = tâm màn = hành vi cũ, không đổi một byte nào.
    #
    # Người xem ngồi trước laptop thì mắt hiếm khi ở ngang tâm màn: nắp ngả ra
    # sau là mắt nằm cao hơn. Khi đó mọi lớp đã lệch sẵn lúc NGHỈ, và mép dưới
    # hở nền. Dời tâm quad mỗi lớp đi -(x0, y0)·t/Dd thì lớp khớp vị trí ở đúng
    # tư thế ngồi thật thay vì ở tâm màn.
    overscan_design_eye_mm: tuple[float, float] = (0.0, 0.0)
    # --- Part C: tro choi tim do vat (PB-015). Khoa phang, tien to game_. ---
    # Ban kinh vat an, mm. [GIA DINH 12 mm — nguoi dung co the doi]
    game_prop_radius_mm: float = 12.0
    layer_seed: int = 0
    # Discard floor for feathered layer alpha (blended, not a hard cutout).
    layer_alpha_threshold: float = 0.05
    # Edge feathering: dilate closes gaps, Gaussian blur softens the silhouette.
    layer_feather_dilate_px: int = 1
    layer_feather_blur_px: int = 5
    # Atmospheric perspective + rim light (breaks the flat-cutout look).
    layer_atmosphere_blend: float = 0.4
    layer_desaturation: float = 0.5
    layer_haze_lift: float = 0.08
    layer_rim_px: int = 2
    layer_rim_intensity: float = 0.3
    rim_light_intensity: float = 0.3
    edge_feather_px: int = 18
    color_bleed_px: int = 3
    rim_light_near_only: bool = True
    #: PB-072: "soft" (mặc định, y hôm nay) | "diecut" (mép cắt dập).
    edge_style: str = "soft"
    #: L2 (TASK_game_1110): tinh lai alpha mep bang guided filter (He et al.
    #: 2010) tren ANH RGB O DO PHAN GIAI GOC, sau buoc feather thuong. Sua
    #: mep "bac thang" cua PB-070b layer_vote: no bau nhan theo vung P10 tinh
    #: o 640 px roi phong INTER_NEAREST, nen voi anh 2048 px mep la mot luoi
    #: khoi ~3 px. Guided filter keo mep ve dung canh mau that cua anh.
    edge_guided_refine: bool = False
    #: Ban kinh cua so ho tro cua guided filter, px anh goc.
    edge_guided_radius_px: int = 8
    #: Be rong dai feather SAU khi da tinh lai, px anh goc (kich ban: 2-3 px).
    edge_guided_feather_px: int = 3
    #: PB-034: "void" (mặc định, y hôm nay) | "box" (hộp quanh các lớp).
    surround: str = "void"
    # PB-073: bóng lớp gần đổ lên lớp xa hơn ("Dimensional Layering").
    layer_shadow: bool = False
    shadow_offset_frac: float = 0.012
    shadow_blur_frac: float = 0.010
    shadow_opacity: float = 0.45
    shadow_angle_deg: float = 90.0
    # Atmospheric mist quad between the two farthest layers (A3).
    mist_enabled: bool = True
    mist_opacity: float = 0.2
    mist_margin: float = 1.2
    # Atmospheric color convergence + value contrast staging (A/B toggle).
    atmospheric_enabled: bool = True
    atmosphere_blend_max: float = 0.35
    far_desaturation: float = 0.6
    near_contrast_boost: float = 1.1
    far_contrast_reduce: float = 0.75
    #: PB-071: "atmo" (mặc định, y hôm nay) | "complement" (xám bằng màu bù).
    far_fog_mode: str = "atmo"
    # Semantic-guided depth adjustment before clustering (3D Ken Burns / SLIDE).
    semantic_depth_enabled: bool = True
    semantic_depth_bands: int = 6
    semantic_depth_similarity: float = 0.06
    semantic_depth_separation: float = 0.08
    semantic_separation_min_horizontal_span: float = 0.6
    semantic_max_segment_variance: float = 0.02
    semantic_min_segment_pixels: int = 150
    semantic_subdivide_pixels: int = 8_000
    semantic_max_work_dim: int = 640
    # Intra-layer relief (P1): gentle per-vertex Z bulge within each clustered layer.
    relief_enabled: bool = True
    relief_amplitude: float = 0.25
    relief_far_falloff: float = 0.3
    relief_smooth_sigma: float = 1.5
    # C1 disocclusion fill: synthesise background behind each silhouette so head
    # movement reveals scenery instead of the void clear colour.
    disocclusion_fill_enabled: bool = True
    disocclusion_fill_px: int = 64
    disocclusion_inpaint_radius: int = 5
    disocclusion_feather_px: int = 8
    disocclusion_max_work_dim: int = 640
    #: L3 (TASK_game_1110): 0 (mac dinh, y hom nay) = khong no. Mot gia tri >0
    #: no mat na lop gan ra ngan nay px (quy theo anh 1280 px rong) truoc khi
    #: tinh lo cua lop xa — bu phan feather tran ra ngoai mat na cung, de vien
    #: mau tien canh khong song sot trong lop xa khi lop gan truot di.
    disocclusion_fringe_dilate_px: int = 0
    #: PB-074: "telea" (mặc định, y hôm nay) | "fsr" | "lama".
    fill_method: str = "telea"
    #: PB-075: cách lấp riêng cho TẤM NỀN (lớp xa nhất): "telea" (mặc định,
    #: y hôm nay) | "shiftmap" | "lama". Lớp giữa vẫn theo ``fill_method``.
    plate_fill: str = "telea"

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ContentSettings:
        """
        Build from a ``content.json`` payload, driven by the dataclass fields.

        Every field is discovered automatically from ``dataclasses.fields()``,
        so **adding a knob to this dataclass is the only step required** — there
        is no second hand-written list to keep in sync.  The previous loader
        parsed ~190 lines of ``payload.get(...)`` by hand, and any field missing
        from that list was silently ignored no matter what the user put in
        ``content.json``.

        Two safety nets the old loader lacked:

        - a value of the wrong shape falls back to the default with a warning
          instead of raising and killing startup;
        - unrecognised keys are reported, so a typo (or a knob that has since
          been removed) is visible rather than silently doing nothing.
        """
        defaults = cls()
        values: dict[str, Any] = {}

        for field_info in fields(cls):
            name = field_info.name
            default = getattr(defaults, name)

            raw = payload.get(name, _MISSING)
            if raw is _MISSING:
                alias = CONTENT_KEY_ALIASES.get(name)
                if alias is not None:
                    raw = payload.get(alias, _MISSING)
            if raw is _MISSING:
                continue

            try:
                values[name] = _coerce_to_default_type(raw, default, name)
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "content.json: bad value for %r (%s) — falling back to %r",
                    name,
                    exc,
                    default,
                )

        known = {item.name for item in fields(cls)}
        known.update(CONTENT_KEY_ALIASES.values())
        # Keys starting with "_" are documentation comments (JSON has none).
        unknown = sorted(
            key for key in payload if key not in known and not key.startswith("_")
        )
        if unknown:
            logger.warning(
                "content.json: %d unrecognised key(s) ignored: %s",
                len(unknown),
                ", ".join(unknown),
            )

        return cls(**values)

    def resolved_media_path(self, root: Path) -> Path:
        """Resolve configured media path relative to project root."""
        raw = self.content_path or self.video_path
        path = Path(raw)
        if not path.is_absolute():
            path = root / raw
        return path

    def resolved_plane_height_m(self, aspect: float = 16.0 / 9.0) -> float:
        """Plane height from config or width/aspect when height is auto."""
        if self.plane_height_m > 0.0:
            return self.plane_height_m
        return self.plane_width_m / max(aspect, 1e-6)


@dataclass
class AppConfig:
    """Root configuration document."""

    camera: CameraSettings = field(default_factory=CameraSettings)
    screen: ScreenSettings = field(default_factory=ScreenSettings)
    webcam_mount: WebcamMountSettings = field(default_factory=WebcamMountSettings)
    user: UserSettings = field(default_factory=UserSettings)
    filter: FilterSettings = field(default_factory=FilterSettings)
    tracking: TrackingSettings = field(default_factory=TrackingSettings)
    adaptive_quality: AdaptiveQualitySettings = field(
        default_factory=AdaptiveQualitySettings
    )
    optimization: OptimizationSettings = field(default_factory=OptimizationSettings)
    renderer: RendererSettings = field(default_factory=RendererSettings)
    ui: UiSettings = field(default_factory=UiSettings)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the full configuration tree."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AppConfig:
        """Build configuration from a nested dictionary."""
        return cls(
            camera=_merge_dataclass(CameraSettings, data.get("camera", {})),
            screen=_merge_dataclass(ScreenSettings, data.get("screen", {})),
            webcam_mount=_merge_dataclass(
                WebcamMountSettings, data.get("webcam_mount", {})
            ),
            user=_merge_dataclass(UserSettings, data.get("user", {})),
            filter=_merge_dataclass(FilterSettings, data.get("filter", {})),
            tracking=_merge_dataclass(TrackingSettings, data.get("tracking", {})),
            adaptive_quality=_merge_dataclass(
                AdaptiveQualitySettings, data.get("adaptive_quality", {})
            ),
            optimization=_merge_dataclass(
                OptimizationSettings, data.get("optimization", {})
            ),
            renderer=_merge_dataclass(RendererSettings, data.get("renderer", {})),
            ui=_merge_dataclass(UiSettings, data.get("ui", {})),
        )


T = TypeVar("T")


def _merge_dataclass(cls: type[T], data: dict[str, Any]) -> T:
    """Construct a dataclass, ignoring unknown keys."""
    if not data:
        return cls()
    field_names = {item.name for item in fields(cls)}
    filtered = {k: v for k, v in data.items() if k in field_names}
    return cls(**filtered)
