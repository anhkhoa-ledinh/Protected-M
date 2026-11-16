"""Cluster a monocular depth map into N discrete alpha-masked layers.

The pipeline is pure NumPy (no OpenGL / torch) so it is deterministic and unit
testable:

1. Optionally adjust depth using semantic segments (:mod:`semantic_depth`) so
   distinct objects (sky vs village) are pushed apart before clustering.
2. Run 1-D k-means (k-means++ seeding + Lloyd iterations) on the raw depth
   values — the underlying floats, never a colormap PNG.
2. Assign every pixel to its nearest centroid to get a hard label map.
3. For each cluster build a boolean mask, then a *feathered* RGBA texture
   (dilate to close gaps, Gaussian blur the edge) so layer boundaries fade
   smoothly instead of showing hard black seams.
4. Sort clusters by depth and map them to a world-Z range (far -> near).
5. Stage each layer's color by distance (:class:`LayerStagingParams`): converge
   its color toward an auto-detected atmosphere tint, stage value contrast
   (near layers punchier, far layers flatter), and desaturate far layers — the
   painting/anime "atmospheric perspective" trick that breaks the flat-cutout
   look. A thin rim light on the foreground silhouette adds extra separation.

The atmosphere color is the mean RGB of the farthest layer's pixels, falling
back to a light blue-gray (:data:`ATMOSPHERE_FALLBACK`, #B8C4D0) when the scene
is too dark to sample a meaningful tint. Color grading runs in linear light
(sRGB is decoded, graded, then re-encoded) so blends and contrast behave like
real haze rather than muddy gamma-space math.

Depth convention: normalized depth ``1.0`` = near (toward viewer), ``0.0`` =
far, matching :func:`src.content.depth_estimator.normalize_depth_map`.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, replace

import cv2
import numpy as np
import numpy.typing as npt

from src.content.disocclusion import (
    DisocclusionParams,
    DisocclusionReport,
    fill_layer_disocclusion,
)
from src.content.debug_marks import paint_zone, top_layer_colour
from src.content.fog_complement import complement_mix, solve_t_for_layer
from src.content.semantic_depth import (
    SemanticDepthParams,
    SemanticDepthReport,
    adjust_depth_semantic,
)

MIST_COLOR_FALLBACK: tuple[float, float, float] = (0.85, 0.87, 0.90)

_MAX_FIT_SAMPLES = 40_000

#: Light blue-gray (#B8C4D0) used as the atmosphere tint when the farthest layer
#: is too dark to sample a meaningful haze color.
ATMOSPHERE_FALLBACK: tuple[float, float, float] = (184.0, 196.0, 208.0)

#: Below this mean luminance (0-255) the farthest layer is considered "too dark"
#: and the atmosphere color falls back to :data:`ATMOSPHERE_FALLBACK`.
DARK_LUMA_THRESHOLD: float = 25.0


@dataclass(frozen=True)
class LayerEdgeParams:
    """Silhouette softening: rim light, wide edge feather, cross-layer color bleed."""

    rim_light_intensity: float = 0.3
    edge_feather_px: int = 18
    color_bleed_px: int = 3
    rim_light_near_only: bool = True
    edge_gradient_threshold: float = 0.04
    """RGB gradient above this marks EDGE-classified silhouette boundaries."""

    edge_style: str = "soft"
    """
    PB-072: kiểu mép lớp. ``"soft"`` (mặc định) = hôm nay: viền sáng, feather
    rộng tới ``edge_feather_px`` ở chỗ tương phản cao, pha màu lớp sau.
    ``"diecut"`` = mép cắt dập như thiệp giấy: **không** viền sáng, dilate 0 +
    blur :data:`DIECUT_BLUR_PX`, **không** feather rộng, **không** pha màu lớp
    sau, và trước khi feather thì tẩy màu nền ra khỏi vành mép
    (:func:`decontaminate_edge_colour`).
    """

    guided_refine: bool = False
    """L2 (TASK_game_1110): tinh lai alpha bang guided filter sau feather thuong."""

    guided_radius_px: int = 8
    guided_feather_px: int = 3


@dataclass(frozen=True)
class LayerStagingParams:
    """
    Per-layer color-grading knobs (atmospheric perspective + value staging).

    Depth is expressed as ``depth_norm`` in ``[0, 1]`` (0 = nearest, 1 =
    farthest). ``enabled=False`` bypasses all grading, which is the "OFF" arm
    of the A/B comparison used in the thesis evaluation.
    """

    enabled: bool = True
    atmosphere_blend_max: float = 0.35
    """Max fraction blended toward the atmosphere at the farthest layer."""

    far_desaturation: float = 0.6
    """Saturation multiplier applied to the farthest layer (``S *= 0.6``)."""

    near_contrast_boost: float = 1.1
    """Contrast multiplier for the nearest layer (>1 = punchier)."""

    far_contrast_reduce: float = 0.75
    """Contrast multiplier for the farthest layer (<1 = flatter/hazier)."""

    far_fog_mode: str = "atmo"
    """
    PB-071: cách xám hoá lớp xa. ``"atmo"`` (mặc định) = hôm nay: trộn về màu
    khí quyển rồi hạ bão hoà. ``"complement"`` = kỹ thuật hoạ sĩ: giữ nguyên
    bước contrast, thay hai bước kia bằng trộn với **màu bù** của chính pixel,
    hệ số ``t`` dò sao cho bão hoà trung bình bằng bản ``"atmo"`` — tức A/B
    hỏi "xám KIỂU NÀO", không phải "xám BAO NHIÊU".
    """


@dataclass(frozen=True)
class LayerGradeReport:
    """The color-grade actually applied to one layer (for F1 / logging)."""

    index: int
    depth_norm: float
    atmosphere_blend: float
    """Fraction blended toward atmosphere color (``depth_norm * blend_max``)."""

    saturation_factor: float
    """HSV saturation multiplier applied (1.0 = untouched)."""

    contrast_factor: float
    """Luminance contrast multiplier applied (1.0 = untouched)."""

    enabled: bool = True

    complement_t: float = 0.0
    """PB-071: hệ số trộn màu bù đã dùng (0.0 = không dùng, chế độ ``atmo``)."""


@dataclass(frozen=True)
class DepthLayer:
    """One discrete depth slice extracted from the depth map."""

    index: int
    """Back-to-front order: 0 = farthest cluster, ``n-1`` = nearest."""

    mean_depth: float
    """Mean normalized depth of the cluster (0 = far, 1 = near)."""

    depth_lo: float
    depth_hi: float
    z_m: float
    """Assigned world Z offset (more negative = farther from viewer)."""

    coverage: float
    """Fraction of image pixels belonging to this cluster (0-1)."""

    mask: npt.NDArray[np.bool_]
    """``HxW`` boolean silhouette of the cluster."""

    rgba: npt.NDArray[np.uint8]
    """``HxWx4`` RGBA texture; RGB is depth-graded, alpha is the feathered mask."""

    depth_norm: float = 0.0
    """Farness in [0, 1] used for atmospheric perspective (1 = farthest)."""

    grade_report: LayerGradeReport | None = None
    """The color-grade applied to this layer (``None`` = legacy path)."""


@dataclass(frozen=True)
class ClusterIntoLayersResult:
    """Output of :func:`cluster_into_layers`."""

    layers: list[DepthLayer]
    semantic_report: SemanticDepthReport | None = None
    mist_color: tuple[float, float, float] = MIST_COLOR_FALLBACK
    timing_ms: dict[str, float] = field(default_factory=dict)
    disocclusion_reports: list[DisocclusionReport] = field(default_factory=list)


def compute_mist_color(far_layer: DepthLayer) -> tuple[float, float, float]:
    """
    Average RGB of the farthest layer (alpha > 0), lightened 15% toward white.
    """
    rgba = far_layer.rgba
    visible = rgba[..., 3] > 0
    if not bool(visible.any()):
        return MIST_COLOR_FALLBACK
    rgb = rgba[..., :3][visible].astype(np.float64)
    if rgb.size == 0:
        return MIST_COLOR_FALLBACK
    mean = rgb.mean(axis=0) / 255.0
    lightened = mean * 0.85 + 0.15
    return (
        float(np.clip(lightened[0], 0.0, 1.0)),
        float(np.clip(lightened[1], 0.0, 1.0)),
        float(np.clip(lightened[2], 0.0, 1.0)),
    )


def layer_feather_scales(
    depth_norm: float, base_blur: int, edge_feather_px: int
) -> tuple[int, int]:
    """
    Interpolate feather radii from normalized depth (0=near, 1=far).

    At N=3 the mid/far values stay close to legacy defaults.
    """
    t = float(np.clip(depth_norm, 0.0, 1.0))
    blur = max(1, int(round(base_blur * (1.0 + 0.15 * t))))
    edge = max(blur + 1, int(round(edge_feather_px * (1.0 + 0.2 * t))))
    return blur, edge


logger = logging.getLogger(__name__)


def _kmeans_1d(
    values: npt.NDArray[np.float64],
    k: int,
    seed: int = 0,
    iters: int = 100,
) -> npt.NDArray[np.float64]:
    """Deterministic 1-D k-means (k-means++ init) returning sorted centroids."""
    v = np.asarray(values, dtype=np.float64).ravel()
    rng = np.random.default_rng(seed)

    # k-means++ seeding.
    first = int(rng.integers(v.shape[0]))
    centers = [float(v[first])]
    for _ in range(1, k):
        dist_sq = np.min(
            np.stack([(v - c) ** 2 for c in centers], axis=0), axis=0
        )
        total = float(dist_sq.sum())
        if total <= 0.0:
            centers.append(float(v[int(rng.integers(v.shape[0]))]))
            continue
        idx = int(rng.choice(v.shape[0], p=dist_sq / total))
        centers.append(float(v[idx]))

    centroids = np.sort(np.asarray(centers, dtype=np.float64))
    for _ in range(iters):
        labels = np.argmin(np.abs(v[:, None] - centroids[None, :]), axis=1)
        updated = centroids.copy()
        for j in range(k):
            members = v[labels == j]
            if members.size:
                updated[j] = float(members.mean())
        updated.sort()
        if np.allclose(updated, centroids):
            centroids = updated
            break
        centroids = updated
    return centroids


def cluster_depth_layers(
    depth: npt.NDArray[np.float32],
    n_clusters: int = 5,
    seed: int = 0,
) -> tuple[npt.NDArray[np.int32], npt.NDArray[np.float64]]:
    """
    Cluster a depth map into ``n_clusters`` bands.

    Returns ``(labels, centroids)`` where ``labels`` is an ``HxW`` int map (0 =
    farthest band) and ``centroids`` are the depth centroids sorted ascending
    (far -> near).
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth map, got shape {depth.shape}")
    if n_clusters < 1:
        raise ValueError("n_clusters must be >= 1")

    flat = depth.reshape(-1).astype(np.float64)
    fit_values = flat
    if flat.shape[0] > _MAX_FIT_SAMPLES:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(flat.shape[0], size=_MAX_FIT_SAMPLES, replace=False)
        fit_values = flat[sample_idx]

    centroids = _kmeans_1d(fit_values, n_clusters, seed=seed)
    labels_flat = np.argmin(
        np.abs(flat[:, None] - centroids[None, :]), axis=1
    ).astype(np.int32)
    labels = labels_flat.reshape(depth.shape)
    return labels, centroids


def classify_silhouette_edges(
    mask: npt.NDArray[np.bool_],
    rgb: npt.NDArray[np.uint8],
    *,
    gradient_threshold: float = 0.04,
) -> npt.NDArray[np.bool_]:
    """
    Mark high-contrast silhouette boundaries (trees, subjects vs sky).

    Used to widen alpha feather only where a hard cutout would be visible.
    """
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    mask_u8 = np.where(mask, np.uint8(255), np.uint8(0)).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    boundary = cv2.dilate(mask_u8, kernel) != cv2.erode(mask_u8, kernel)
    return boundary & (grad >= float(gradient_threshold))


def _silhouette_band(mask: npt.NDArray[np.bool_], width_px: int) -> npt.NDArray[np.bool_]:
    """Pixels within ``width_px`` of the mask silhouette (inside and outside)."""
    width = max(1, int(width_px))
    mask_u8 = np.where(mask, np.uint8(255), np.uint8(0)).astype(np.uint8)
    k = 2 * width + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    dilated = cv2.dilate(mask_u8, kernel) > 0
    eroded = cv2.erode(mask_u8, kernel) > 0
    return dilated & ~eroded


def feather_mask_alpha(
    mask: npt.NDArray[np.bool_],
    dilate_px: int = 1,
    blur_px: int = 5,
    *,
    edge_mask: npt.NDArray[np.bool_] | None = None,
    edge_blur_px: int = 0,
) -> npt.NDArray[np.uint8]:
    """
    Turn a hard boolean mask into a soft ``uint8`` alpha channel.

    Steps: dilate a few pixels to close small gaps/holes, then Gaussian blur
    the edge so alpha ramps 0->255 instead of a jagged binary cutout. This is
    what removes the hard black seams between adjacent depth layers.
    """
    alpha = np.where(mask, np.uint8(255), np.uint8(0)).astype(np.uint8)
    if dilate_px > 0:
        k = 2 * int(dilate_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        alpha = cv2.dilate(alpha, kernel)
    if blur_px > 0:
        # Gaussian kernel must be odd; blur_px is the approximate edge radius.
        k = int(blur_px) | 1
        alpha = cv2.GaussianBlur(alpha, (k, k), 0)

    wide_blur = max(int(edge_blur_px), int(blur_px))
    if wide_blur > blur_px:
        wide = feather_mask_alpha(mask, dilate_px=dilate_px, blur_px=wide_blur)
        band = _silhouette_band(mask, max(wide_blur // 2, 3))
        if edge_mask is not None:
            band = band | edge_mask
        alpha = np.where(band, wide, alpha)

    return np.ascontiguousarray(alpha, dtype=np.uint8)


def refine_edge_alpha_guided(
    rgb: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.uint8],
    *,
    radius_px: int = 8,
    feather_px: int = 3,
    eps: float = 0.02,
) -> npt.NDArray[np.uint8]:
    """
    L2 (TASK_game_1110) — sửa mép "bậc thang" bằng guided filter, full-res.

    Nguồn của bậc thang: ``layer_method="kmeans_vote"`` (PB-070b) bầu nhãn
    theo vùng P10, mà vùng đó tính ở ``WORK_MAX_DIM=640`` (``layer_regions.py``)
    rồi phóng lên cỡ gốc bằng ``INTER_NEAREST``. Trên ảnh 2048 px cạnh dài, hệ
    số phóng ~3,2×, nên mỗi bước nhãn là một khối ~3 px vuông — đúng "bậc
    thang" ở ghế và tóc mà người dùng thấy trên 07/11. ``adaptive_k`` và
    ``semantic_depth`` cũng hạ xuống làm việc ở ≤ 640/960 px trước khi phóng
    lại, nên cùng một cơ chế có thể góp phần, nhưng ``kmeans_vote`` là nguồn
    lớn nhất vì nó sửa trực tiếp NHÃN từng pixel chứ không chỉ một ngưỡng.

    Guided filter (He, Sun, Tang 2010) không thu nhỏ gì: nó chạy ở ĐÚNG độ
    phân giải của ``rgb`` (ảnh gốc), dùng chính ảnh làm "guide" — trong mỗi
    cửa sổ bán kính ``radius_px``, ngõ ra là một hàm TUYẾN TÍNH của ``rgb``,
    nên mép alpha bật theo đúng cạnh màu thật của ảnh thay vì bám lưới bầu
    640 px. Sau đó xiết lại thành một dải hẹp ``feather_px`` (2–3 px theo kịch
    bản) quanh đường bao mới — guided filter một mình có thể để lại quầng mềm
    rộng hơn mức "cắt dập" mà hồ sơ game muốn.
    """
    guide = np.ascontiguousarray(rgb, dtype=np.uint8)
    smoothed = cv2.ximgproc.guidedFilter(
        guide, np.ascontiguousarray(alpha, dtype=np.uint8),
        radius=max(1, int(radius_px)), eps=float(eps) ** 2 * 255.0 * 255.0)
    mask = smoothed > 127
    return feather_mask_alpha(mask, dilate_px=0, blur_px=max(1, int(feather_px)))


def apply_color_bleed(
    rgb: npt.NDArray[np.uint8],
    alpha: npt.NDArray[np.uint8],
    behind_rgb: npt.NDArray[np.uint8],
    bleed_px: int,
) -> npt.NDArray[np.uint8]:
    """
    Blend foreground RGB toward the layer behind in the soft-alpha edge zone.

    Prevents a hard color break at silhouettes when layers are composited.
    """
    if bleed_px <= 0:
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    if rgb.shape != behind_rgb.shape:
        raise ValueError("rgb and behind_rgb shape mismatch")

    k = 2 * int(bleed_px) + 1
    behind_smooth = cv2.GaussianBlur(behind_rgb, (k, k), 0)
    soft = (alpha > 0) & (alpha < 255)
    if not bool(soft.any()):
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    t = 1.0 - alpha.astype(np.float32) / 255.0
    t = np.clip(t * 0.85, 0.0, 0.65)
    out = rgb.astype(np.float32)
    t3 = t[..., None]
    out = out * (1.0 - t3) + behind_smooth.astype(np.float32) * t3
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


#: PB-072: bán kính blur của mép die-cut (thay cho 5 px + feather rộng 18 px).
DIECUT_BLUR_PX = 3


def decontaminate_edge_colour(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    band_px: int,
) -> npt.NDArray[np.uint8]:
    """
    Cho vành ``band_px`` NGOÀI mask mượn màu của điểm trong mask gần nhất.

    Vì sao cần: ``extract_layer_rgba`` giữ RGB ở **mọi** điểm rồi mới áp alpha
    mềm, nên vành mép của lớp gần mang màu **nền** và kéo theo màu nền khi đầu
    dịch — đúng kiểu quầng matting. Tẩy màu trước khi feather thì mép mềm chỉ
    còn tán màu của chính vật, không bao giờ tán màu nền.

    ``distanceTransformWithLabels`` với ``DIST_LABEL_PIXEL`` trả nhãn của điểm
    nền gần nhất; ở đây "nền" của phép biến đổi chính là các điểm TRONG mask.
    """
    if not bool(mask.any()) or int(band_px) <= 0:
        return np.ascontiguousarray(rgb, dtype=np.uint8)
    inside_is_zero = np.where(mask, np.uint8(0), np.uint8(255))
    distance, labels = cv2.distanceTransformWithLabels(
        inside_is_zero, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
    )
    # Nhãn tại CHÍNH các điểm trong mask cho biết nhãn nào ứng với điểm nào.
    flat_of_label = np.zeros(int(labels.max()) + 1, dtype=np.int64)
    ys, xs = np.nonzero(mask)
    flat_of_label[labels[mask]] = np.ravel_multi_index((ys, xs), mask.shape)
    nearest = rgb.reshape(-1, 3)[flat_of_label[labels]].reshape(rgb.shape)
    band = (~mask) & (distance <= float(band_px))
    out = np.ascontiguousarray(rgb, dtype=np.uint8).copy()
    out[band] = nearest[band]
    return out


def extract_layer_rgba(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    *,
    dilate_px: int = 1,
    blur_px: int = 5,
    behind_rgb: npt.NDArray[np.uint8] | None = None,
    edge: LayerEdgeParams | None = None,
) -> npt.NDArray[np.uint8]:
    """
    Build an ``HxWx4`` RGBA texture: full RGB copied, alpha = feathered mask.

    RGB is kept everywhere (not zeroed outside the cluster) so the soft edge
    blends the true image colors — adjacent layers share those colors at the
    boundary, leaving no black gap when composited back-to-front.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {rgb.shape}")
    if mask.shape != rgb.shape[:2]:
        raise ValueError(
            f"Mask shape {mask.shape} does not match image {rgb.shape[:2]}"
        )
    rgba = np.zeros((*mask.shape, 4), dtype=np.uint8)
    graded_rgb = np.ascontiguousarray(rgb, dtype=np.uint8)
    edge_mask = None
    edge_blur = 0
    bleed_px = 0
    if edge is not None:
        edge_mask = classify_silhouette_edges(
            mask, graded_rgb, gradient_threshold=edge.edge_gradient_threshold
        )
        edge_blur = int(edge.edge_feather_px)
        bleed_px = int(edge.color_bleed_px)

    alpha = feather_mask_alpha(
        mask,
        dilate_px=dilate_px,
        blur_px=blur_px,
        edge_mask=edge_mask,
        edge_blur_px=edge_blur,
    )
    if edge is not None and edge.guided_refine:
        alpha = refine_edge_alpha_guided(
            graded_rgb, alpha, radius_px=edge.guided_radius_px,
            feather_px=edge.guided_feather_px)
    if behind_rgb is not None and bleed_px > 0:
        graded_rgb = apply_color_bleed(graded_rgb, alpha, behind_rgb, bleed_px)

    rgba[..., :3] = graded_rgb
    rgba[..., 3] = alpha
    return rgba


def estimate_atmosphere_color(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    *,
    dark_fallback: tuple[float, float, float] | None = None,
    dark_luma_threshold: float = DARK_LUMA_THRESHOLD,
) -> tuple[float, float, float]:
    """
    Auto-detect the atmosphere/sky color as the mean RGB of the farthest layer.

    Nothing is hardcoded — the haze tint is sampled from the scene's own
    background pixels. When ``dark_fallback`` is given and the sampled color is
    darker than ``dark_luma_threshold`` (Rec.601 luma, 0-255), that fallback is
    returned instead so a near-black background does not produce a muddy haze.
    """
    pixels = rgb[mask]
    if pixels.size == 0:
        pixels = rgb.reshape(-1, 3)
    mean = pixels.reshape(-1, 3).astype(np.float64).mean(axis=0)
    color = (float(mean[0]), float(mean[1]), float(mean[2]))
    if dark_fallback is not None:
        luma = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
        if luma < dark_luma_threshold:
            return dark_fallback
    return color


def _srgb_to_linear(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Decode sRGB (0-1) to linear light."""
    a = 0.055
    return np.where(x <= 0.04045, x / 12.92, ((x + a) / (1.0 + a)) ** 2.4).astype(
        np.float32
    )


def _linear_to_srgb(x: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """Encode linear light back to sRGB (0-1)."""
    a = 0.055
    x = np.clip(x, 0.0, 1.0)
    return np.where(
        x <= 0.0031308, x * 12.92, (1.0 + a) * np.power(x, 1.0 / 2.4) - a
    ).astype(np.float32)


def staging_factors(
    depth_norm: float, params: LayerStagingParams
) -> tuple[float, float, float]:
    """
    Map a layer's farness to ``(atmosphere_blend, saturation, contrast)``.

    The curve is anchored so the *middle* layer is neutral (Katana-Zero value
    staging): nearest boosts contrast, farthest reduces contrast + desaturates,
    mid is untouched.

    - ``depth_norm = 0`` (near): contrast ``near_contrast_boost``, sat 1.0
    - ``depth_norm = 0.5`` (mid): contrast 1.0, sat 1.0, no atmosphere
    - ``depth_norm = 1`` (far): contrast ``far_contrast_reduce``, sat
      ``far_desaturation``, full atmosphere blend
    """
    d = float(np.clip(depth_norm, 0.0, 1.0))
    blend = d * params.atmosphere_blend_max
    if d <= 0.5:
        t = d / 0.5
        contrast = params.near_contrast_boost + (1.0 - params.near_contrast_boost) * t
        saturation = 1.0
    else:
        t = (d - 0.5) / 0.5
        contrast = 1.0 + (params.far_contrast_reduce - 1.0) * t
        saturation = 1.0 + (params.far_desaturation - 1.0) * t
    return blend, saturation, contrast


def complement_fog(
    lin_contrast: npt.NDArray[np.float32],
    atmo_out: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_] | None,
) -> tuple[npt.NDArray[np.uint8], float]:
    """
    PB-071 ``far_fog_mode="complement"``: xám bằng màu bù, không bằng tấm xám.

    ``lin_contrast`` là lớp SAU bước contrast và TRƯỚC bước trộn khí quyển;
    ``atmo_out`` là kết quả đầy đủ của chế độ ``atmo``, chỉ dùng làm **đích
    bão hoà**. Trả ``(ảnh, t)``.
    """
    base = np.clip(_linear_to_srgb(lin_contrast) * 255.0,
                   0.0, 255.0).astype(np.uint8)
    t = float(solve_t_for_layer(base, atmo_out, mask))
    return np.ascontiguousarray(complement_mix(base, t), dtype=np.uint8), t


def stage_layer_color(
    rgb: npt.NDArray[np.uint8],
    depth_norm: float,
    atmosphere_color: tuple[float, float, float],
    params: LayerStagingParams,
    *,
    index: int = 0,
    luminance_pivot: float | None = None,
    mask: npt.NDArray[np.bool_] | None = None,
) -> tuple[npt.NDArray[np.uint8], LayerGradeReport]:
    """
    Grade one layer by distance and report exactly what was applied.

    Order (contrast -> atmosphere convergence in linear light, then saturation
    in HSV):

    1. contrast around ``luminance_pivot`` (mean linear luma of the layer)
    2. ``color = lerp(color, atmosphere_color, depth_norm * atmosphere_blend_max)``
    3. desaturate: ``S *= saturation_factor``

    Với ``params.far_fog_mode == "complement"`` (PB-071) bước 1 giữ nguyên còn
    bước 2 và 3 được thay bằng :func:`complement_fog`; ``mask`` là mask của lớp
    và chỉ dùng ở đó (đo bão hoà đúng trên điểm của lớp, không phải cả ảnh).

    Returns the graded ``HxWx3`` image and a :class:`LayerGradeReport`. When
    ``params.enabled`` is false the image is returned untouched with a neutral
    report (the A/B "OFF" arm).
    """
    blend, sat_factor, contrast_factor = staging_factors(depth_norm, params)
    report = LayerGradeReport(
        index=index,
        depth_norm=float(np.clip(depth_norm, 0.0, 1.0)),
        atmosphere_blend=blend,
        saturation_factor=sat_factor,
        contrast_factor=contrast_factor,
        enabled=params.enabled,
    )
    if not params.enabled:
        neutral = LayerGradeReport(
            index=index,
            depth_norm=report.depth_norm,
            atmosphere_blend=0.0,
            saturation_factor=1.0,
            contrast_factor=1.0,
            enabled=False,
        )
        return np.ascontiguousarray(rgb, dtype=np.uint8), neutral

    lin = _srgb_to_linear(rgb.astype(np.float32) / 255.0)

    if contrast_factor != 1.0:
        luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
        pivot = (
            float(luma.mean()) if luminance_pivot is None else float(luminance_pivot)
        )
        lin = pivot + (lin - pivot) * contrast_factor
        lin = np.clip(lin, 0.0, 1.0)

    lin_contrast = lin                  # PB-071: bản chỉ mới chỉnh contrast

    if blend > 0.0:
        atmo_lin = _srgb_to_linear(
            np.asarray(atmosphere_color, dtype=np.float32) / 255.0
        )
        lin = lin * (1.0 - blend) + atmo_lin[None, None, :] * blend

    out = np.clip(_linear_to_srgb(lin) * 255.0, 0.0, 255.0).astype(np.uint8)

    if sat_factor < 1.0:
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[..., 1] *= sat_factor
        out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)

    # Lớp không bị sương (blend = 0 và sat = 1) thì hai chế độ phải giống nhau.
    if (str(params.far_fog_mode).lower().strip() == "complement"
            and (blend > 0.0 or sat_factor < 1.0)):
        out, t_mix = complement_fog(lin_contrast, out, mask)
        report = replace(report, complement_t=t_mix)

    return np.ascontiguousarray(out, dtype=np.uint8), report


def _layer_luminance_pivot(rgb: npt.NDArray[np.uint8], mask: npt.NDArray[np.bool_]) -> float:
    """Mean *linear* luma (0-1) of a layer's masked pixels; contrast pivot."""
    pixels = rgb[mask] if mask.any() else rgb.reshape(-1, 3)
    lin = _srgb_to_linear(pixels.astype(np.float32) / 255.0)
    luma = 0.2126 * lin[..., 0] + 0.7152 * lin[..., 1] + 0.0722 * lin[..., 2]
    return float(luma.mean())


def apply_atmospheric_perspective(
    rgb: npt.NDArray[np.uint8],
    depth_norm: float,
    atmosphere_color: tuple[float, float, float],
    *,
    atmosphere_blend: float = 0.4,
    desaturation: float = 0.5,
    haze_lift: float = 0.08,
) -> npt.NDArray[np.uint8]:
    """
    Fade a layer toward the atmosphere with distance (``depth_norm``: 0 near,
    1 far).

    1. ``color = lerp(color, atmosphere_color, depth_norm * atmosphere_blend)``
    2. desaturate: ``S *= (1 - depth_norm * desaturation)``
    3. haze: lift brightness ``V`` slightly for far layers
    """
    d = float(np.clip(depth_norm, 0.0, 1.0))
    if d <= 0.0:
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    img = rgb.astype(np.float32)
    atmo = np.asarray(atmosphere_color, dtype=np.float32)
    t = d * atmosphere_blend
    img = img * (1.0 - t) + atmo[None, None, :] * t

    hsv = cv2.cvtColor(
        np.clip(img, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV
    ).astype(np.float32)
    hsv[..., 1] *= max(0.0, 1.0 - d * desaturation)
    hsv[..., 2] = np.clip(hsv[..., 2] + d * haze_lift * 255.0, 0.0, 255.0)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
    return np.ascontiguousarray(out, dtype=np.uint8)


def add_rim_light(
    rgb: npt.NDArray[np.uint8],
    mask: npt.NDArray[np.bool_],
    *,
    width_px: int = 2,
    intensity: float = 0.6,
    color: tuple[float, float, float] = (255.0, 255.0, 255.0),
) -> npt.NDArray[np.uint8]:
    """
    Brighten a thin band along the inside of ``mask``'s silhouette.

    Used on the foreground layer only to add a subtle glow that separates it
    from the background.
    """
    if width_px <= 0 or intensity <= 0.0:
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    mask_u8 = np.where(mask, np.uint8(255), np.uint8(0)).astype(np.uint8)
    k = 2 * int(width_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    eroded = cv2.erode(mask_u8, kernel)
    rim = (mask_u8 > 0) & (eroded == 0)
    if not rim.any():
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    out = rgb.astype(np.float32)
    rim_color = np.asarray(color, dtype=np.float32)
    a = float(np.clip(intensity, 0.0, 1.0))
    out[rim] = out[rim] * (1.0 - a) + rim_color[None, :] * a
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


def inter_layer_gap_m(
    z_near_m: float, z_far_m: float, layer_count: int
) -> float:
    """Even spacing between discrete layer planes (meters)."""
    if layer_count <= 1:
        return abs(z_near_m - z_far_m)
    return abs(z_near_m - z_far_m) / (layer_count - 1)


def relief_depth_scale(depth_norm: float, far_falloff: float) -> float:
    """
    Relief multiplier from layer farness (same ``depth_norm`` as atmospheric staging).

    ``depth_norm=0`` (near) -> 1.0; ``depth_norm=1`` (far) -> ``far_falloff``.
    """
    d = float(np.clip(depth_norm, 0.0, 1.0))
    falloff = float(np.clip(far_falloff, 0.0, 1.0))
    return 1.0 - d * (1.0 - falloff)


def assign_layer_z(
    rank: int,
    layer_count: int,
    z_near_m: float,
    z_far_m: float,
) -> float:
    """
    Map a far->near ``rank`` (0 = farthest) to an evenly spaced world Z.

    Even spacing guarantees visibly distinct layers regardless of how the
    depth clusters bunch together.
    """
    if layer_count <= 1:
        return z_near_m
    t = rank / (layer_count - 1)
    return z_far_m + (z_near_m - z_far_m) * t


def cluster_into_layers(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.float32],
    n_clusters: int = 3,
    *,
    z_near_m: float = 0.0,
    z_far_m: float = -0.35,
    seed: int = 0,
    feather_dilate_px: int = 1,
    feather_blur_px: int = 5,
    atmosphere_blend: float = 0.4,
    desaturation: float = 0.5,
    haze_lift: float = 0.08,
    rim_light_px: int = 2,
    rim_intensity: float = 0.6,
    atmosphere_color: tuple[float, float, float] | None = None,
    staging: LayerStagingParams | None = None,
    semantic: SemanticDepthParams | None = None,
    edge: LayerEdgeParams | None = None,
    disocclusion: DisocclusionParams | None = None,
    layer_method: str = "kmeans",
) -> ClusterIntoLayersResult:
    """
    Cluster ``depth`` into discrete layers and slice ``rgb`` into RGBA textures.

    Default ``n_clusters`` is 3 (foreground / mid / background), which suits
    most photos; raise it (e.g. 5) for scenes with clear multi-object depth
    separation. Layer alpha is feathered (dilate + blur) for smooth edges.

    Color grading depends on ``staging``:

    - ``staging`` provided (preferred): each layer is graded with
      :func:`stage_layer_color` (atmosphere convergence + value contrast
      staging in linear light) and its :class:`LayerGradeReport` is attached to
      the returned :class:`DepthLayer`. ``staging.enabled=False`` yields raw,
      ungraded layers — the "OFF" arm of the A/B comparison.
    - ``staging=None`` (legacy): the older
      :func:`apply_atmospheric_perspective` path is used.

    The returned list is ordered back-to-front (farthest layer first) so the
    caller can render it directly with alpha blending.
    """
    if rgb.shape[:2] != depth.shape:
        raise ValueError(
            f"RGB {rgb.shape[:2]} and depth {depth.shape} sizes differ"
        )

    semantic_report: SemanticDepthReport | None = None
    work_depth = depth
    t_sem0 = time.perf_counter()
    if semantic is not None and semantic.enabled:
        work_depth, semantic_report = adjust_depth_semantic(rgb, depth, semantic)
    semantic_ms = (time.perf_counter() - t_sem0) * 1000.0

    t_clu0 = time.perf_counter()
    t_km0 = time.perf_counter()
    labels, centroids = cluster_depth_layers(work_depth, n_clusters, seed=seed)
    # PB-070b: bầu theo vùng NGAY SAU k-means và **không cụm lại** — tâm cụm
    # phải giữ nguyên, nếu không thì mất đúng thứ tự che khuất mà SPIKE-15 bảo
    # toàn được. Mặc định "kmeans" đi thẳng qua, không gọi gì.
    if str(layer_method).lower().strip() in {"kmeans_vote", "vote"}:
        from src.content.layer_vote import vote_on_regions

        labels, _frac = vote_on_regions(labels, centroids, work_depth, rgb)
    kmeans_ms = (time.perf_counter() - t_km0) * 1000.0
    present = [j for j in range(centroids.shape[0]) if bool((labels == j).any())]
    if len(present) < n_clusters:
        logger.warning(
            "Depth clustering produced %d non-empty layers (requested %d)",
            len(present),
            n_clusters,
        )

    count = len(present)
    t_mask0 = time.perf_counter()
    masks = [labels == cluster_id for cluster_id in present]
    masks_ms = (time.perf_counter() - t_mask0) * 1000.0

    # Auto-detect the atmosphere tint from the farthest layer (rank 0), with a
    # light blue-gray fallback when that layer is too dark to sample.
    if atmosphere_color is None and masks:
        atmosphere_color = estimate_atmosphere_color(
            rgb, masks[0], dark_fallback=ATMOSPHERE_FALLBACK
        )
    atmo = atmosphere_color or ATMOSPHERE_FALLBACK

    edge_params = edge if edge is not None else LayerEdgeParams()
    rim_intensity = float(edge_params.rim_light_intensity) if edge is not None else rim_intensity

    layers: list[DepthLayer] = []
    staging_ms = rim_ms = feather_ms = composite_ms = 0.0
    disocclusion_reports: list[DisocclusionReport] = []
    t_dis_total = 0.0
    composite_rgb = rgb.astype(np.float32)
    for rank, mask in enumerate(masks):
        member_depths = depth[mask]
        z = assign_layer_z(rank, count, z_near_m, z_far_m)
        # Farness: rank 0 (far) -> 1.0, nearest -> 0.0.
        depth_norm = 1.0 - rank / (count - 1) if count > 1 else 0.0

        report: LayerGradeReport | None = None
        t_stage0 = time.perf_counter()
        if staging is not None:
            graded, report = stage_layer_color(
                rgb,
                depth_norm,
                atmo,
                staging,
                index=rank,
                luminance_pivot=_layer_luminance_pivot(rgb, mask),
                mask=mask,
            )
            if report.complement_t > 0.0:
                # PB-071: chỉ in khi chế độ màu bù thực sự chạy, nên nhật ký
                # của chế độ mặc định không đổi một dòng nào.
                logger.info(
                    "Layer %d: far_fog_mode=complement t=%.4f", rank,
                    report.complement_t,
                )
            logger.info(
                "Layer %d/%d: depth_norm=%.2f blend=%.0f%% sat=%.2f "
                "contrast=%.2f atmo=(%.0f,%.0f,%.0f) %s",
                rank,
                count,
                depth_norm,
                report.atmosphere_blend * 100.0,
                report.saturation_factor,
                report.contrast_factor,
                atmo[0],
                atmo[1],
                atmo[2],
                "ON" if staging.enabled else "OFF",
            )
        else:
            graded = apply_atmospheric_perspective(
                rgb,
                depth_norm,
                atmo,
                atmosphere_blend=atmosphere_blend,
                desaturation=desaturation,
                haze_lift=haze_lift,
            )

        staging_ms += (time.perf_counter() - t_stage0) * 1000.0

        # Rim light on nearest layer only (subtle by default).
        # PB-072: mép cắt dập — không viền sáng, không feather rộng, không pha
        # màu lớp sau, và tẩy màu nền khỏi vành mép trước khi feather.
        diecut = str(edge_params.edge_style).lower().strip() == "diecut"

        t_rim0 = time.perf_counter()
        apply_rim = count > 1 and rank == count - 1
        if edge_params.rim_light_near_only is False and count > 1:
            apply_rim = rim_intensity > 0.0
        if diecut:
            apply_rim = False
        if apply_rim and rim_intensity > 0.0:
            graded = add_rim_light(
                graded, mask, width_px=rim_light_px, intensity=rim_intensity
            )

        rim_ms += (time.perf_counter() - t_rim0) * 1000.0

        t_feather0 = time.perf_counter()
        behind = np.clip(composite_rgb, 0, 255).astype(np.uint8)
        blur_layer = int(feather_blur_px)
        edge_for_layer = edge_params if edge is not None else None
        if edge is not None:
            blur_layer, edge_feather = layer_feather_scales(
                depth_norm, feather_blur_px, edge_params.edge_feather_px
            )
            edge_for_layer = replace(edge_params, edge_feather_px=edge_feather)
        if diecut:
            # ``edge=None`` tắt cả feather rộng lẫn pha màu, đúng một chỗ.
            graded = decontaminate_edge_colour(graded, mask, DIECUT_BLUR_PX + 1)
            blur_layer, edge_for_layer, behind = DIECUT_BLUR_PX, None, None
        rgba = extract_layer_rgba(
            graded,
            mask,
            dilate_px=0 if diecut else feather_dilate_px,
            blur_px=blur_layer,
            behind_rgb=behind,
            edge=edge_for_layer,
        )

        feather_ms += (time.perf_counter() - t_feather0) * 1000.0

        # C1 disocclusion fill: synthesise background behind this silhouette so
        # sliding a nearer layer across it reveals scenery, not the void.  Only
        # layers that have something in front of them can be uncovered.
        if disocclusion is not None and disocclusion.enabled and rank + 1 < count:
            nearer = np.zeros_like(mask)
            for other in masks[rank + 1:]:
                nearer |= other
            t_dis0 = time.perf_counter()
            # rank 0 là lớp XA NHẤT — tấm nền, sau nó không còn gì che.
            plate = rank == 0
            filled_rgb, band_alpha, dis_report = fill_layer_disocclusion(
                graded, mask, nearer, disocclusion, layer_index=rank, plate=plate
            )
            t_dis_total += (time.perf_counter() - t_dis0) * 1000.0
            # Tấm nền có thể có lỗ mà dải rỗng (fill_px nhỏ), nên cổng xét thêm.
            if dis_report.band_px or plate:
                fill_zone = band_alpha > 0
                rgba[fill_zone, :3] = filled_rgb[fill_zone]
                # Extend alpha outward; never reduce the silhouette's own alpha.
                rgba[..., 3] = np.maximum(rgba[..., 3], band_alpha)
                if plate:
                    # Mép feather của chính tấm nền cũng dưới 255; tấm nền phải
                    # đục toàn phần thì phía sau mới không còn vùng trống.
                    rgba[..., 3] = 255
                disocclusion_reports.append(dis_report)

        # PB-074 (đo): tô riêng lớp GẦN NHẤT để tìm lại đường bao của nó trên
        # khung hình. Không đặt HCPD_MARK_TOP thì không đụng gì.
        if rank == count - 1:
            paint_zone(rgba[..., :3], mask, top_layer_colour())

        t_comp0 = time.perf_counter()
        alpha_f = rgba[..., 3].astype(np.float32) / 255.0
        for ch in range(3):
            composite_rgb[..., ch] = (
                rgba[..., ch].astype(np.float32) * alpha_f
                + composite_rgb[..., ch] * (1.0 - alpha_f)
            )

        composite_ms += (time.perf_counter() - t_comp0) * 1000.0

        layers.append(
            DepthLayer(
                index=rank,
                mean_depth=float(member_depths.mean()),
                depth_lo=float(member_depths.min()),
                depth_hi=float(member_depths.max()),
                z_m=z,
                coverage=float(mask.mean()),
                mask=mask,
                rgba=rgba,
                depth_norm=depth_norm,
                grade_report=report,
            )

        )
    mist_color = compute_mist_color(layers[0]) if layers else MIST_COLOR_FALLBACK
    clustering_ms = (time.perf_counter() - t_clu0) * 1000.0
    return ClusterIntoLayersResult(
        layers=layers,
        semantic_report=semantic_report,
        mist_color=mist_color,
        timing_ms={
            "semantic_adjust": semantic_ms,
            "clustering": clustering_ms,
            "disocclusion_fill": t_dis_total,
            # Phan ra ben trong "clustering" de biet 4 giay di dau
            "_kmeans": kmeans_ms,
            "_masks": masks_ms,
            "_staging": staging_ms,
            "_rim": rim_ms,
            "_feather_rgba": feather_ms,
            "_composite": composite_ms,
        },
        disocclusion_reports=disocclusion_reports,
    )
