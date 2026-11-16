"""Depth-gradient seam weld weights for Phase 2 soft boundaries.

For each pixel on a layer–layer seam, produce ``w ∈ [0, 1]``:
- ``w = 1`` → full Z-weld (smooth depth transition)
- ``w = 0`` → hard step (occlusion)

Depth gradient decides; RGB silhouette edges only confirm hard seams when
depth already shows a discontinuity. Decision uses the raw depth map
(pre-semantic), never the flattened post-semantic depth.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

from src.content.layer_clustering import classify_silhouette_edges

logger = logging.getLogger(__name__)

# Absolute Sobel magnitude on depth∈[0,1] (see :func:`depth_gradient_magnitude`).
DEPTH_EDGE_LOW = 0.05
DEPTH_EDGE_HIGH = 0.20
RGB_EDGE_THRESHOLD = 0.04


@dataclass(frozen=True)
class SeamStats:
    """Aggregate weld-weight breakdown for one layer–layer seam."""

    layer_a: int
    layer_b: int
    length_px: int
    hard_frac: float
    soft_frac: float
    mixed_frac: float
    mean_w: float


@dataclass(frozen=True)
class BoundaryClassifierResult:
    """Per-load seam classification (no welding applied yet)."""

    weld_weight: npt.NDArray[np.float32]
    """HxW map; non-seam pixels are NaN, seam pixels hold w in [0, 1]."""

    depth_grad_norm: npt.NDArray[np.float32]
    """HxW normalized depth-gradient magnitude used for the decision."""

    seams: list[SeamStats]
    depth_edge_low: float = DEPTH_EDGE_LOW
    depth_edge_high: float = DEPTH_EDGE_HIGH


def depth_gradient_magnitude(depth: npt.NDArray[np.float32]) -> npt.NDArray[np.float32]:
    """
    Absolute Sobel magnitude of ``depth`` (assumed in ``[0, 1]``).

    No per-image rescaling: a unit depth step yields mag ≳ 1, while a gentle
    ramp across tens of pixels yields mag ≪ ``DEPTH_EDGE_LOW`` (0.05). That
    keeps ``DEPTH_EDGE_LOW/HIGH`` absolute and comparable across images.
    """
    d = np.asarray(depth, dtype=np.float32)
    gx = cv2.Sobel(d, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(d, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np.ascontiguousarray(mag, dtype=np.float32)


def weld_weight_from_depth_grad(
    depth_grad_norm: npt.NDArray[np.float32],
    *,
    low: float = DEPTH_EDGE_LOW,
    high: float = DEPTH_EDGE_HIGH,
) -> npt.NDArray[np.float32]:
    """
    Map normalized depth gradient → weld weight.

    ``grad < low`` → 1.0, ``grad > high`` → 0.0, linear ramp between.
    """
    g = np.asarray(depth_grad_norm, dtype=np.float32)
    lo = float(low)
    hi = max(float(high), lo + 1e-6)
    w = np.ones_like(g, dtype=np.float32)
    mid = (g >= lo) & (g <= hi)
    w[g > hi] = 0.0
    w[mid] = (hi - g[mid]) / (hi - lo)
    return np.clip(w, 0.0, 1.0)


def _rgb_edge_map(
    rgb: npt.NDArray[np.uint8],
    *,
    threshold: float = RGB_EDGE_THRESHOLD,
) -> npt.NDArray[np.bool_]:
    """Full-image RGB Sobel edge mask (tests / fallback without a layer mask)."""
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    return grad >= float(threshold)


def _seam_mask(
    mask_a: npt.NDArray[np.bool_], mask_b: npt.NDArray[np.bool_]
) -> npt.NDArray[np.bool_]:
    """Pixels of A∪B that touch the other layer (4-connected via 3×3 dilate)."""
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    a = np.where(mask_a, np.uint8(255), np.uint8(0))
    b = np.where(mask_b, np.uint8(255), np.uint8(0))
    a_touch = mask_a & (cv2.dilate(b, kernel) > 0)
    b_touch = mask_b & (cv2.dilate(a, kernel) > 0)
    return a_touch | b_touch


def _seam_stats(
    layer_a: int,
    layer_b: int,
    seam: npt.NDArray[np.bool_],
    weld: npt.NDArray[np.float32],
) -> SeamStats | None:
    if not bool(seam.any()):
        return None
    vals = weld[seam]
    n = int(vals.size)
    hard = float((vals <= 0.25).mean())
    soft = float((vals >= 0.75).mean())
    mixed = float(1.0 - hard - soft)
    return SeamStats(
        layer_a=layer_a,
        layer_b=layer_b,
        length_px=n,
        hard_frac=hard,
        soft_frac=soft,
        mixed_frac=max(0.0, mixed),
        mean_w=float(vals.mean()),
    )


def classify_layer_seams(
    depth: npt.NDArray[np.float32],
    layer_masks: list[npt.NDArray[np.bool_]],
    rgb: npt.NDArray[np.uint8] | None = None,
    *,
    depth_edge_low: float = DEPTH_EDGE_LOW,
    depth_edge_high: float = DEPTH_EDGE_HIGH,
    rgb_edge_threshold: float = RGB_EDGE_THRESHOLD,
) -> BoundaryClassifierResult:
    """
    Classify every inter-layer seam pixel by depth gradient (optional RGB confirm).

    ``depth`` must be the raw Depth-Anything map (pre-semantic). ``layer_masks``
    are the clustered layer silhouettes (far→near order). Returns a weld-weight
    map (NaN off-seam) and per-seam statistics for logging / weldmaps.
    """
    if depth.ndim != 2:
        raise ValueError(f"Expected 2-D depth, got {depth.shape}")
    h, w = depth.shape
    for i, mask in enumerate(layer_masks):
        if mask.shape != (h, w):
            raise ValueError(f"mask[{i}] shape {mask.shape} != depth {(h, w)}")

    grad = depth_gradient_magnitude(depth)
    base_w = weld_weight_from_depth_grad(
        grad, low=depth_edge_low, high=depth_edge_high
    )

    rgb_edges: npt.NDArray[np.bool_] | None = None
    if rgb is not None:
        if rgb.shape[:2] != (h, w):
            raise ValueError("rgb and depth size mismatch")
        # Reuse existing RGB silhouette detector per layer; OR across layers.
        rgb_edges = np.zeros((h, w), dtype=bool)
        for mask in layer_masks:
            rgb_edges |= classify_silhouette_edges(
                mask, rgb, gradient_threshold=rgb_edge_threshold
            )

    weld = np.full((h, w), np.nan, dtype=np.float32)
    seams: list[SeamStats] = []
    n_layers = len(layer_masks)

    for i in range(n_layers):
        for j in range(i + 1, n_layers):
            seam = _seam_mask(layer_masks[i], layer_masks[j])
            if not bool(seam.any()):
                continue
            w_vals = base_w[seam].copy()
            if rgb_edges is not None:
                # Depth decides; RGB only confirms hard when depth already elevated.
                confirm = seam & rgb_edges & (grad > depth_edge_low)
                w_vals = np.where(confirm[seam], 0.0, w_vals)
            weld[seam] = w_vals
            stats = _seam_stats(i, j, seam, weld)
            if stats is not None:
                seams.append(stats)
                logger.info(
                    "Seam %d|%d: hard=%.0f%% soft=%.0f%% mixed=%.0f%% length=%dpx mean_w=%.2f",
                    stats.layer_a,
                    stats.layer_b,
                    stats.hard_frac * 100.0,
                    stats.soft_frac * 100.0,
                    stats.mixed_frac * 100.0,
                    stats.length_px,
                    stats.mean_w,
                )

    return BoundaryClassifierResult(
        weld_weight=weld,
        depth_grad_norm=grad,
        seams=seams,
        depth_edge_low=float(depth_edge_low),
        depth_edge_high=float(depth_edge_high),
    )


def render_weldmap(
    rgb: npt.NDArray[np.uint8],
    result: BoundaryClassifierResult,
    *,
    seam_alpha: float = 0.75,
) -> npt.NDArray[np.uint8]:
    """
    Overlay seam weld weights on the source image.

    Red = hard (w≈0), green = soft weld (w≈1), yellow = ramp.
    """
    base = rgb.astype(np.float32).copy()
    wmap = result.weld_weight
    seam = np.isfinite(wmap)
    if not bool(seam.any()):
        return np.ascontiguousarray(rgb, dtype=np.uint8)

    # Widen the seam so it is visible in the overlay, carrying each pixel's
    # weight to its NEAREST seam pixel.
    #
    # This used to `cv2.dilate` the weight map, but dilation is a local MAXIMUM:
    # it propagated the softest weight outward and painted the whole band green,
    # making every weldmap look far more weldable than the classifier said. A
    # nearest-seam lookup is unbiased.
    ys, xs = np.nonzero(seam)
    inverted = np.where(seam, np.uint8(0), np.uint8(255))
    dist, labels = cv2.distanceTransformWithLabels(
        inverted, cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL
    )
    # DIST_LABEL_PIXEL numbers the zero pixels 1..N in row-major order, which is
    # the order np.nonzero returns — so a leading 0 makes the LUT 1-indexed.
    lut = np.concatenate(([np.float32(0.0)], wmap[ys, xs].astype(np.float32)))
    w_vis = lut[labels]
    wide = dist <= 2.5

    ww = np.clip(np.nan_to_num(w_vis, nan=0.0), 0.0, 1.0)
    # green * w + red * (1-w); yellow in between
    color = np.zeros_like(base)
    color[..., 0] = 255.0 * (1.0 - ww)  # R hard
    color[..., 1] = 255.0 * ww  # G soft
    color[..., 2] = 40.0 * ww * (1.0 - ww)  # slight yellow mid

    a = float(np.clip(seam_alpha, 0.0, 1.0))
    out = base.copy()
    out[wide] = out[wide] * (1.0 - a) + color[wide] * a
    return np.ascontiguousarray(np.clip(out, 0, 255).astype(np.uint8))


# Re-export for callers that want RGB silhouette confirmation on a layer mask.
__all__ = [
    "DEPTH_EDGE_LOW",
    "DEPTH_EDGE_HIGH",
    "RGB_EDGE_THRESHOLD",
    "SeamStats",
    "BoundaryClassifierResult",
    "depth_gradient_magnitude",
    "weld_weight_from_depth_grad",
    "classify_layer_seams",
    "render_weldmap",
    "classify_silhouette_edges",
]
