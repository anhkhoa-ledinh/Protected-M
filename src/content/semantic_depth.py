"""Semantic-guided depth adjustment before layer clustering.

Inspired by semantic-aware depth pipelines (Niklaus et al. 2019 — 3D Ken Burns;
SLIDE 2021). Monocular depth alone often places semantically distinct objects
(sky vs village, hair vs cheek) at nearly identical depths. We segment the
image once per load, flatten coherent regions to a shared depth, and push
adjacent similar-depth segments apart so histogram / k-means clustering assigns
each object its own layer.

Segmentation is CPU-friendly OpenCV only: depth-band markers + watershed on
RGB/depth gradients (no training, no SAM dependency).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import cv2
import numpy as np
import numpy.typing as npt

logger = logging.getLogger(__name__)

_WATERSHED_BORDER = -1
_WATERSHED_BG = 0


@dataclass(frozen=True)
class SemanticDepthParams:
    """Knobs for pre-clustering semantic depth adjustment."""

    enabled: bool = True
    depth_band_count: int = 6
    """Coarse depth bands used to seed region markers."""

    similarity_threshold: float = 0.06
    """Adjacent segments with median depth within this gap are pushed apart."""

    separation_offset: float = 0.08
    """Depth delta applied between separated adjacent segments."""

    min_separation_vertical_gap: float = 0.10
    """Only push apart similar-depth segments separated this much in Y (0-1)."""

    min_separation_horizontal_span: float = 0.6
    """
    Both segments must span at least this fraction of the frame width before
    they may be pushed apart.

    The separation rule encodes landscape semantics — sky above village above
    road — where every band stretches across the frame.  A person is *also* a
    vertically extended object with torso above legs at similar depth, so
    without this guard the rule fires on them too and slices the figure in half
    at a cluster boundary.  Landscape bands span ~0.9 of the width; a standing
    figure spans ~0.2-0.4.  Set to 0.0 to restore the old behaviour.
    """

    max_segment_variance: float = 0.02
    """Segments with internal depth variance below this are flattened to median."""

    min_segment_pixels: int = 150
    """Drop or merge segments smaller than this."""

    subdivide_above_pixels: int = 8_000
    """Large depth-band blobs are split further via watershed."""

    max_work_dim: int = 640
    """Downscale longest side for one-shot segmentation (full-res labels upscaled)."""


@dataclass(frozen=True)
class SemanticDepthReport:
    """Summary of semantic depth adjustment for logging / debug overlay."""

    num_segments: int
    segments_flattened: int
    segment_pairs_separated: int
    depth_range_before: tuple[float, float]
    depth_range_after: tuple[float, float]


@dataclass(frozen=True)
class _SegmentInfo:
    seg_id: int
    median: float
    variance: float
    centroid_y: float
    pixel_count: int
    horizontal_span: float = 0.0
    """Width of the segment's bounding box as a fraction of the frame (0-1)."""


def _depth_range(depth: npt.NDArray[np.float32]) -> tuple[float, float]:
    return float(depth.min()), float(depth.max())


def _resize_work(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.float32],
    max_dim: int,
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.float32], float]:
    h, w = depth.shape
    longest = max(h, w)
    if longest <= max_dim:
        return rgb, depth, 1.0
    scale = max_dim / float(longest)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small_rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
    small_depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return small_rgb, small_depth, scale


def _combined_gradient(
    rgb: npt.NDArray[np.uint8], depth: npt.NDArray[np.float32]
) -> npt.NDArray[np.float32]:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad = cv2.magnitude(gx, gy)
    dgx = cv2.Sobel(depth, cv2.CV_32F, 1, 0, ksize=3)
    dgy = cv2.Sobel(depth, cv2.CV_32F, 0, 1, ksize=3)
    dgrad = cv2.magnitude(dgx, dgy)
    grad = cv2.normalize(grad, None, 0.0, 1.0, cv2.NORM_MINMAX)
    dgrad = cv2.normalize(dgrad, None, 0.0, 1.0, cv2.NORM_MINMAX)
    return grad * 0.72 + dgrad * 0.28


def _depth_band_labels(depth: npt.NDArray[np.float32], n_bands: int) -> npt.NDArray[np.int32]:
    edges = np.percentile(
        depth, np.linspace(0.0, 100.0, max(2, int(n_bands)) + 1)
    ).astype(np.float32)
    edges = np.unique(edges)
    if edges.size < 2:
        return np.zeros(depth.shape, dtype=np.int32)
    # Interior edges only — digitize assigns 0..len(edges)-2.
    labels = np.digitize(depth, edges[1:-1], right=False).astype(np.int32)
    return labels


def _watershed_subdivide(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.float32],
    region_mask: npt.NDArray[np.bool_],
    gradient: npt.NDArray[np.float32],
    *,
    min_pixels: int,
) -> list[npt.NDArray[np.bool_]]:
    """Split a large region into sub-masks using watershed on combined gradient."""
    if int(region_mask.sum()) < 2 * min_pixels:
        return [region_mask]

    h, w = depth.shape
    elevation = np.where(region_mask, gradient, 1.0).astype(np.float32)
    elevation = cv2.normalize(elevation, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    region_u8 = region_mask.astype(np.uint8)
    sure_fg = cv2.erode(region_u8, np.ones((3, 3), np.uint8), iterations=2)
    num_cc, cc = cv2.connectedComponents(sure_fg)
    markers = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for cc_id in range(1, num_cc):
        cc_mask = cc == cc_id
        if int(cc_mask.sum()) >= min_pixels // 2:
            markers[cc_mask] = next_id
            next_id += 1

    if next_id == 1:
        rows = np.where(region_mask)[0]
        if rows.size == 0:
            return [region_mask]
        splits = np.percentile(rows, [33.0, 66.0])
        sub_masks: list[npt.NDArray[np.bool_]] = []
        y = np.arange(h, dtype=np.float32)[:, None]
        bounds = [float(rows.min()), float(splits[0]), float(splits[1]), float(rows.max()) + 1.0]
        for lo, hi in zip(bounds[:-1], bounds[1:]):
            sub = region_mask & (y >= lo) & (y < hi)
            if int(sub.sum()) >= min_pixels // 2:
                sub_masks.append(sub)
        return sub_masks or [region_mask]

    unknown = region_u8.copy()
    markers[unknown == 0] = _WATERSHED_BG
    # OpenCV watershed expects a 3-channel image.
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    cv2.watershed(bgr, markers)

    sub_masks: list[npt.NDArray[np.bool_]] = []
    for marker_id in range(1, next_id):
        sub = (markers == marker_id) & region_mask
        if int(sub.sum()) >= min_pixels // 2:
            sub_masks.append(sub)
    return sub_masks or [region_mask]


def segment_image(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.float32],
    params: SemanticDepthParams,
) -> npt.NDArray[np.int32]:
    """
    Segment an image into depth-coherent regions (GrabCut-free, OpenCV only).

    Pipeline: coarse depth bands -> connected components -> watershed split on
    large blobs using RGB + depth gradient barriers.
    """
    small_rgb, small_depth, scale = _resize_work(rgb, depth, params.max_work_dim)
    h, w = small_depth.shape
    gradient = _combined_gradient(small_rgb, small_depth)
    band_labels = _depth_band_labels(small_depth, params.depth_band_count)

    segments = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    min_px = max(32, int(params.min_segment_pixels))

    for band in range(int(band_labels.max()) + 1):
        band_mask = band_labels == band
        if not bool(band_mask.any()):
            continue
        band_u8 = band_mask.astype(np.uint8)
        num_cc, cc = cv2.connectedComponents(band_u8)
        for cc_id in range(1, num_cc):
            cc_mask = cc == cc_id
            count = int(cc_mask.sum())
            if count < min_px:
                continue
            if count >= int(params.subdivide_above_pixels):
                for sub in _watershed_subdivide(
                    small_rgb,
                    small_depth,
                    cc_mask,
                    gradient,
                    min_pixels=min_px,
                ):
                    segments[sub] = next_id
                    next_id += 1
            else:
                segments[cc_mask] = next_id
                next_id += 1

    segments = _merge_small_segments(segments, min_px)
    unique_seg = [int(v) for v in np.unique(segments) if v > 0]
    if len(unique_seg) < 2:
        segments = _spatial_depth_grid_segments(small_depth, min_px)
    if scale < 1.0:
        full_h, full_w = depth.shape
        segments = cv2.resize(
            segments.astype(np.float32),
            (full_w, full_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(np.int32)
    return segments


def _merge_small_segments(
    segments: npt.NDArray[np.int32], min_pixels: int
) -> npt.NDArray[np.int32]:
    """Merge undersized segments into the largest adjacent label."""
    out = segments.copy()
    for seg_id in sorted(int(v) for v in np.unique(out) if v > 0):
        mask = out == seg_id
        if int(mask.sum()) >= min_pixels:
            continue
        dilated = cv2.dilate(
            mask.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=1
        )
        neighbors = out[dilated.astype(bool) & ~mask]
        neighbors = neighbors[(neighbors > 0) & (neighbors != seg_id)]
        if neighbors.size == 0:
            continue
        target = int(np.bincount(neighbors.astype(np.int64)).argmax())
        out[mask] = target
    return out


def _spatial_depth_grid_segments(
    depth: npt.NDArray[np.float32], min_pixels: int
) -> npt.NDArray[np.int32]:
    """
    Fallback markers from depth x vertical grid when watershed finds one blob.

    Separates sky / mid-ground / foreground even when depth values collapse.
    """
    h, w = depth.shape
    n_y, n_d = 4, 4
    d_edges = np.percentile(depth, np.linspace(8.0, 92.0, n_d + 1))
    d_bins = np.digitize(depth, d_edges[1:-1], right=False)
    y_bins = (np.arange(h, dtype=np.int32)[:, None] * n_y // max(h, 1))
    coarse = (d_bins * n_y + y_bins).astype(np.int32)
    segments = np.zeros((h, w), dtype=np.int32)
    next_id = 1
    for label in sorted(int(v) for v in np.unique(coarse)):
        mask = coarse == label
        if int(mask.sum()) < min_pixels:
            continue
        segments[mask] = next_id
        next_id += 1
    return segments


def _collect_segments(
    depth: npt.NDArray[np.float32], segments: npt.NDArray[np.int32]
) -> list[_SegmentInfo]:
    """
    Per-segment statistics.

    MEASURED, do not "optimise" again without re-measuring: replacing this loop
    with scipy.ndimage.median/variance/find_objects made it 2x SLOWER on a
    2 MP image (110 ms -> 353 ms) and no faster on 4.75 MP. With only 4-30
    labels, numpy's vectorised `==` beats scipy's per-label aggregation.
    """
    infos: list[_SegmentInfo] = []
    h, w = depth.shape[:2]
    for seg_id in sorted(int(v) for v in np.unique(segments) if v > 0):
        mask = segments == seg_id
        vals = depth[mask].astype(np.float64)
        rows, cols = np.where(mask)
        infos.append(
            _SegmentInfo(
                seg_id=seg_id,
                median=float(np.median(vals)),
                variance=float(vals.var()) if vals.size > 1 else 0.0,
                centroid_y=float(rows.mean() / max(h - 1, 1)),
                pixel_count=int(mask.sum()),
                horizontal_span=float(
                    (int(cols.max()) - int(cols.min()) + 1) / max(w, 1)
                ),
            )
        )
    return infos


def _adjacent_segment_pairs(
    segments: npt.NDArray[np.int32],
) -> set[tuple[int, int]]:
    """Return unordered pairs of segment ids that share a boundary."""
    pairs: set[tuple[int, int]] = set()
    for shift in ((0, 1), (1, 0)):
        a = segments
        b = np.roll(segments, shift=shift, axis=shift[0])
        diff = (a != b) & (a > 0) & (b > 0)
        if not bool(diff.any()):
            continue
        id_a = a[diff]
        id_b = b[diff]
        # Vectorised: the Python loop here ran once per boundary pixel, which on
        # a 4.75 MP image with 30 segments is hundreds of thousands of iterations.
        lo = np.minimum(id_a, id_b)
        hi = np.maximum(id_a, id_b)
        for left, right in np.unique(np.stack([lo, hi], axis=1), axis=0).tolist():
            pairs.add((int(left), int(right)))
    return pairs


def _flatten_segments(
    depth: npt.NDArray[np.float32],
    segments: npt.NDArray[np.int32],
    max_variance: float,
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Snap *coherent* segments to their median depth (face+body unity).

    Only low-variance segments are flattened.  A segment whose internal depth
    varies a lot carries real structure — a receding wall, a sloping road — and
    collapsing it to one value would throw that away.

    NOTE: this used to flatten every segment unconditionally.  The old branch
    read ``if info.variance > max_variance: median = np.median(depth[mask])
    else: median = info.median``, but ``info.median`` IS ``np.median(depth[mask])``
    — both arms computed the same number, so ``max_segment_variance`` had no
    effect and ``flattened`` always equalled the segment count.
    """
    infos = _collect_segments(depth, segments)
    if not infos:
        return depth.copy(), 0

    # Lookup table indexed by segment id: one vectorised pass instead of one
    # full-image boolean comparison per segment.
    size = max(int(segments.max()), max(i.seg_id for i in infos)) + 1
    lut = np.zeros(size, dtype=np.float32)
    touched = np.zeros(size, dtype=bool)
    flattened = 0
    for info in infos:
        if info.variance > max_variance:
            continue
        lut[info.seg_id] = np.float32(info.median)
        touched[info.seg_id] = True
        flattened += 1

    if flattened == 0:
        return depth.copy(), 0
    idx = np.clip(segments, 0, size - 1)
    adjusted = np.where(touched[idx], lut[idx], depth).astype(np.float32)
    return adjusted, flattened


def _separate_similar_adjacent_segments(
    depth: npt.NDArray[np.float32],
    segments: npt.NDArray[np.int32],
    *,
    similarity_threshold: float,
    separation_offset: float,
    min_vertical_gap: float,
    min_horizontal_span: float = 0.0,
) -> tuple[npt.NDArray[np.float32], int]:
    """
    Push apart adjacent segments whose medians are too close in depth.

    Ordering uses vertical centroid (top = farther / lower depth value) following
    landscape semantics — sky above village above road.
    """
    adjusted = depth.copy()
    infos = {info.seg_id: info for info in _collect_segments(adjusted, segments)}
    pairs = _adjacent_segment_pairs(segments)
    if not pairs:
        return adjusted, 0

    parent: dict[int, int] = {sid: sid for sid in infos}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for left, right in pairs:
        li, ri = infos.get(left), infos.get(right)
        if li is None or ri is None:
            continue
        if abs(li.median - ri.median) > similarity_threshold:
            continue
        if abs(li.centroid_y - ri.centroid_y) < min_vertical_gap:
            continue
        # Landscape bands stretch across the frame; the torso and legs of a
        # standing figure do not.  Without this the rule slices people in half.
        if (
            li.horizontal_span < min_horizontal_span
            or ri.horizontal_span < min_horizontal_span
        ):
            continue
        union(left, right)

    groups: dict[int, list[_SegmentInfo]] = {}
    for info in infos.values():
        groups.setdefault(find(info.seg_id), []).append(info)

    size = int(segments.max()) + 1
    lut = np.zeros(size, dtype=np.float32)
    touched = np.zeros(size, dtype=bool)
    separated_pairs = 0
    for group in groups.values():
        if len(group) < 2:
            continue
        group.sort(key=lambda s: s.centroid_y)
        n = len(group)
        base = float(np.median([s.median for s in group]))
        for i, info in enumerate(group):
            # Top of frame (small y) -> farther -> lower depth (0 = far).
            target = base + (i - (n - 1) / 2.0) * separation_offset
            target = float(np.clip(target, 0.0, 1.0))
            if info.seg_id < size:
                lut[info.seg_id] = np.float32(target)
                touched[info.seg_id] = True
        separated_pairs += n - 1

    if separated_pairs:
        idx = np.clip(segments, 0, size - 1)
        adjusted = np.where(touched[idx], lut[idx], adjusted).astype(np.float32)
    return adjusted, separated_pairs


def adjust_depth_semantic(
    rgb: npt.NDArray[np.uint8],
    depth: npt.NDArray[np.float32],
    params: SemanticDepthParams,
) -> tuple[npt.NDArray[np.float32], SemanticDepthReport]:
    """
    Adjust a depth map using lightweight segmentation before layer clustering.

    1. Segment via depth-band watershed (CPU, one-shot).
    2. Flatten low-variance segments to median depth (face+body coherence).
    3. Push apart adjacent segments with similar depth (village vs sky).
    """
    if not params.enabled:
        lo, hi = _depth_range(depth)
        return depth, SemanticDepthReport(0, 0, 0, (lo, hi), (lo, hi))

    range_before = _depth_range(depth)
    _t = time.perf_counter()
    segments = segment_image(rgb, depth, params)
    _ms_segment = (time.perf_counter() - _t) * 1000.0

    _t = time.perf_counter()
    num_segments = len({int(v) for v in np.unique(segments) if v > 0})
    _ms_count = (time.perf_counter() - _t) * 1000.0

    _t = time.perf_counter()
    adjusted, flattened = _flatten_segments(
        depth, segments, params.max_segment_variance
    )
    _ms_flatten = (time.perf_counter() - _t) * 1000.0

    _t = time.perf_counter()
    adjusted, separated = _separate_similar_adjacent_segments(
        adjusted,
        segments,
        similarity_threshold=params.similarity_threshold,
        separation_offset=params.separation_offset,
        min_vertical_gap=params.min_separation_vertical_gap,
        min_horizontal_span=params.min_separation_horizontal_span,
    )
    _ms_separate = (time.perf_counter() - _t) * 1000.0
    logger.info(
        "  semantic breakdown: segment=%.0fms count_unique=%.0fms flatten=%.0fms "
        "separate=%.0fms  (segments=%d)",
        _ms_segment, _ms_count, _ms_flatten, _ms_separate, num_segments,
    )
    range_after = _depth_range(adjusted)

    report = SemanticDepthReport(
        num_segments=num_segments,
        segments_flattened=flattened,
        segment_pairs_separated=separated,
        depth_range_before=range_before,
        depth_range_after=range_after,
    )
    logger.info(
        "Semantic depth adjust: segments=%d flattened=%d separated=%d "
        "depth [%.3f,%.3f] -> [%.3f,%.3f]",
        num_segments,
        flattened,
        separated,
        range_before[0],
        range_before[1],
        range_after[0],
        range_after[1],
    )
    return np.ascontiguousarray(adjusted, dtype=np.float32), report
