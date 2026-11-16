# GVF-threshold K-rule (2026-07-10):
# - Decision: smallest K with gvf_variance[K] >= GVF_THRESHOLD (default 0.90).
# - Silhouette removed from the hot path (was 40-111s O(n^2) freeze).
# - Optional LOG_SILHOUETTE_COMPARISON=False keeps thesis sil gated off by default.

"""Automatic cluster-count selection for depth-layer extraction."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import cv2
import numpy as np
import numpy.typing as npt

from src.content.layer_clustering import _kmeans_1d

logger = logging.getLogger(__name__)

_MAX_SAMPLE_PIXELS = 10_000
_FLAT_DEPTH_STD = 1e-4
_N_INIT = 4  # modest restarts for 1-D inertia; silhouette no longer drives selection

K_POLICY_DEFAULT = "gvf"
GVF_THRESHOLD = 0.90
LOG_SILHOUETTE_COMPARISON = False


@dataclass(frozen=True)
class AdaptiveKReport:
    """Result of one-shot K selection at image load."""

    chosen_k: int
    scores: dict[int, float]
    """Optional silhouette scores (empty unless LOG_SILHOUETTE_COMPARISON)."""

    flat_fallback: bool
    depth_std: float
    k_policy: str = K_POLICY_DEFAULT
    gvf_variance: dict[int, float] = field(default_factory=dict)
    """True GVF = 1 - WCSS/TSS per K."""

    gvf_threshold: float = GVF_THRESHOLD
    rejections: list[str] = field(default_factory=list)
    sample_count: int = 0
    kmeans_fit_count: int = 0
    timing_ms: dict[str, float] = field(default_factory=dict)

    @property
    def gvf(self) -> dict[int, float]:
        """Alias for overlay/back-compat — true variance GVF."""
        return self.gvf_variance


@dataclass
class _KMeansFitCounter:
    """Mutable counter for how many ``_kmeans_1d`` calls run in one selection."""

    count: int = 0


def _downsample_depth_values(depth: npt.NDArray[np.float32]) -> npt.NDArray[np.float64]:
    """Return at most ~10k depth samples for cheap scoring."""
    flat = depth.reshape(-1).astype(np.float64)
    if flat.size <= _MAX_SAMPLE_PIXELS:
        return flat
    h, w = depth.shape
    scale = float(np.sqrt(_MAX_SAMPLE_PIXELS / max(flat.size, 1)))
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    small = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return small.reshape(-1).astype(np.float64)


def _assign_labels_1d(values: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64]) -> npt.NDArray[np.int32]:
    return np.argmin(np.abs(values[:, None] - centroids[None, :]), axis=1).astype(np.int32)


def _silhouette_1d(values: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]) -> float:
    """Mean silhouette coefficient for 1-D clustered depth samples (optional / gated)."""
    n = values.shape[0]
    unique = np.unique(labels)
    if unique.size < 2 or n < 3:
        return 0.0

    scores = np.zeros(n, dtype=np.float64)
    for idx in range(n):
        same = labels == labels[idx]
        other_clusters = [c for c in unique if c != labels[idx]]
        if int(same.sum()) <= 1:
            a = 0.0
        else:
            a = float(np.abs(values[same] - values[idx]).sum() / (int(same.sum()) - 1))
        b = float(
            min(
                np.abs(values[labels == c] - values[idx]).mean()
                for c in other_clusters
            )
        )
        denom = max(a, b)
        scores[idx] = (b - a) / denom if denom > 1e-12 else 0.0
    return float(scores.mean())


def _fit_kmeans(
    values: npt.NDArray[np.float64],
    k: int,
    seed: int,
    counter: _KMeansFitCounter | None = None,
) -> npt.NDArray[np.float64]:
    if counter is not None:
        counter.count += 1
    return _kmeans_1d(values, k, seed=seed)


def _wcss(values: npt.NDArray[np.float64], centroids: npt.NDArray[np.float64], labels: npt.NDArray[np.int32]) -> float:
    total = 0.0
    for j, center in enumerate(centroids):
        members = values[labels == j]
        if members.size:
            total += float(np.sum((members - float(center)) ** 2))
    return total


def _compute_gvf_variance(
    values: npt.NDArray[np.float64],
    centroids: npt.NDArray[np.float64],
    labels: npt.NDArray[np.int32],
    tss: float,
) -> float:
    """True Goodness-of-Variance-Fit: ``1 - WCSS/TSS``."""
    if tss <= 1e-12:
        return 0.0
    return float(1.0 - _wcss(values, centroids, labels) / tss)


def _best_gvf_fit(
    samples: npt.NDArray[np.float64],
    k: int,
    seed: int,
    tss: float,
    *,
    n_init: int = _N_INIT,
    counter: _KMeansFitCounter | None = None,
) -> tuple[float, npt.NDArray[np.float64], npt.NDArray[np.int32]]:
    """Best GVF over ``n_init`` k-means restarts (pick lowest WCSS / highest GVF)."""
    runs = max(1, int(n_init))
    best_gvf = -1.0
    best_centroids = np.zeros(k, dtype=np.float64)
    best_labels = np.zeros(samples.shape[0], dtype=np.int32)
    for i in range(runs):
        centroids = _fit_kmeans(samples, k, seed=seed + i, counter=counter)
        labels = _assign_labels_1d(samples, centroids)
        gvf = _compute_gvf_variance(samples, centroids, labels, tss)
        if gvf > best_gvf:
            best_gvf = gvf
            best_centroids = centroids
            best_labels = labels
    return best_gvf, best_centroids, best_labels


def _assert_gvf_variance_monotonic(gvf_variance: dict[int, float]) -> None:
    """Log-only check: true GVF should be non-decreasing in K."""
    keys = sorted(gvf_variance)
    for prev_k, k in zip(keys, keys[1:]):
        if gvf_variance[k] + 1e-9 < gvf_variance[prev_k]:
            logger.warning(
                "gvf_variance non-monotonic at K=%d: %.4f < gvf_var[K=%d]=%.4f "
                "(possible KMeans issue — not clamped)",
                k,
                gvf_variance[k],
                prev_k,
                gvf_variance[prev_k],
            )


def _choose_k_gvf_threshold(
    gvf_variance: dict[int, float],
    *,
    k_min: int,
    k_max: int,
    threshold: float,
) -> tuple[int, list[str]]:
    """
    Jenks-style rule: smallest K with gvf_variance[K] >= threshold; else k_max.

    Used for both ``gvf`` / ``richness`` / ``parsimony`` policy aliases — one correct
    rule until a distinct richness variant is specified.
    """
    rejections: list[str] = []
    for k in range(k_min, k_max + 1):
        gvf = float(gvf_variance.get(k, 0.0))
        if gvf >= threshold:
            return k, rejections
        rejections.append(f"K={k}: gvf_var={gvf:.3f}<{threshold:.2f}")
    return k_max, rejections


def _log_adaptive_k(report: AdaptiveKReport) -> None:
    gvf_var = {k: round(v, 3) for k, v in sorted(report.gvf_variance.items())}
    k_ms = float(report.timing_ms.get("total_k_selection", 0.0))
    logger.info(
        "Adaptive K [gvf]: selected K=%d | gvf_var=%s | threshold=%.2f | "
        "flat_fallback=%s | k_selection=%.1fms",
        report.chosen_k,
        gvf_var,
        report.gvf_threshold,
        report.flat_fallback,
        k_ms,
    )
    if report.scores:
        sil = {k: round(v, 3) for k, v in sorted(report.scores.items())}
        logger.info("  silhouette comparison (gated): sil=%s", sil)
    if report.rejections:
        logger.info("  gvf rejections: %s", "; ".join(report.rejections))


def select_optimal_k(
    depth_map: npt.NDArray[np.float32],
    k_min: int = 2,
    k_max: int = 4,
    *,
    seed: int = 0,
    k_policy: str = K_POLICY_DEFAULT,
    gvf_threshold: float = GVF_THRESHOLD,
    log_silhouette_comparison: bool = LOG_SILHOUETTE_COMPARISON,
) -> tuple[int, AdaptiveKReport]:
    """
    Return optimal cluster count for depth layering via GVF threshold.

    Fits 1-D k-means on a downsampled depth sample (~10k), computes
    ``gvf_variance = 1 - WCSS/TSS``, and picks the smallest K whose GVF
    meets ``gvf_threshold``. Silhouette is not used on the hot path.

    The flat-scene fallback keys off ``_FLAT_DEPTH_STD`` and near-zero TSS, not
    a caller-supplied threshold.  Parameters from the pre-GVF rule
    (``tolerance``, ``flat_threshold``, ``min_layer_coverage``,
    ``min_depth_separation``, ``min_gvf_gain``) were accepted-then-discarded,
    which made the matching content.json keys silently inert; they have been
    removed rather than left as a trap.
    """
    t0 = time.perf_counter()
    k_min = max(2, int(k_min))
    k_max = max(k_min, int(k_max))
    policy = (k_policy or K_POLICY_DEFAULT).strip().lower()
    # richness/parsimony aliases map to the same GVF-threshold rule for now.
    if policy in ("richness", "parsimony", "gvf"):
        policy = "gvf"
    else:
        logger.warning("Unknown k_policy '%s' — using gvf threshold", k_policy)
        policy = "gvf"

    threshold = float(gvf_threshold)
    counter = _KMeansFitCounter()
    samples = _downsample_depth_values(depth_map)
    depth_std = float(samples.std())
    sample_count = int(samples.size)

    if depth_std < _FLAT_DEPTH_STD:
        total_ms = (time.perf_counter() - t0) * 1000.0
        report = AdaptiveKReport(
            chosen_k=k_min,
            scores={},
            flat_fallback=True,
            depth_std=depth_std,
            k_policy=policy,
            gvf_variance={k: 0.0 for k in range(k_min, k_max + 1)},
            gvf_threshold=threshold,
            rejections=[f"flat depth std={depth_std:.2e}"],
            sample_count=sample_count,
            kmeans_fit_count=0,
            timing_ms={"total_k_selection": total_ms},
        )
        _log_adaptive_k(report)
        return k_min, report

    mean = float(samples.mean())
    tss = float(np.sum((samples - mean) ** 2))
    if tss <= 1e-12:
        total_ms = (time.perf_counter() - t0) * 1000.0
        report = AdaptiveKReport(
            chosen_k=k_min,
            scores={},
            flat_fallback=True,
            depth_std=depth_std,
            k_policy=policy,
            gvf_variance={k: 0.0 for k in range(k_min, k_max + 1)},
            gvf_threshold=threshold,
            rejections=["flat TSS near zero"],
            sample_count=sample_count,
            kmeans_fit_count=0,
            timing_ms={"total_k_selection": total_ms},
        )
        _log_adaptive_k(report)
        return k_min, report

    gvf_variance: dict[int, float] = {}
    best_labels: dict[int, npt.NDArray[np.int32]] = {}
    for k in range(k_min, k_max + 1):
        gvf, _centroids, labels = _best_gvf_fit(
            samples, k, seed=seed, tss=tss, n_init=_N_INIT, counter=counter
        )
        gvf_variance[k] = gvf
        best_labels[k] = labels

    _assert_gvf_variance_monotonic(gvf_variance)

    chosen, rejections = _choose_k_gvf_threshold(
        gvf_variance, k_min=k_min, k_max=k_max, threshold=threshold
    )

    scores: dict[int, float] = {}
    if log_silhouette_comparison:
        for k, labels in best_labels.items():
            scores[k] = _silhouette_1d(samples, labels)

    total_ms = (time.perf_counter() - t0) * 1000.0
    logger.info(
        "  k_selection detail: kmeans_fits=%d | samples=%d | total=%.1fms",
        counter.count,
        sample_count,
        total_ms,
    )

    report = AdaptiveKReport(
        chosen_k=chosen,
        scores=scores,
        flat_fallback=False,
        depth_std=depth_std,
        k_policy=policy,
        gvf_variance=gvf_variance,
        gvf_threshold=threshold,
        rejections=rejections,
        sample_count=sample_count,
        kmeans_fit_count=counter.count,
        timing_ms={"total_k_selection": total_ms},
    )
    _log_adaptive_k(report)
    return chosen, report
